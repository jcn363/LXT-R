//! Standalone LTX-2.3 inference demo.
//!
//! Demonstrates the full video diffusion pipeline:
//! 1. Initialize model with random or loaded weights
//! 2. Create noisy video latent
//! 3. Run denoising loop with scheduler + transformer + CFG
//! 4. Output denoised latent
//!
//! Usage:
//!   cargo run --bin ltx-inference -- --steps 4
//!   cargo run --bin ltx-inference -- --weights path/to/model.safetensors --steps 20
//!   cargo run --bin ltx-inference -- --weights path/to/model.safetensors --height 32 --width 32 --frames 4

use ltx_attention::RopeType;
use ltx_components::{EulerStep, Ltx2Scheduler, CFG};
use ltx_norm::RMSNorm;
use ltx_patchify::{patchify_5d, unpatchify_5d};
use ltx_transformer::block::BasicAVTransformerBlock;
use ltx_transformer::model::LTXModel;
use ltx_types::{Guider, Scheduler, STABILITY_EPS};
use tch::nn::ModuleT;
use tch::{Device, Kind, Tensor};

fn parse_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn parse_i64(args: &[String], flag: &str, default: i64) -> i64 {
    parse_arg(args, flag)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn build_model(vs: &tch::nn::Path, dim: i64, patch_dim: i64, num_layers: i64) -> LTXModel {
    let mut blocks = Vec::new();
    for i in 0..num_layers {
        blocks.push(BasicAVTransformerBlock::new(
            &(vs / "blocks" / i),
            dim,
            4,
            dim / 4,
            None,
            RopeType::Interleaved,
        ));
    }
    let norm_out = RMSNorm::default_eps_with_path(vs / "norm_out", dim);
    let proj_out = tch::nn::linear(vs / "proj_out", dim, patch_dim, Default::default());
    LTXModel::new(blocks, norm_out, proj_out)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // CLI args
    let n_steps = parse_i64(&args, "--steps", 20) as usize;
    let weights_path = parse_arg(&args, "--weights");
    let prompt = parse_arg(&args, "--prompt")
        .unwrap_or_else(|| "a colorful abstract pattern".to_string());
    let h = parse_i64(&args, "--height", 16);
    let w = parse_i64(&args, "--width", 16);
    let f = parse_i64(&args, "--frames", 4);
    let cfg_scale: f64 = parse_arg(&args, "--cfg")
        .and_then(|s| s.parse().ok())
        .unwrap_or(7.5);

    let use_random = weights_path.is_none();
    let mode = if use_random { "Random Weights" } else { "Loaded Weights" };

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           LTX-2.3 Rust Inference Demo ({:<20}) ║", mode);
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Configuration
    let (b, c) = (1i64, 4i64);
    let (p1, p2, p3) = (2i64, 4i64, 4i64);
    let patch_dim = c * p1 * p2 * p3;
    let dim = if use_random { patch_dim } else { 2048 };
    let num_layers = if use_random { 2 } else { 28 };

    println!("Configuration:");
    println!("  Prompt:       {prompt}");
    println!("  Video shape:  ({b}, {c}, {f}, {h}, {w})");
    println!("  Patch size:   ({p1}, {p2}, {p3})");
    println!("  Sequence len: {}", (f / p1) * (h / p2) * (w / p3));
    println!("  Model dim:    {dim}");
    println!("  Layers:       {num_layers}");
    println!("  Denoising:    {n_steps} steps");
    println!("  CFG scale:    {cfg_scale}");
    if let Some(ref path) = weights_path {
        println!("  Weights:      {path}");
    }
    println!();

    // Build model
    tch::maybe_init_cuda();
    let vs = tch::nn::VarStore::new(Device::Cpu);
    let model = build_model(&vs.root(), dim, patch_dim, num_layers);

    // Create patchify_proj for real weights (maps patch_dim → dim)
    let patchify_proj = if !use_random {
        Some(tch::nn::linear(vs.root() / "patchify_proj", patch_dim, dim, Default::default()))
    } else {
        None
    };

    // Load weights if provided
    if let Some(ref path) = weights_path {
        println!("Loading weights from: {path}");

        let data = std::fs::read(path).expect("failed to read weights file");
        let safetensor = safetensors::SafeTensors::deserialize(&data).expect("failed to deserialize");

        let mut loaded = 0u32;
        let mut skipped = 0u32;

        let _no_grad = tch::no_grad_guard();
        let mut vars = vs.variables();

        for (_var_name, var_tensor) in vars.iter_mut() {
            if let Ok(view) = safetensor.tensor(_var_name) {
                let kind = match view.dtype() {
                    safetensors::Dtype::F32 => tch::Kind::Float,
                    safetensors::Dtype::F16 => tch::Kind::Half,
                    safetensors::Dtype::BF16 => tch::Kind::BFloat16,
                    _ => tch::Kind::Float,
                };
                let shape: Vec<i64> = view.shape().iter().map(|&s| s as i64).collect();
                let loaded_tensor = Tensor::from_data_size(view.data(), &shape, kind);

                if var_tensor.size() == loaded_tensor.size() {
                    var_tensor.copy_(&loaded_tensor);
                    loaded += 1;
                } else {
                    skipped += 1;
                }
            } else {
                skipped += 1;
            }
        }

        println!("Loaded {loaded} tensors, skipped {skipped} (missing or mismatched)");
    }

    // Count parameters
    let vars = vs.variables();
    let n_params: usize = vars.values().map(|t| t.numel()).sum();
    println!("Model initialized: {n_params} parameters ({:.2} MB)",
        n_params as f64 * 4.0 / 1024.0 / 1024.0);

    // Initialize noisy latent
    tch::manual_seed(42);
    let noise = Tensor::randn([b, c, f, h, w], (Kind::Float, Device::Cpu));
    let mut x = noise.shallow_clone();

    // Text encoder context (random for now — text encoder not yet wired)
    let context = Tensor::randn([1, 4, dim], (Kind::Float, Device::Cpu));

    println!("\nDenoising loop:");
    println!("  Step | Sigma    | Output Mean | Output Std");
    println!("  -----|----------|-------------|------------");

    // Setup components
    let scheduler = Ltx2Scheduler::default();
    let guider = CFG::new(cfg_scale);
    let step = EulerStep::new();
    let sigmas = scheduler.sigmas(n_steps);

    // Denoising loop
    for i in 0..n_steps {
        let sigma = sigmas[i];
        let next_sigma = sigmas[i + 1];

        let patched = patchify_5d(&x, p1, p2, p3);

        let projected = if let Some(ref proj) = patchify_proj {
            proj.forward_t(&patched, false)
        } else {
            patched
        };

        let timestep = Tensor::from_slice(&[sigma as f32]);
        let cond_pred = model.forward(&projected, &timestep, &context, None, None);

        let uncond_context = Tensor::zeros(
            [1, context.size()[1], context.size()[2]],
            (Kind::Float, Device::Cpu),
        );
        let uncond_pred = model.forward(&projected, &timestep, &uncond_context, None, None);

        let guided = guider.guide(&cond_pred, &uncond_pred);
        let denoised = unpatchify_5d(&guided, b, c, f, h, w, p1, p2, p3);

        x = step.step(&x, sigma, next_sigma, &denoised, Kind::Float);

        let mean = x.mean(Kind::Float).double_value(&[]);
        let std = x.std(false).double_value(&[]);
        println!("  {:4} | {:8.4} | {:11.6} | {:10.6}", i + 1, sigma, mean, std);
    }

    // Final output
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                       Inference Complete                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Output shape:  {:?}", x.size());
    println!("Output mean:   {:.6}", x.mean(Kind::Float).double_value(&[]));
    println!("Output std:    {:.6}", x.std(false).double_value(&[]));
    println!("Output min:    {:.6}", x.min().double_value(&[]));
    println!("Output max:    {:.6}", x.max().double_value(&[]));
    println!("Output finite: {}", x.isfinite().all().double_value(&[]) > 0.0);

    // Decode latent to pixel space (simple channel mixing — VAE decoder not yet wired)
    let pixel = {
        let p = x.clamp(-1.0, 1.0);
        let r = p.narrow(1, 0, 1).squeeze_dim(1);
        let g = p.narrow(1, 1, 1).squeeze_dim(1);
        let b = p.narrow(1, 2, 1).squeeze_dim(1);
        Tensor::stack(&[&r, &g, &b], 1)
    };

    // Normalize to [0, 255]
    let pixel_min = pixel.min().double_value(&[]);
    let pixel_max = pixel.max().double_value(&[]);
    let pixel = (pixel - pixel_min) / (pixel_max - pixel_min + STABILITY_EPS);
    let pixel = (pixel * 255.0).to_kind(Kind::Uint8);

    // Save frames
    let frames_dir = "output_frames";
    std::fs::create_dir_all(frames_dir).expect("failed to create frames directory");

    use std::io::Write;
    for frame_idx in 0..f {
        let frame = pixel.narrow(2, frame_idx, 1).reshape([3, h, w]);
        let frame = frame.permute([1, 2, 0]);

        let pgm_path = format!("{frames_dir}/frame_{frame_idx:04}.pgm");
        let header = format!("P6\n{w} {h}\n255\n");
        let mut file = std::fs::File::create(&pgm_path).expect("failed to create PGM file");
        file.write_all(header.as_bytes()).unwrap();

        let data = frame.reshape([h * w * 3]);
        let bytes: Vec<u8> = (0..data.size()[0])
            .map(|i| data.double_value(&[i]) as u8)
            .collect();
        file.write_all(&bytes).unwrap();
    }

    println!("Frames saved to: {frames_dir}/ (0-{}.pgm)", f - 1);

    // Create video with ffmpeg
    let video_path = "output_video.mp4";
    let ffmpeg_result = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-framerate", "8",
            "-i", &format!("{frames_dir}/frame_%04d.pgm"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            video_path,
        ])
        .output();

    match ffmpeg_result {
        Ok(output) if output.status.success() => {
            println!("Video created: {video_path}");
        }
        _ => {
            println!("ffmpeg not available or failed. Video frames saved as PGM files.");
        }
    }

    println!("\nInference complete. Output is a {:?} tensor.", x.size());
}
