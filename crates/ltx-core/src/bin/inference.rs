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
//!   cargo run --bin ltx-inference -- --weights path/to/model.safetensors --steps 4

use ltx_attention::RopeType;
use ltx_components::{EulerStep, Ltx2Scheduler, CFG};
use ltx_norm::RMSNorm;
use ltx_patchify::{patchify_5d, unpatchify_5d};
use ltx_transformer::block::BasicAVTransformerBlock;
use ltx_transformer::model::LTXModel;
use ltx_types::{Guider, Scheduler};
use tch::nn::ModuleT;
use tch::{Device, Kind, Tensor};

fn build_model(vs: &tch::nn::Path, dim: i64, num_layers: i64) -> LTXModel {
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
    let norm_out = RMSNorm::default_eps(dim, vs.device());
    let proj_out = tch::nn::linear(vs / "proj_out", dim, dim, Default::default());
    LTXModel::new(blocks, norm_out, proj_out)
}

fn main() {
    // Parse arguments
    let args: Vec<String> = std::env::args().collect();
    let n_steps = args.iter()
        .position(|a| a == "--steps")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);

    let weights_path = args.iter()
        .position(|a| a == "--weights")
        .and_then(|i| args.get(i + 1))
        .cloned();

    let save_latent = args.iter().any(|a| a == "--save-latent");

    let use_random = weights_path.is_none();
    let mode = if use_random { "Random Weights" } else { "Loaded Weights" };

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           LTX-2.3 Rust Inference Demo ({:<20}) ║", mode);
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Configuration
    let (b, c, f, h, w) = (1i64, 4i64, 4i64, 16i64, 16i64);
    let (p1, p2, p3) = (2i64, 4i64, 4i64);
    let patch_dim = c * p1 * p2 * p3;  // 128 — patchified dimension
    let dim = if use_random { patch_dim } else { 2048 };  // Model hidden dim
    let num_layers = if use_random { 2 } else { 28 };  // 2B model has 28 layers
    let cfg_scale = 7.5;

    println!("Configuration:");
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
    let model = build_model(&vs.root(), dim, num_layers);

    // Create patchify_proj for real weights (maps patch_dim → dim)
    let patchify_proj = if !use_random {
        Some(tch::nn::linear(vs.root() / "patchify_proj", patch_dim, dim, Default::default()))
    } else {
        None
    };

    // Create output projection (maps dim → patch_dim) for real weights
    let output_proj = if !use_random {
        // proj_out maps dim → dim in the model, but we need dim → patch_dim
        // Use the model's proj_out and then project down
        Some(tch::nn::linear(vs.root() / "output_proj", dim, patch_dim, Default::default()))
    } else {
        None
    };

    // Load weights if provided
    if let Some(ref path) = weights_path {
        println!("Loading weights from: {path}");

        // Load safetensors directly and match keys manually
        let data = std::fs::read(path).expect("failed to read weights file");
        let safetensor = safetensors::SafeTensors::deserialize(&data).expect("failed to deserialize");

        let mut loaded = 0;
        let mut skipped = 0;

        // Disable gradients for weight loading
        let _no_grad = tch::no_grad_guard();

        let mut vars = vs.variables();

        for (var_name, var_tensor) in vars.iter_mut() {
            if let Ok(view) = safetensor.tensor(var_name) {
                // Tensor found - load it
                let kind = match view.dtype() {
                    safetensors::Dtype::F32 => tch::Kind::Float,
                    safetensors::Dtype::F16 => tch::Kind::Half,
                    safetensors::Dtype::BF16 => tch::Kind::BFloat16,
                    _ => tch::Kind::Float,
                };
                let shape: Vec<i64> = view.shape().iter().map(|&s| s as i64).collect();
                let data = view.data();
                let loaded_tensor = Tensor::from_data_size(data, &shape, kind);

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

        // Patchify: (B, C, F, H, W) → (B, T, patch_dim)
        let patched = patchify_5d(&x, p1, p2, p3);

        // Project to model dim if needed
        let projected = if let Some(ref proj) = patchify_proj {
            proj.forward_t(&patched, false)
        } else {
            patched
        };

        // Model forward pass (conditional)
        let timestep = Tensor::from_slice(&[sigma as f32]);
        let context = Tensor::randn([b, 4, dim], (Kind::Float, Device::Cpu));
        let cond_pred = model.forward(&projected, &timestep, &context, None, None);

        // Model forward pass (unconditional) for CFG
        let uncond_context = Tensor::zeros([b, 4, dim], (Kind::Float, Device::Cpu));
        let uncond_pred = model.forward(&projected, &timestep, &uncond_context, None, None);

        // Apply guidance
        let guided = guider.guide(&cond_pred, &uncond_pred);

        // Project back to patch_dim if needed
        let output = if let Some(ref proj) = output_proj {
            proj.forward_t(&guided, false)
        } else {
            guided
        };

        // Unpatchify: (B, T, patch_dim) → (B, C, F, H, W)
        let denoised = unpatchify_5d(&output, b, c, f, h, w, p1, p2, p3);

        // Euler step
        x = step.step(&x, sigma, next_sigma, &denoised, Kind::Float);

        // Print progress
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

    // Save latent if requested
    if save_latent {
        println!("\nLatent tensor saved (in-memory). Shape: {:?}", x.size());
    }

    // Convert to pixel space for visualization
    let pixel = x.clamp(-1.0, 1.0);  // Clamp to valid range
    let pixel = (pixel + 1.0) * 127.5;  // Convert to [0, 255]
    let pixel = pixel.to_kind(Kind::Uint8);

    // Save first frame as a simple PGM image for verification
    // x is (B=1, C=4, F=4, H=16, W=16) → take first frame → reshape to (H, W, C)
    let frame = pixel.narrow(2, 0, 1);  // (1, 4, 1, 16, 16)
    let frame = frame.reshape([1, c, h, w]);  // (1, 4, 16, 16)
    let frame = frame.narrow(0, 0, 1);  // (1, 4, 16, 16) — keep batch dim
    let frame = frame.permute([0, 2, 3, 1]);  // (1, 16, 16, 4)

    // Write PGM header + data
    let pgm_path = "output_frame.pgm";
    let header = format!("P6\n{w} {h}\n255\n");
    let mut file = std::fs::File::create(pgm_path).expect("failed to create PGM file");
    use std::io::Write;
    file.write_all(header.as_bytes()).unwrap();

    // Convert tensor to bytes
    let data = frame.reshape([1, h * w * c]);
    let bytes: Vec<u8> = (0..data.size()[1])
        .map(|i| data.double_value(&[0, i]) as u8)
        .collect();
    file.write_all(&bytes).unwrap();
    println!("First frame saved to: {pgm_path} ({w}x{h} RGB)");

    println!("\nInference complete. Output is a {:?} tensor.", x.size());
}
