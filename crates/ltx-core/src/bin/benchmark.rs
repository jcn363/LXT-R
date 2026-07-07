//! LTX-2.3 inference benchmark tool.
//!
//! Measures denoising throughput (frames/sec) and memory usage across
//! different configurations. Useful for profiling and comparing settings.
//!
//! # Usage
//!
//! ```bash
//! # Quick benchmark (random weights, 2 steps)
//! cargo run --release --bin ltx-benchmark
//!
//! # Benchmark with real weights
//! cargo run --release --bin ltx-benchmark -- --weights model.safetensors
//!
//! # Custom resolution and steps
//! cargo run --release --bin ltx-benchmark -- --height 32 --width 32 --frames 8 --steps 10
//!
//! # Compare step methods
//! cargo run --release --bin ltx-benchmark -- --compare-step-methods
//!
//! # Compare guiders
//! cargo run --release --bin ltx-benchmark -- --compare-guiders
//! ```

use std::time::Instant;

use clap::Parser;
use ltx_attention::RopeType;
use ltx_components::{EulerStep, Res2sStep, Ltx2Scheduler, CFG, APG, STG};
use ltx_norm::RMSNorm;
use ltx_patchify::{patchify_5d, unpatchify_5d};
use ltx_transformer::block::BasicAVTransformerBlock;
use ltx_transformer::model::LTXModel;
use ltx_types::{Guider, Scheduler};
use tch::nn::ModuleT;
use tch::{Device, Kind, Tensor};

#[derive(Parser)]
#[command(name = "ltx-benchmark", about = "LTX-2.3 inference benchmark")]
struct Args {
    /// Path to transformer weights
    #[arg(short, long)]
    weights: Option<String>,

    /// Device
    #[arg(long, default_value = "auto")]
    device: String,

    /// Video height (latent pixels)
    #[arg(long, default_value_t = 16)]
    height: i64,

    /// Video width (latent pixels)
    #[arg(long, default_value_t = 16)]
    width: i64,

    /// Number of frames
    #[arg(short, long, default_value_t = 4)]
    frames: i64,

    /// Denoising steps
    #[arg(short, long, default_value_t = 20)]
    steps: usize,

    /// Number of warmup iterations
    #[arg(long, default_value_t = 2)]
    warmup: usize,

    /// Number of benchmark iterations
    #[arg(long, default_value_t = 5)]
    iterations: usize,

    /// Compare step methods (euler vs res2s)
    #[arg(long)]
    compare_step_methods: bool,

    /// Compare guiders (cfg, apg, stg)
    #[arg(long)]
    compare_guiders: bool,
}

fn parse_device(s: &str) -> Device {
    match s {
        "auto" => {
            if tch::Cuda::is_available() { Device::Cuda(0) }
            else if cfg!(target_os = "macos") && tch::utils::has_mps() { Device::Mps }
            else { Device::Cpu }
        }
        "cpu" => Device::Cpu,
        s if s.starts_with("cuda:") => {
            let id: usize = s[5..].parse().unwrap_or(0);
            Device::Cuda(id)
        }
        "cuda" => Device::Cuda(0),
        _ => Device::Cpu,
    }
}

fn build_model(vs: &tch::nn::Path, dim: i64, patch_dim: i64, num_layers: i64) -> LTXModel {
    let mut blocks = Vec::new();
    for i in 0..num_layers {
        blocks.push(BasicAVTransformerBlock::new(
            &(vs / "blocks" / i),
            dim, 4, dim / 4, None, RopeType::Interleaved,
        ));
    }
    let norm_out = RMSNorm::default_eps_with_path(vs / "norm_out", dim);
    let proj_out = tch::nn::linear(vs / "proj_out", dim, patch_dim, Default::default());
    LTXModel::new(blocks, norm_out, proj_out)
}

fn load_weights(vs: &tch::nn::VarStore, path: &str) -> u32 {
    let data = std::fs::read(path).expect("read weights");
    let st = safetensors::SafeTensors::deserialize(&data).expect("deserialize");
    let mut loaded = 0u32;
    let _no_grad = tch::no_grad_guard();
    let mut vars = vs.variables();
    for (name, tensor) in vars.iter_mut() {
        for attempt in 0..3 {
            let ckpt_name = match attempt {
                0 => name.clone(),
                1 => format!("encoder.{}", name.replace('/', ".")),
                _ => name.replace('/', "."),
            };
            if let Ok(view) = st.tensor(&ckpt_name) {
                let kind = match view.dtype() {
                    safetensors::Dtype::F16 => tch::Kind::Half,
                    safetensors::Dtype::BF16 => tch::Kind::BFloat16,
                    _ => tch::Kind::Float,
                };
                let shape: Vec<i64> = view.shape().iter().map(|&s| s as i64).collect();
                let loaded_t = Tensor::from_data_size(view.data(), &shape, kind);
                if tensor.size() == loaded_t.size() {
                    tensor.copy_(&loaded_t);
                    loaded += 1;
                    break;
                }
            }
        }
    }
    loaded
}

/// Benchmark a full denoising loop.
#[allow(clippy::too_many_arguments)]
fn bench_denoising_loop(
    model: &LTXModel,
    context: &Tensor,
    height: i64,
    width: i64,
    frames: i64,
    steps: usize,
    device: Device,
    p1: i64,
    p2: i64,
    p3: i64,
    patchify_proj: Option<&tch::nn::Linear>,
) -> (f64, f64) {
    let scheduler = Ltx2Scheduler::default();
    let sigmas = scheduler.sigmas(steps);
    let mut x = Tensor::randn([1, 4, frames, height, width], (Kind::Float, device));
    let uncond = Tensor::zeros([1, context.size()[1], context.size()[2]], (Kind::Float, device));

    let t0 = Instant::now();
    for i in 0..steps {
        let sigma = sigmas[i];
        let next_sigma = sigmas[i + 1];
        let patched = patchify_5d(&x, p1, p2, p3);
        let projected = if let Some(proj) = patchify_proj {
            proj.forward_t(&patched, false)
        } else { patched };
        let ts = Tensor::from_slice(&[sigma as f32]).to_device(device);
        let cond = model.forward(&projected, &ts, context, None, None);
        let uncond_pred = model.forward(&projected, &ts, &uncond, None, None);
        let guided = CFG::new(7.5).guide(&cond, &uncond_pred);
        let denoised = unpatchify_5d(&guided, 1, 4, frames, height, width, p1, p2, p3);
        x = EulerStep::new().step(&x, sigma, next_sigma, &denoised, Kind::Float);
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let total_frames = frames as f64 * steps as f64;
    (elapsed, total_frames / elapsed)
}

fn main() {
    let args = Args::parse();
    let device = parse_device(&args.device);
    tch::maybe_init_cuda();

    let (b, c) = (1i64, 4i64);
    let (p1, p2, p3) = (2i64, 4i64, 4i64);
    let patch_dim = c * p1 * p2 * p3;
    let use_random = args.weights.is_none();
    let dim = if use_random { patch_dim } else { 2048 };
    let num_layers = if use_random { 2 } else { 28 };

    // Build model
    eprintln!("Building model on {device:?}...");
    let vs = tch::nn::VarStore::new(device);
    let patchify_proj = if !use_random {
        Some(tch::nn::linear(vs.root() / "patchify_proj", patch_dim, dim, Default::default()))
    } else {
        None
    };

    if let Some(ref path) = args.weights {
        let loaded = load_weights(&vs, path);
        eprintln!("Loaded {loaded} weights");
    }

    let n_params: usize = vs.variables().values().map(|t| t.numel()).sum();
    eprintln!("Model: {dim}d, {num_layers} layers, {:.1}M params", n_params as f64 / 1e6);

    let model = build_model(&vs.root(), dim, patch_dim, num_layers);
    let context = Tensor::randn([1, 4, dim], (Kind::Float, device));

    eprintln!("\n=== Benchmark Configuration ===");
    eprintln!("Resolution: {}x{} latent ({}x{} pixels)", args.width, args.height, args.width * 32, args.height * 32);
    eprintln!("Frames: {}, Steps: {}", args.frames, args.steps);
    eprintln!("Warmup: {}, Iterations: {}", args.warmup, args.iterations);
    eprintln!();

    if args.compare_step_methods {
        // Compare Euler vs Res2s
        eprintln!("=== Step Method Comparison ===");
        for method_name in &["euler", "res2s"] {
            let mut times = Vec::new();
            for _ in 0..args.warmup + args.iterations {
                let mut x = Tensor::randn([1, c, args.frames, args.height, args.width], (Kind::Float, device));
                let uncond = Tensor::zeros([1, context.size()[1], context.size()[2]], (Kind::Float, device));
                let scheduler = Ltx2Scheduler::default();
                let sigmas = scheduler.sigmas(args.steps);

                let t0 = Instant::now();
                for i in 0..args.steps {
                    let sigma = sigmas[i];
                    let next_sigma = sigmas[i + 1];
                    let patched = patchify_5d(&x, p1, p2, p3);
                    let projected = if let Some(ref proj) = patchify_proj {
                        proj.forward_t(&patched, false)
                    } else { patched };
                    let ts = Tensor::from_slice(&[sigma as f32]).to_device(device);
                    let cond = model.forward(&projected, &ts, &context, None, None);
                    let uncond_pred = model.forward(&projected, &ts, &uncond, None, None);
                    let guided = CFG::new(7.5).guide(&cond, &uncond_pred);
                    let denoised = unpatchify_5d(&guided, b, c, args.frames, args.height, args.width, p1, p2, p3);
                    x = match *method_name {
                        "res2s" => Res2sStep::new(1.0).step(&x, sigma, next_sigma, &denoised, Kind::Float),
                        _ => EulerStep::new().step(&x, sigma, next_sigma, &denoised, Kind::Float),
                    };
                }
                times.push(t0.elapsed().as_secs_f64());
            }
            // Skip warmup
            let bench_times: Vec<f64> = times[args.warmup..].to_vec();
            let avg = bench_times.iter().sum::<f64>() / bench_times.len() as f64;
            let fps = (args.frames as f64 * args.steps as f64) / avg;
            eprintln!("{method_name:>8}: avg={avg:.3}s, {fps:.1} frames/sec");
        }
    } else if args.compare_guiders {
        // Compare CFG vs APG vs STG
        eprintln!("=== Guider Comparison ===");
        for (guider_name, scale) in &[("cfg", 7.5), ("apg", 7.5), ("stg", 7.5)] {
            let mut times = Vec::new();
            for _ in 0..args.warmup + args.iterations {
                let mut x = Tensor::randn([1, c, args.frames, args.height, args.width], (Kind::Float, device));
                let uncond = Tensor::zeros([1, context.size()[1], context.size()[2]], (Kind::Float, device));
                let scheduler = Ltx2Scheduler::default();
                let sigmas = scheduler.sigmas(args.steps);

                let t0 = Instant::now();
                for i in 0..args.steps {
                    let sigma = sigmas[i];
                    let _next_sigma = sigmas[i + 1];
                    let patched = patchify_5d(&x, p1, p2, p3);
                    let projected = if let Some(ref proj) = patchify_proj {
                        proj.forward_t(&patched, false)
                    } else { patched };
                    let ts = Tensor::from_slice(&[sigma as f32]).to_device(device);
                    let cond = model.forward(&projected, &ts, &context, None, None);
                    let uncond_pred = model.forward(&projected, &ts, &uncond, None, None);
                    let guided = match *guider_name {
                        "apg" => APG::new(*scale, 0.0).guide(&cond, &uncond_pred),
                        "stg" => STG::new(*scale, 3.0).guide(&cond, &uncond_pred),
                        _ => CFG::new(*scale).guide(&cond, &uncond_pred),
                    };
                    let denoised = unpatchify_5d(&guided, b, c, args.frames, args.height, args.width, p1, p2, p3);
                    x = EulerStep::new().step(&x, sigmas[i], sigmas[i + 1], &denoised, Kind::Float);
                }
                times.push(t0.elapsed().as_secs_f64());
            }
            let bench_times: Vec<f64> = times[args.warmup..].to_vec();
            let avg = bench_times.iter().sum::<f64>() / bench_times.len() as f64;
            let fps = (args.frames as f64 * args.steps as f64) / avg;
            eprintln!("{guider_name:>8}: avg={avg:.3}s, {fps:.1} frames/sec");
        }
    } else {
        // Standard throughput benchmark
        eprintln!("=== Throughput Benchmark ===");
        let mut times = Vec::new();
        for iter in 0..args.warmup + args.iterations {
            let (elapsed, fps) = bench_denoising_loop(
                &model, &context, args.height, args.width, args.frames,
                args.steps, device, p1, p2, p3, patchify_proj.as_ref(),
            );
            if iter >= args.warmup {
                times.push(elapsed);
                eprintln!("  iter {}: {elapsed:.3}s ({fps:.1} frames/sec)", iter - args.warmup);
            }
        }

        let avg = times.iter().sum::<f64>() / times.len() as f64;
        let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let fps = (args.frames as f64 * args.steps as f64) / avg;

        eprintln!();
        eprintln!("=== Results ===");
        eprintln!("  avg: {avg:.3}s, min: {min:.3}s, max: {max:.3}s");
        eprintln!("  throughput: {fps:.1} frames/sec");
        eprintln!("  per-step:   {:.1}ms", avg * 1000.0 / args.steps as f64);
    }

    // Memory estimate
    let total_bytes: usize = vs.variables().values().map(|t| {
        let elem_size = match t.kind() {
            Kind::Float | Kind::Int => 4,
            Kind::Half | Kind::BFloat16 | Kind::Int16 => 2,
            Kind::Double => 8,
            Kind::Uint8 | Kind::Int8 | Kind::QInt8 | Kind::QUInt8 | Kind::QInt32 | Kind::Bool => 1,
            _ => 4,
        };
        t.numel() * elem_size
    }).sum();
    eprintln!();
    eprintln!("=== Memory ===");
    eprintln!("  model weights: {:.1} MB", total_bytes as f64 / 1e6);
    if device != Device::Cpu {
        eprintln!("  device: {device:?} (GPU inference)");
    } else {
        eprintln!("  device: CPU");
    }

    eprintln!("\ndone");
}
