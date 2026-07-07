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
use ltx_components::{EulerStep, Ltx2Scheduler, Res2sStep, APG, CFG, STG};
use ltx_norm::RMSNorm;
use ltx_patchify::{patchify_5d, unpatchify_5d};
use ltx_quantization::{
    dequantize_int8, int4_mm::quantize_weight_to_int4_per_group, quantize_to_int8_per_tensor,
};
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

    /// Data type for model weights: fp32, fp16, bf16
    #[arg(long, default_value = "fp32")]
    dtype: String,
}

fn parse_device(s: &str) -> Device {
    match s {
        "auto" => {
            if tch::Cuda::is_available() {
                Device::Cuda(0)
            } else if cfg!(target_os = "macos") && tch::utils::has_mps() {
                Device::Mps
            } else {
                Device::Cpu
            }
        }
        "cpu" => Device::Cpu,
        s if s.starts_with("cuda:") => {
            let id: usize = s[5..].parse().unwrap_or(0);
            Device::Cuda(id)
        }
        "cuda" => Device::Cuda(0),
        s if s.starts_with("rocm:") => {
            let id: usize = s[5..].parse().unwrap_or(0);
            Device::Cuda(id)
        }
        "rocm" => Device::Cuda(0),
        _ => Device::Cpu,
    }
}

fn parse_dtype(s: &str) -> Kind {
    match s {
        "fp16" | "half" => Kind::Half,
        "bf16" | "bfloat16" => Kind::BFloat16,
        _ => Kind::Float,
    }
}

/// Cast all VarStore variables to the target dtype.
fn cast_vs_dtype(vs: &tch::nn::VarStore, target: Kind) {
    let mut vars = vs.variables();
    for (_name, tensor) in vars.iter_mut() {
        if tensor.kind() != target {
            *tensor = tensor.to_kind(target);
        }
    }
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

/// Quantization state for weight-only quantized inference.
enum QuantState {
    None,
    Int8(Vec<(String, Tensor, Tensor)>),
    Int4(
        Vec<(String, Tensor, Tensor)>,
        std::collections::HashMap<String, Vec<i64>>,
    ),
}

/// Benchmark a full denoising loop.
#[allow(clippy::too_many_arguments)]
fn bench_denoising_loop(
    model: &LTXModel,
    vs: &tch::nn::VarStore,
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
    quant: &QuantState,
) -> (f64, f64) {
    let scheduler = Ltx2Scheduler::default();
    let sigmas = scheduler.sigmas(steps);
    let mut x = Tensor::randn([1, 4, frames, height, width], (Kind::Float, device));
    let uncond = Tensor::zeros(
        [1, context.size()[1], context.size()[2]],
        (Kind::Float, device),
    );

    let t0 = Instant::now();
    for i in 0..steps {
        let sigma = sigmas[i];
        let next_sigma = sigmas[i + 1];
        let patched = patchify_5d(&x, p1, p2, p3);
        let projected = if let Some(proj) = patchify_proj {
            proj.forward_t(&patched, false)
        } else {
            patched
        };
        let ts = Tensor::from_slice(&[sigma as f32]).to_device(device);

        // For quantized inference: swap dequantized weights, run forward, swap back
        let (cond, uncond_pred) = match quant {
            QuantState::None => {
                let c = model.forward(&projected, &ts, context, None, None);
                let u = model.forward(&projected, &ts, &uncond, None, None);
                (c, u)
            }
            QuantState::Int8(q) => {
                let c = with_int8_weights(vs, q, device, || {
                    model.forward(&projected, &ts, context, None, None)
                });
                let u = with_int8_weights(vs, q, device, || {
                    model.forward(&projected, &ts, &uncond, None, None)
                });
                (c, u)
            }
            QuantState::Int4(q, shapes) => {
                let c = with_int4_weights(vs, q, shapes, device, || {
                    model.forward(&projected, &ts, context, None, None)
                });
                let u = with_int4_weights(vs, q, shapes, device, || {
                    model.forward(&projected, &ts, &uncond, None, None)
                });
                (c, u)
            }
        };

        let guided = CFG::new(7.5).guide(&cond, &uncond_pred);
        let denoised = unpatchify_5d(&guided, 1, 4, frames, height, width, p1, p2, p3);
        x = EulerStep::new().step(&x, sigma, next_sigma, &denoised, Kind::Float);
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let total_frames = frames as f64 * steps as f64;
    (elapsed, total_frames / elapsed)
}

/// Quantize all VarStore variables to INT8, returning quantized copies + scales.
/// Original FP32 weights are preserved in VarStore for computation.
fn quantize_vs_int8(vs: &tch::nn::VarStore) -> Vec<(String, Tensor, Tensor)> {
    let vars = vs.variables();
    let mut quantized = Vec::new();
    for (name, tensor) in vars.iter() {
        if tensor.kind() == Kind::Float {
            let (q, scale) = quantize_to_int8_per_tensor(tensor);
            quantized.push((name.clone(), q, scale));
        }
    }
    quantized
}

/// Quantize all VarStore variables to INT4 (per-group), returning packed + scales.
fn quantize_vs_int4(vs: &tch::nn::VarStore) -> Vec<(String, Tensor, Tensor)> {
    let vars = vs.variables();
    let mut quantized = Vec::new();
    for (name, tensor) in vars.iter() {
        if tensor.kind() == Kind::Float && tensor.dim() >= 2 {
            let (packed, scales) = quantize_weight_to_int4_per_group(tensor, 128);
            quantized.push((name.clone(), packed, scales));
        } else if tensor.kind() == Kind::Float {
            // Scalars/small tensors: keep as FP32
            quantized.push((
                name.clone(),
                tensor.shallow_clone(),
                Tensor::zeros([], (Kind::Float, tch::Device::Cpu)),
            ));
        }
    }
    quantized
}

/// Swap dequantized INT8 weights into VarStore for one forward pass, then restore quantized.
fn with_int8_weights<F: FnOnce() -> R, R>(
    vs: &tch::nn::VarStore,
    quantized: &[(String, Tensor, Tensor)],
    device: tch::Device,
    f: F,
) -> R {
    let originals: Vec<(String, Tensor)> = {
        let vars = vs.variables();
        quantized
            .iter()
            .filter_map(|(name, _q, _scale)| {
                vars.get(name).map(|t| (name.clone(), t.shallow_clone()))
            })
            .collect()
    };
    // Swap in dequantized weights
    {
        let mut vars = vs.variables();
        for (name, q, scale) in quantized {
            if let Some(var) = vars.get_mut(name) {
                let dq = dequantize_int8(q, scale, Kind::Float).to_device(device);
                *var = dq;
            }
        }
    }
    let result = f();
    // Restore originals
    {
        let mut vars = vs.variables();
        for (name, orig) in &originals {
            if let Some(var) = vars.get_mut(name) {
                *var = orig.shallow_clone();
            }
        }
    }
    result
}

/// Swap dequantized INT4 weights into VarStore for one forward pass, then restore.
fn with_int4_weights<F: FnOnce() -> R, R>(
    vs: &tch::nn::VarStore,
    quantized: &[(String, Tensor, Tensor)],
    shapes: &std::collections::HashMap<String, Vec<i64>>,
    device: tch::Device,
    f: F,
) -> R {
    let originals: Vec<(String, Tensor)> = {
        let vars = vs.variables();
        quantized
            .iter()
            .filter_map(|(name, _packed, _scales)| {
                vars.get(name).map(|t| (name.clone(), t.shallow_clone()))
            })
            .collect()
    };
    // Swap in dequantized INT4 weights
    {
        let mut vars = vs.variables();
        for (name, packed, scales) in quantized {
            if let Some(var) = vars.get_mut(name) {
                if scales.size().is_empty() && scales.double_value(&[]) == 0.0 {
                    // Scalar zero = kept as-is (small tensor)
                    continue;
                }
                let shape = shapes.get(name).cloned().unwrap_or_else(|| var.size());
                let num_groups = scales.size()[1];
                let group_size = 128i64;
                let out_features = shape[0];
                let in_features = shape[1];

                // Unpack INT4
                let low = packed.bitwise_and(0x0F).to_kind(Kind::Float);
                let shift_amt = Tensor::from_slice(&[4i64]).to_kind(Kind::Int8);
                let high = packed
                    .bitwise_right_shift(&shift_amt)
                    .bitwise_and(0x0F)
                    .to_kind(Kind::Float);
                let unpacked =
                    Tensor::stack(&[&low, &high], 2).reshape([out_features, in_features]);
                // Dequantize
                let scales_expanded = scales
                    .unsqueeze(2)
                    .expand([out_features, num_groups, group_size], true)
                    .reshape([out_features, in_features]);
                let dq = (unpacked * scales_expanded).to_device(device);
                *var = dq;
            }
        }
    }
    let result = f();
    // Restore originals
    {
        let mut vars = vs.variables();
        for (name, orig) in &originals {
            if let Some(var) = vars.get_mut(name) {
                *var = orig.shallow_clone();
            }
        }
    }
    result
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
        Some(tch::nn::linear(
            vs.root() / "patchify_proj",
            patch_dim,
            dim,
            Default::default(),
        ))
    } else {
        None
    };

    if let Some(ref path) = args.weights {
        let loaded = load_weights(&vs, path);
        eprintln!("Loaded {loaded} weights");
    }

    // Apply standard dtype (fp16/bf16) if requested
    let target_dtype = parse_dtype(&args.dtype);
    let is_quantized = matches!(args.dtype.as_str(), "int8" | "int4");
    if target_dtype != Kind::Float && !is_quantized {
        cast_vs_dtype(&vs, target_dtype);
        eprintln!("Cast model weights to {:?}", target_dtype);
    }

    // For INT8/INT4: quantize weights, keep FP32 for compute
    let int8_quantized: Vec<(String, Tensor, Tensor)> = if args.dtype == "int8" {
        eprintln!("Quantizing weights to INT8 (per-tensor symmetric)...");
        quantize_vs_int8(&vs)
    } else {
        Vec::new()
    };

    let int4_quantized: Vec<(String, Tensor, Tensor)> = if args.dtype == "int4" {
        eprintln!("Quantizing weights to INT4 (per-group, group_size=128)...");
        quantize_vs_int4(&vs)
    } else {
        Vec::new()
    };

    let n_params: usize = vs.variables().values().map(|t| t.numel()).sum();
    let weight_bytes_fp32: usize = vs.variables().values().map(|t| t.numel() * 4).sum();
    let weight_bytes_actual = if args.dtype == "int8" {
        int8_quantized
            .iter()
            .map(|(_, q, s)| q.numel() + s.numel())
            .sum::<usize>()
    } else if args.dtype == "int4" {
        int4_quantized
            .iter()
            .map(|(_, p, s)| {
                if s.size().is_empty() && s.double_value(&[]) == 0.0 {
                    p.numel() * 4
                } else {
                    p.numel() + s.numel()
                }
            })
            .sum::<usize>()
    } else {
        vs.variables()
            .values()
            .map(|t| {
                let es = match t.kind() {
                    Kind::Float | Kind::Int => 4,
                    Kind::Half | Kind::BFloat16 | Kind::Int16 => 2,
                    _ => 4,
                };
                t.numel() * es
            })
            .sum()
    };
    let compression = weight_bytes_fp32 as f64 / weight_bytes_actual.max(1) as f64;
    eprintln!(
        "Model: {dim}d, {num_layers} layers, {:.1}M params",
        n_params as f64 / 1e6
    );
    eprintln!("  FP32 weights: {:.1} MB", weight_bytes_fp32 as f64 / 1e6);
    eprintln!(
        "  {} weights: {:.1} MB ({:.1}x compression)",
        args.dtype.to_uppercase(),
        weight_bytes_actual as f64 / 1e6,
        compression
    );

    // Save INT4 shapes for dequantization
    let int4_shapes: std::collections::HashMap<String, Vec<i64>> = if args.dtype == "int4" {
        let vars = vs.variables();
        int4_quantized
            .iter()
            .filter_map(|(name, _, _)| vars.get(name).map(|t| (name.clone(), t.size())))
            .collect()
    } else {
        std::collections::HashMap::new()
    };

    let model = build_model(&vs.root(), dim, patch_dim, num_layers);
    let context = Tensor::randn([1, 4, dim], (Kind::Float, device));

    eprintln!("\n=== Benchmark Configuration ===");
    eprintln!(
        "Resolution: {}x{} latent ({}x{} pixels)",
        args.width,
        args.height,
        args.width * 32,
        args.height * 32
    );
    eprintln!("Frames: {}, Steps: {}", args.frames, args.steps);
    eprintln!("Warmup: {}, Iterations: {}", args.warmup, args.iterations);
    eprintln!("Dtype: {}", args.dtype);
    if is_quantized {
        eprintln!("Quantization: weight-only, dequantize to FP32 before each forward pass");
    }
    eprintln!();

    if args.compare_step_methods {
        // Compare Euler vs Res2s
        eprintln!("=== Step Method Comparison ===");
        for method_name in &["euler", "res2s"] {
            let mut times = Vec::new();
            for _ in 0..args.warmup + args.iterations {
                let mut x = Tensor::randn(
                    [1, c, args.frames, args.height, args.width],
                    (Kind::Float, device),
                );
                let uncond = Tensor::zeros(
                    [1, context.size()[1], context.size()[2]],
                    (Kind::Float, device),
                );
                let scheduler = Ltx2Scheduler::default();
                let sigmas = scheduler.sigmas(args.steps);

                let t0 = Instant::now();
                for i in 0..args.steps {
                    let sigma = sigmas[i];
                    let next_sigma = sigmas[i + 1];
                    let patched = patchify_5d(&x, p1, p2, p3);
                    let projected = if let Some(ref proj) = patchify_proj {
                        proj.forward_t(&patched, false)
                    } else {
                        patched
                    };
                    let ts = Tensor::from_slice(&[sigma as f32]).to_device(device);
                    let cond = model.forward(&projected, &ts, &context, None, None);
                    let uncond_pred = model.forward(&projected, &ts, &uncond, None, None);
                    let guided = CFG::new(7.5).guide(&cond, &uncond_pred);
                    let denoised = unpatchify_5d(
                        &guided,
                        b,
                        c,
                        args.frames,
                        args.height,
                        args.width,
                        p1,
                        p2,
                        p3,
                    );
                    x = match *method_name {
                        "res2s" => {
                            Res2sStep::new(1.0).step(&x, sigma, next_sigma, &denoised, Kind::Float)
                        }
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
                let mut x = Tensor::randn(
                    [1, c, args.frames, args.height, args.width],
                    (Kind::Float, device),
                );
                let uncond = Tensor::zeros(
                    [1, context.size()[1], context.size()[2]],
                    (Kind::Float, device),
                );
                let scheduler = Ltx2Scheduler::default();
                let sigmas = scheduler.sigmas(args.steps);

                let t0 = Instant::now();
                for i in 0..args.steps {
                    let sigma = sigmas[i];
                    let _next_sigma = sigmas[i + 1];
                    let patched = patchify_5d(&x, p1, p2, p3);
                    let projected = if let Some(ref proj) = patchify_proj {
                        proj.forward_t(&patched, false)
                    } else {
                        patched
                    };
                    let ts = Tensor::from_slice(&[sigma as f32]).to_device(device);
                    let cond = model.forward(&projected, &ts, &context, None, None);
                    let uncond_pred = model.forward(&projected, &ts, &uncond, None, None);
                    let guided = match *guider_name {
                        "apg" => APG::new(*scale, 0.0).guide(&cond, &uncond_pred),
                        "stg" => STG::new(*scale, 3.0).guide(&cond, &uncond_pred),
                        _ => CFG::new(*scale).guide(&cond, &uncond_pred),
                    };
                    let denoised = unpatchify_5d(
                        &guided,
                        b,
                        c,
                        args.frames,
                        args.height,
                        args.width,
                        p1,
                        p2,
                        p3,
                    );
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
        let quant_state = if args.dtype == "int8" {
            QuantState::Int8(int8_quantized)
        } else if args.dtype == "int4" {
            QuantState::Int4(int4_quantized, int4_shapes)
        } else {
            QuantState::None
        };
        let mut times = Vec::new();
        for iter in 0..args.warmup + args.iterations {
            let (elapsed, fps) = bench_denoising_loop(
                &model,
                &vs,
                &context,
                args.height,
                args.width,
                args.frames,
                args.steps,
                device,
                p1,
                p2,
                p3,
                patchify_proj.as_ref(),
                &quant_state,
            );
            if iter >= args.warmup {
                times.push(elapsed);
                eprintln!(
                    "  iter {}: {elapsed:.3}s ({fps:.1} frames/sec)",
                    iter - args.warmup
                );
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

    // Memory estimate — model weights
    let total_bytes: usize = vs
        .variables()
        .values()
        .map(|t| {
            let elem_size = match t.kind() {
                Kind::Float | Kind::Int => 4,
                Kind::Half | Kind::BFloat16 | Kind::Int16 => 2,
                Kind::Double => 8,
                Kind::Uint8
                | Kind::Int8
                | Kind::QInt8
                | Kind::QUInt8
                | Kind::QInt32
                | Kind::Bool => 1,
                _ => 4,
            };
            t.numel() * elem_size
        })
        .sum();
    eprintln!();
    eprintln!("=== Memory ===");
    eprintln!("  model weights: {:.1} MB", total_bytes as f64 / 1e6);

    // GPU VRAM tracking
    if device != Device::Cpu {
        eprintln!("  device: {device:?} (GPU inference)");
        // Try nvidia-smi first (CUDA), then rocm-smi (ROCm)
        let vram_used = std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .or_else(|| {
                // ROCm: use rocm-smi for VRAM info
                std::process::Command::new("rocm-smi")
                    .args(["--showmeminfo", "vram", "--csv"])
                    .output()
                    .ok()
                    .and_then(|o| String::from_utf8(o.stdout).ok())
                    .and_then(|csv| {
                        // Parse CSV: "device,used,total" (values in bytes)
                        let lines: Vec<&str> = csv.lines().collect();
                        if lines.len() >= 2 {
                            let vals: Vec<&str> = lines[1].split(',').collect();
                            if vals.len() >= 3 {
                                let used: f64 = vals[1].trim().parse().unwrap_or(0.0);
                                let total: f64 = vals[2].trim().parse().unwrap_or(0.0);
                                Some(format!("{:.0}, {:.0}", used / 1e6, total / 1e6))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
            });
        if let Some(vram) = vram_used {
            let parts: Vec<&str> = vram.split(", ").collect();
            if parts.len() == 2 {
                let used_mb: f64 = parts[0].trim().parse().unwrap_or(0.0);
                let total_mb: f64 = parts[1].trim().parse().unwrap_or(0.0);
                let pct = used_mb / total_mb * 100.0;
                eprintln!("  VRAM used: {used_mb:.0} MB / {total_mb:.0} MB ({pct:.0}%)");
            }
        }

        // Estimate VRAM needed for real 28-layer model at multiple dtypes
        let real_dim: usize = 2048;
        let real_layers: usize = 28;
        // Per layer: attention Q/K/V/O (4*dim^2) + FFN (8*dim^2) + adaLN (6*dim) + norms (3*dim)
        let per_layer_params: usize =
            4 * real_dim * real_dim + 8 * real_dim * real_dim + 6 * real_dim + 3 * real_dim;
        let proj_out_params = real_dim * 128; // dim * patch_dim
        let total_params = real_layers * per_layer_params + proj_out_params;
        // Activation memory: 2 forward passes, 11*dim per token per layer
        let seq_len = ((args.frames / 2) * (args.height / 4) * (args.width / 4)) as usize;
        let per_layer_act = seq_len * 11 * real_dim * 4; // FP32 activations
        let total_act = per_layer_act * real_layers * 2
            + 4 * (args.frames as usize) * (args.height as usize) * (args.width as usize) * 4;

        let vram_total: f64 = 2048.0;
        eprintln!();
        eprintln!("  Real 28L model VRAM estimate ({total_params:.0} params):");
        eprintln!(
            "  {:<8} {:>10} {:>14} {:>14} {:>6}",
            "Dtype", "Weights", "Activations", "Total VRAM", "Fits?"
        );
        eprintln!(
            "  {:<8} {:>10} {:>14} {:>14} {:>6}",
            "-----", "-------", "-----------", "----------", "----"
        );
        for (name, bytes_per_param) in &[("FP32", 4.0), ("FP16", 2.0), ("INT8", 1.0), ("INT4", 0.5)]
        {
            let weight_bytes = (total_params as f64 * bytes_per_param) as usize;
            let total_vram = weight_bytes + total_act;
            let fits = (total_vram as f64) < vram_total * 1e6;
            eprintln!(
                "  {:<8} {:>9.0} MB {:>13.1} MB {:>13.0} MB {:>6}",
                name,
                weight_bytes as f64 / 1e6,
                total_act as f64 / 1e6,
                total_vram as f64 / 1e6,
                if fits { "YES" } else { "NO" }
            );
        }
    } else {
        eprintln!("  device: CPU");
    }

    eprintln!("\ndone");
}
