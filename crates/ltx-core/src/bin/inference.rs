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

    let use_random = weights_path.is_none();
    let mode = if use_random { "Random Weights" } else { "Loaded Weights" };

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           LTX-2.3 Rust Inference Demo ({:<20}) ║", mode);
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Configuration
    let (b, c, f, h, w) = (1i64, 4i64, 4i64, 16i64, 16i64);
    let (p1, p2, p3) = (2i64, 4i64, 4i64);
    let dim = c * p1 * p2 * p3;  // 256 — must match patchified dimension
    let num_layers = 2;
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
    let mut vs = tch::nn::VarStore::new(Device::Cpu);
    let model = build_model(&vs.root(), dim, num_layers);

    // Load weights if provided
    if let Some(ref path) = weights_path {
        println!("Loading weights from: {path}");
        vs.load(path).expect("failed to load weights");
        println!("Weights loaded successfully!");
    }

    // Count parameters
    let vars = vs.variables();
    let n_params: usize = vars.iter().map(|(_, t)| t.numel()).sum();
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

        // Patchify: (B, C, F, H, W) → (B, T, D)
        let patched = patchify_5d(&x, p1, p2, p3);

        // Model forward pass (conditional)
        let timestep = Tensor::from_slice(&[sigma as f32]);
        let context = Tensor::randn([b, 4, dim], (Kind::Float, Device::Cpu));
        let cond_pred = model.forward(&patched, &timestep, &context, None, None);

        // Model forward pass (unconditional) for CFG
        let uncond_context = Tensor::zeros([b, 4, dim], (Kind::Float, Device::Cpu));
        let uncond_pred = model.forward(&patched, &timestep, &uncond_context, None, None);

        // Apply guidance
        let guided = guider.guide(&cond_pred, &uncond_pred);

        // Unpatchify: (B, T, D) → (B, C, F, H, W)
        let denoised = unpatchify_5d(&guided, b, c, f, h, w, p1, p2, p3);

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
    println!("\nInference complete. Output is a {:?} tensor.", x.size());
}
