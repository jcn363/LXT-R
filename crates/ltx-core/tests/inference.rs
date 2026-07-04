/// Full inference pipeline test for the LTX-2.3 Rust implementation.
/// Simulates a complete video diffusion denoising loop:
/// 1. Initialize noisy latent
/// 2. Run denoising loop with scheduler, model, and guidance
/// 3. Verify output is finite and within expected range
use ltx_attention::RopeType;
use ltx_components::{EulerStep, Ltx2Scheduler, CFG};
use ltx_norm::RMSNorm;
use ltx_patchify::{patchify_5d, unpatchify_5d};
use ltx_transformer::block::BasicAVTransformerBlock;
use ltx_transformer::model::LTXModel;
use ltx_types::{Guider, Scheduler};
use tch::{Device, Kind, Tensor};

fn make_vs() -> tch::nn::VarStore {
    tch::nn::VarStore::new(Device::Cpu)
}

fn build_model(vs: &tch::nn::Path, dim: i64) -> LTXModel {
    let mut blocks = Vec::new();
    for i in 0..2 {
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

/// Full video diffusion inference: noisy latent → denoised output.
///
/// This test simulates the complete denoising loop:
/// 1. Create a noisy video latent
/// 2. Patchify into sequence
/// 3. Run N denoising steps with scheduler, model, and guidance
/// 4. Unpatchify back to video latent
/// 5. Verify output is finite and within expected range
#[test]
fn test_full_video_inference() {
    let vs = make_vs();
    let (b, c, f, h, w) = (1i64, 4i64, 4i64, 16i64, 16i64);
    let (p1, p2, p3) = (2i64, 4i64, 4i64);
    let d = c * p1 * p2 * p3;
    let model = build_model(&vs.root(), d);

    // Initialize noisy latent
    let noise = Tensor::randn([b, c, f, h, w], (Kind::Float, Device::Cpu));
    let mut x = noise.shallow_clone();

    // Setup components
    let scheduler = Ltx2Scheduler::default();
    let guider = CFG::new(7.5);
    let step = EulerStep::new();

    // Get sigma schedule
    let n_steps = 4;
    let sigmas = scheduler.sigmas(n_steps);

    // Denoising loop
    for i in 0..n_steps {
        let sigma = sigmas[i];
        let next_sigma = sigmas[i + 1];

        // Patchify: (B, C, F, H, W) → (B, T, D)
        let patched = patchify_5d(&x, p1, p2, p3);

        // Model forward pass (conditional)
        let timestep = Tensor::from_slice(&[sigma as f32]);
        let context = Tensor::randn([b, 4, d], (Kind::Float, Device::Cpu));
        let cond_pred = model.forward(&patched, &timestep, &context, None, None);

        // Model forward pass (unconditional) for CFG
        let uncond_context = Tensor::zeros([b, 4, d], (Kind::Float, Device::Cpu));
        let uncond_pred = model.forward(&patched, &timestep, &uncond_context, None, None);

        // Apply guidance
        let guided = guider.guide(&cond_pred, &uncond_pred);

        // Unpatchify: (B, T, D) → (B, C, F, H, W)
        let denoised = unpatchify_5d(&guided, b, c, f, h, w, p1, p2, p3);

        // Euler step
        x = step.step(&x, sigma, next_sigma, &denoised, Kind::Float);
    }

    // Verify output
    assert_eq!(x.size(), vec![b, c, f, h, w]);
    assert!(x.isfinite().all().double_value(&[]) > 0.0, "Output contains NaN/Inf");
    assert!(x.abs().max().double_value(&[]) < 100.0, "Output has extreme values");
}

/// Full audio diffusion inference: noisy latent → denoised output.
#[test]
fn test_full_audio_inference() {
    let vs = make_vs();
    let (b, c, t, f) = (1i64, 4i64, 8i64, 16i64);
    let d = c * f;
    let model = build_model(&vs.root(), d);

    // Initialize noisy latent
    let mut x = Tensor::randn([b, c, t, f], (Kind::Float, Device::Cpu));

    // Setup components
    let scheduler = Ltx2Scheduler::default();
    let guider = CFG::new(7.5);
    let step = EulerStep::new();

    let n_steps = 4;
    let sigmas = scheduler.sigmas(n_steps);

    // Denoising loop
    for i in 0..n_steps {
        let sigma = sigmas[i];
        let next_sigma = sigmas[i + 1];

        // Patchify audio: (B, C, T, F) → (B, T, C*F)
        use ltx_patchify::{patchify_audio, unpatchify_audio};
        let patched = patchify_audio(&x);

        // Model forward pass
        let timestep = Tensor::from_slice(&[sigma as f32]);
        let context = Tensor::randn([b, 4, d], (Kind::Float, Device::Cpu));
        let cond_pred = model.forward(&patched, &timestep, &context, None, None);

        // Unconditional for CFG
        let uncond_context = Tensor::zeros([b, 4, d], (Kind::Float, Device::Cpu));
        let uncond_pred = model.forward(&patched, &timestep, &uncond_context, None, None);

        // Apply guidance
        let guided = guider.guide(&cond_pred, &uncond_pred);

        // Unpatchify: (B, T, C*F) → (B, C, T, F)
        let denoised = unpatchify_audio(&guided, c, f);

        // Euler step
        x = step.step(&x, sigma, next_sigma, &denoised, Kind::Float);
    }

    // Verify output
    assert_eq!(x.size(), vec![b, c, t, f]);
    assert!(x.isfinite().all().double_value(&[]) > 0.0, "Output contains NaN/Inf");
}

/// Inference with multiple CFG scales to verify guidance works correctly.
#[test]
fn test_inference_various_cfg_scales() {
    let vs = make_vs();
    let (b, c, f, h, w) = (1i64, 4i64, 4i64, 8i64, 8i64);
    let (p1, p2, p3) = (2i64, 4i64, 4i64);
    let d = c * p1 * p2 * p3;
    let model = build_model(&vs.root(), d);

    let n_steps = 3;
    let sigmas = Ltx2Scheduler::default().sigmas(n_steps);

    for &cfg_scale in &[1.0, 3.0, 7.5, 15.0] {
        let guider = CFG::new(cfg_scale);
        let step = EulerStep::new();
        let mut x = Tensor::randn([b, c, f, h, w], (Kind::Float, Device::Cpu));

        for i in 0..n_steps {
            let sigma = sigmas[i];
            let next_sigma = sigmas[i + 1];

            let patched = patchify_5d(&x, p1, p2, p3);
            let timestep = Tensor::from_slice(&[sigma as f32]);
            let context = Tensor::randn([b, 4, d], (Kind::Float, Device::Cpu));
            let cond_pred = model.forward(&patched, &timestep, &context, None, None);

            let uncond_context = Tensor::zeros([b, 4, d], (Kind::Float, Device::Cpu));
            let uncond_pred = model.forward(&patched, &timestep, &uncond_context, None, None);

            let guided = guider.guide(&cond_pred, &uncond_pred);
            let denoised = unpatchify_5d(&guided, b, c, f, h, w, p1, p2, p3);
            x = step.step(&x, sigma, next_sigma, &denoised, Kind::Float);
        }

        assert!(x.isfinite().all().double_value(&[]) > 0.0,
            "CFG scale {cfg_scale}: output contains NaN/Inf");
    }
}

/// Inference with Res2sStep instead of EulerStep.
#[test]
fn test_inference_res2s_step() {
    use ltx_components::Res2sStep;

    let vs = make_vs();
    let (b, c, f, h, w) = (1i64, 4i64, 4i64, 8i64, 8i64);
    let (p1, p2, p3) = (2i64, 4i64, 4i64);
    let d = c * p1 * p2 * p3;
    let model = build_model(&vs.root(), d);

    let n_steps = 4;
    let sigmas = Ltx2Scheduler::default().sigmas(n_steps);
    let step = Res2sStep::new(1.0);
    let mut x = Tensor::randn([b, c, f, h, w], (Kind::Float, Device::Cpu));

    for i in 0..n_steps {
        let sigma = sigmas[i];
        let next_sigma = sigmas[i + 1];

        let patched = patchify_5d(&x, p1, p2, p3);
        let timestep = Tensor::from_slice(&[sigma as f32]);
        let context = Tensor::randn([b, 4, d], (Kind::Float, Device::Cpu));
        let denoised_pred = model.forward(&patched, &timestep, &context, None, None);

        let denoised = unpatchify_5d(&denoised_pred, b, c, f, h, w, p1, p2, p3);
        x = step.step(&x, sigma, next_sigma, &denoised, Kind::Float);
    }

    assert!(x.isfinite().all().double_value(&[]) > 0.0, "Res2s output contains NaN/Inf");
}

/// Inference with deeper transformer (4 layers).
#[test]
fn test_inference_deep_transformer() {
    let vs = make_vs();
    let (b, c, f, h, w) = (1i64, 4i64, 4i64, 8i64, 8i64);
    let (p1, p2, p3) = (2i64, 4i64, 4i64);
    let d = c * p1 * p2 * p3;

    // Build 4-layer transformer
    let mut blocks = Vec::new();
    for i in 0..4 {
        blocks.push(BasicAVTransformerBlock::new(
            &(vs.root() / "blocks" / i),
            d,
            4,
            d / 4,
            None,
            RopeType::Interleaved,
        ));
    }
    let norm_out = RMSNorm::default_eps(d, vs.root().device());
    let proj_out = tch::nn::linear(vs.root() / "proj_out", d, d, Default::default());
    let model = LTXModel::new(blocks, norm_out, proj_out);

    let n_steps = 3;
    let sigmas = Ltx2Scheduler::default().sigmas(n_steps);
    let step = EulerStep::new();
    let mut x = Tensor::randn([b, c, f, h, w], (Kind::Float, Device::Cpu));

    for i in 0..n_steps {
        let sigma = sigmas[i];
        let next_sigma = sigmas[i + 1];

        let patched = patchify_5d(&x, p1, p2, p3);
        let timestep = Tensor::from_slice(&[sigma as f32]);
        let context = Tensor::randn([b, 4, d], (Kind::Float, Device::Cpu));
        let denoised_pred = model.forward(&patched, &timestep, &context, None, None);

        let denoised = unpatchify_5d(&denoised_pred, b, c, f, h, w, p1, p2, p3);
        x = step.step(&x, sigma, next_sigma, &denoised, Kind::Float);
    }

    assert!(x.isfinite().all().double_value(&[]) > 0.0, "Deep transformer output contains NaN/Inf");
}

/// Inference with batch > 1.
#[test]
fn test_inference_batch() {
    let vs = make_vs();
    let (b, c, f, h, w) = (2i64, 4i64, 4i64, 8i64, 8i64);
    let (p1, p2, p3) = (2i64, 4i64, 4i64);
    let d = c * p1 * p2 * p3;
    let model = build_model(&vs.root(), d);

    let n_steps = 3;
    let sigmas = Ltx2Scheduler::default().sigmas(n_steps);
    let step = EulerStep::new();
    let mut x = Tensor::randn([b, c, f, h, w], (Kind::Float, Device::Cpu));

    for i in 0..n_steps {
        let sigma = sigmas[i];
        let next_sigma = sigmas[i + 1];

        let patched = patchify_5d(&x, p1, p2, p3);
        let timestep = Tensor::from_slice(&[sigma as f32]);
        let context = Tensor::randn([b, 4, d], (Kind::Float, Device::Cpu));
        let denoised_pred = model.forward(&patched, &timestep, &context, None, None);

        let denoised = unpatchify_5d(&denoised_pred, b, c, f, h, w, p1, p2, p3);
        x = step.step(&x, sigma, next_sigma, &denoised, Kind::Float);
    }

    assert_eq!(x.size(), vec![b, c, f, h, w]);
    assert!(x.isfinite().all().double_value(&[]) > 0.0, "Batch output contains NaN/Inf");
}
