/// Integration tests for the video diffusion pipeline.
/// Tests the end-to-end flow: patchify → transformer → unpatchify
/// which is the core computation in LTX video generation.
use ltx_attention::RopeType;
use ltx_norm::RMSNorm;
use ltx_patchify::{patchify_5d, unpatchify_5d};
use ltx_transformer::block::BasicAVTransformerBlock;
use ltx_transformer::model::LTXModel;
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

/// Full video diffusion pipeline: 5D latent → patchify → transformer → unpatchify.
///
/// This is the core computation in LTX video generation. The pipeline:
/// 1. Takes a 5D video latent (B, C, F, H, W)
/// 2. Patchifies into (B, T, D) sequence
/// 3. Runs through transformer blocks
/// 4. Unpatchifies back to (B, C, F, H, W)
#[test]
fn test_video_diffusion_pipeline() {
    let vs = make_vs();
    let dim = 64;
    let _model = build_model(&vs.root(), dim);

    // Video latent: batch=1, channels=4, frames=4, height=16, width=16
    let (b, c, f, h, w) = (1i64, 4i64, 4i64, 16i64, 16i64);
    let latent = Tensor::randn([b, c, f, h, w], (Kind::Float, Device::Cpu));

    // Patchify: (B, C, F, H, W) → (B, T, D) where D = C * p1 * p2 * p3
    let (p1, p2, p3) = (2i64, 4i64, 4i64);
    let patched = patchify_5d(&latent, p1, p2, p3);
    let seq_len = (f / p1) * (h / p2) * (w / p3);
    let d = c * p1 * p2 * p3;
    assert_eq!(patched.size(), vec![b, seq_len, d]);

    // Transformer needs dim == d for the projection to work.
    // If d != dim, we need a projection. For this test, use dim == d.
    let vs2 = make_vs();
    let model2 = build_model(&vs2.root(), d);

    // Run transformer
    let timestep = Tensor::from_slice(&[0.5]);
    let context = Tensor::randn([1, 4, d], (Kind::Float, Device::Cpu));
    let transformer_out = model2.forward(&patched, &timestep, &context, None, None);
    assert_eq!(transformer_out.size(), vec![b, seq_len, d]);

    // Unpatchify: (B, T, D) → (B, C, F, H, W)
    let unp = unpatchify_5d(&transformer_out, b, c, f, h, w, p1, p2, p3);
    assert_eq!(unp.size(), vec![b, c, f, h, w]);

    // Output should be finite
    assert!(unp.isfinite().all().double_value(&[]) > 0.0);
}

/// Pipeline with different spatial resolutions.
#[test]
fn test_video_pipeline_various_resolutions() {
    let vs = make_vs();
    let (p1, p2, p3) = (2i64, 4i64, 4i64);

    for &(f, h, w) in &[
        (4i64, 16i64, 16i64),
        (8i64, 32i64, 32i64),
        (2i64, 8i64, 8i64),
    ] {
        let c = 4i64;
        let b = 1i64;
        let d = c * p1 * p2 * p3;
        let model = build_model(&vs.root(), d);

        let latent = Tensor::randn([b, c, f, h, w], (Kind::Float, Device::Cpu));
        let patched = patchify_5d(&latent, p1, p2, p3);
        let seq_len = (f / p1) * (h / p2) * (w / p3);

        let timestep = Tensor::from_slice(&[0.5]);
        let context = Tensor::randn([1, 4, d], (Kind::Float, Device::Cpu));
        let out = model.forward(&patched, &timestep, &context, None, None);
        assert_eq!(out.size(), vec![b, seq_len, d]);

        let unp = unpatchify_5d(&out, b, c, f, h, w, p1, p2, p3);
        assert_eq!(unp.size(), vec![b, c, f, h, w]);
        assert!(unp.isfinite().all().double_value(&[]) > 0.0);
    }
}

/// Pipeline with batch > 1.
#[test]
fn test_video_pipeline_batch() {
    let vs = make_vs();
    let (b, c, f, h, w) = (2i64, 4i64, 4i64, 16i64, 16i64);
    let (p1, p2, p3) = (2i64, 4i64, 4i64);
    let d = c * p1 * p2 * p3;
    let model = build_model(&vs.root(), d);

    let latent = Tensor::randn([b, c, f, h, w], (Kind::Float, Device::Cpu));
    let patched = patchify_5d(&latent, p1, p2, p3);
    let seq_len = (f / p1) * (h / p2) * (w / p3);

    let timestep = Tensor::randn([b], (Kind::Float, Device::Cpu));
    let context = Tensor::randn([b, 4, d], (Kind::Float, Device::Cpu));
    let out = model.forward(&patched, &timestep, &context, None, None);
    assert_eq!(out.size(), vec![b, seq_len, d]);

    let unp = unpatchify_5d(&out, b, c, f, h, w, p1, p2, p3);
    assert_eq!(unp.size(), vec![b, c, f, h, w]);
}

/// Pipeline with multiple transformer layers.
#[test]
fn test_video_pipeline_deep_transformer() {
    let vs = make_vs();
    let (b, c, f, h, w) = (1i64, 4i64, 4i64, 16i64, 16i64);
    let (p1, p2, p3) = (2i64, 4i64, 4i64);
    let d = c * p1 * p2 * p3;

    // Build a 4-layer transformer
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

    let latent = Tensor::randn([b, c, f, h, w], (Kind::Float, Device::Cpu));
    let patched = patchify_5d(&latent, p1, p2, p3);
    let seq_len = (f / p1) * (h / p2) * (w / p3);

    let timestep = Tensor::from_slice(&[0.5]);
    let context = Tensor::randn([1, 4, d], (Kind::Float, Device::Cpu));
    let out = model.forward(&patched, &timestep, &context, None, None);
    assert_eq!(out.size(), vec![b, seq_len, d]);

    let unp = unpatchify_5d(&out, b, c, f, h, w, p1, p2, p3);
    assert_eq!(unp.size(), vec![b, c, f, h, w]);
    assert!(unp.isfinite().all().double_value(&[]) > 0.0);
}

/// Pipeline with RoPE applied (simulating positional encoding).
#[test]
fn test_video_pipeline_with_rope() {
    use ltx_attention::{apply_rotary_emb, precompute_freqs_cis};

    let vs = make_vs();
    let (b, c, f, h, w) = (1i64, 4i64, 4i64, 16i64, 16i64);
    let (p1, p2, p3) = (2i64, 4i64, 4i64);
    let d = c * p1 * p2 * p3;
    let model = build_model(&vs.root(), d);

    let latent = Tensor::randn([b, c, f, h, w], (Kind::Float, Device::Cpu));
    let patched = patchify_5d(&latent, p1, p2, p3);
    let seq_len = (f / p1) * (h / p2) * (w / p3);

    // Precompute RoPE frequencies
    let (cos, sin) = precompute_freqs_cis(d, seq_len, 10000.0, RopeType::Interleaved, Device::Cpu);

    // Apply RoPE to the patched tensor (as if it were Q/K)
    let (q_rot, _k_rot) = apply_rotary_emb(&patched, &patched, &cos, &sin, RopeType::Interleaved);

    // Run transformer with RoPE-preprocessed input
    let timestep = Tensor::from_slice(&[0.5]);
    let context = Tensor::randn([1, 4, d], (Kind::Float, Device::Cpu));
    let out = model.forward(&q_rot, &timestep, &context, None, None);
    assert_eq!(out.size(), vec![b, seq_len, d]);

    let unp = unpatchify_5d(&out, b, c, f, h, w, p1, p2, p3);
    assert_eq!(unp.size(), vec![b, c, f, h, w]);
    assert!(unp.isfinite().all().double_value(&[]) > 0.0);
}
