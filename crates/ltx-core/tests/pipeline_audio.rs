/// Integration tests for the audio diffusion pipeline.
/// Tests: patchify_audio → transformer → unpatchify_audio
use ltx_attention::RopeType;
use ltx_norm::RMSNorm;
use ltx_patchify::{patchify_audio, unpatchify_audio};
use ltx_transformer::block::BasicAVTransformerBlock;
use ltx_transformer::model::LTXModel;
use tch::{Device, Kind, Tensor};

fn make_vs() -> tch::nn::VarStore {
    tch::nn::VarStore::new(Device::Cpu)
}

/// Full audio diffusion pipeline: 4D audio → patchify → transformer → unpatchify.
///
/// The audio pipeline:
/// 1. Takes a 4D audio tensor (B, C, T, F)
/// 2. Patchifies into (B, T, C*F) sequence
/// 3. Runs through transformer blocks
/// 4. Unpatchifies back to (B, C, T, F)
#[test]
fn test_audio_diffusion_pipeline() {
    let vs = make_vs();
    // For testing, use a smaller dimension
    let c_test = 4i64;
    let f_test = 16i64;
    let d_test = c_test * f_test;
    let t_test = 8i64;

    let mut blocks = Vec::new();
    for i in 0..2 {
        blocks.push(BasicAVTransformerBlock::new(
            &(vs.root() / "blocks" / i),
            d_test,
            4,
            d_test / 4,
            None,
            RopeType::Interleaved,
        ));
    }
    let norm_out = RMSNorm::default_eps(d_test, vs.root().device());
    let proj_out = tch::nn::linear(vs.root() / "proj_out", d_test, d_test, Default::default());
    let model = LTXModel::new(blocks, norm_out, proj_out);

    // Audio: (B, C, T, F)
    let audio = Tensor::randn([1, c_test, t_test, f_test], (Kind::Float, Device::Cpu));

    // Patchify: (B, C, T, F) → (B, T, C*F)
    let patched = patchify_audio(&audio);
    assert_eq!(patched.size(), vec![1, t_test, d_test]);

    // Transformer
    let timestep = Tensor::from_slice(&[0.5]);
    let context = Tensor::randn([1, 4, d_test], (Kind::Float, Device::Cpu));
    let out = model.forward(&patched, &timestep, &context, None, None);
    assert_eq!(out.size(), vec![1, t_test, d_test]);

    // Unpatchify: (B, T, C*F) → (B, C, T, F)
    let unp = unpatchify_audio(&out, c_test, f_test);
    assert_eq!(unp.size(), vec![1, c_test, t_test, f_test]);

    // Output should be finite
    assert!(unp.isfinite().all().double_value(&[]) > 0.0);
}

/// Audio pipeline with different channel/frequency configs.
#[test]
fn test_audio_pipeline_various_configs() {
    let vs = make_vs();

    for &(c, f, t) in &[(4i64, 16i64, 8i64), (8i64, 32i64, 16i64), (2i64, 8i64, 4i64)] {
        let d = c * f;
        let b = 1i64;

        let mut blocks = Vec::new();
        for i in 0..2 {
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

        let audio = Tensor::randn([b, c, t, f], (Kind::Float, Device::Cpu));
        let patched = patchify_audio(&audio);
        assert_eq!(patched.size(), vec![b, t, d]);

        let timestep = Tensor::from_slice(&[0.5]);
        let context = Tensor::randn([1, 4, d], (Kind::Float, Device::Cpu));
        let out = model.forward(&patched, &timestep, &context, None, None);
        assert_eq!(out.size(), vec![b, t, d]);

        let unp = unpatchify_audio(&out, c, f);
        assert_eq!(unp.size(), vec![b, c, t, f]);
        assert!(unp.isfinite().all().double_value(&[]) > 0.0);
    }
}

/// Audio pipeline with batch > 1.
#[test]
fn test_audio_pipeline_batch() {
    let vs = make_vs();
    let (b, c, t, f) = (4i64, 4i64, 8i64, 16i64);
    let d = c * f;

    let mut blocks = Vec::new();
    for i in 0..2 {
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

    let audio = Tensor::randn([b, c, t, f], (Kind::Float, Device::Cpu));
    let patched = patchify_audio(&audio);
    let timestep = Tensor::randn([b], (Kind::Float, Device::Cpu));
    let context = Tensor::randn([b, 4, d], (Kind::Float, Device::Cpu));

    let out = model.forward(&patched, &timestep, &context, None, None);
    assert_eq!(out.size(), vec![b, t, d]);

    let unp = unpatchify_audio(&out, c, f);
    assert_eq!(unp.size(), vec![b, c, t, f]);
    assert!(unp.isfinite().all().double_value(&[]) > 0.0);
}
