/// End-to-end tests for the Video VAE encoder/decoder.
///
/// Strategy: use tiny tensors to avoid OOM on CPU. Tests that need checkpoint
/// weights use the smallest viable input; tests that verify architecture shape
/// flow use random weights only.
use ltx_video_vae::configurator::{build_encoder, build_decoder};
use ltx_video_vae::load_vae_weights;
use ltx_types::NormLayerType;

fn weights_path(name: &str) -> String {
    let root = format!("{}/../..", env!("CARGO_MANIFEST_DIR"));
    format!("{root}/weights/{name}")
}

const CKPT: &str = "ltx-video-2b-v0.9.1.safetensors";

// ---------------------------------------------------------------------------
// Encoder-only tests
// ---------------------------------------------------------------------------

#[test]
fn test_encoder_weight_loading() {
    let vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let _encoder = build_encoder(&(vs.root() / "encoder"), NormLayerType::Group, 32, false);
    let loaded = load_vae_weights(&vs, &weights_path(CKPT), "vae.");
    let total = vs.variables().len();
    assert_eq!(loaded as usize, total, "all encoder variables should match checkpoint");
}

#[test]
fn test_encoder_forward_shape_small() {
    let vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let encoder = build_encoder(&(vs.root() / "encoder"), NormLayerType::Group, 32, false);
    load_vae_weights(&vs, &weights_path(CKPT), "vae.");

    // [1, 3, 4, 32, 32] → after space_to_depth(r=4): [1, 48, 4, 8, 8]
    // after 3 DownsampleConv (stride 2,2,2): T=4→2→1, spatial 8→4→2→1
    let input = tch::Tensor::randn([1, 3, 4, 32, 32], (tch::Kind::Float, tch::Device::Cpu));
    let latent = encoder.encode_mean(&input);
    assert_eq!(latent.size(), vec![1, 128, 1, 1, 1]);
}

#[test]
fn test_encoder_forward_shape_medium() {
    let vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let encoder = build_encoder(&(vs.root() / "encoder"), NormLayerType::Group, 32, false);
    load_vae_weights(&vs, &weights_path(CKPT), "vae.");

    // [1, 3, 4, 64, 64] → after space_to_depth(r=4): [1, 48, 4, 16, 16]
    // after 3 DownsampleConv: T=4→2→1, spatial 16→8→4→2
    let input = tch::Tensor::randn([1, 3, 4, 64, 64], (tch::Kind::Float, tch::Device::Cpu));
    let raw = encoder.forward(&input);
    assert_eq!(raw.size(), vec![1, 129, 1, 2, 2]);
    let latent = encoder.encode_mean(&input);
    assert_eq!(latent.size(), vec![1, 128, 1, 2, 2]);
}

// ---------------------------------------------------------------------------
// Decoder-only tests (random weights — no checkpoint loading)
// ---------------------------------------------------------------------------

#[test]
fn test_decoder_forward_shape_random_weights() {
    // Minimal latent: [1, 128, 1, 2, 2]
    // Decoder conv_in: 128 → 1024
    // Block 0 (ResBlocks x8): [1, 1024, 1, 2, 2]
    // Block 1 (CompressAll): [1, 512, 1, 4, 4]
    // Block 2 (ResBlocks x7): [1, 512, 1, 4, 4]
    // Block 3 (CompressAll): [1, 256, 1, 8, 8]
    // Block 4 (ResBlocks x6): [1, 256, 1, 8, 8]
    // Block 5 (CompressAll): [1, 128, 1, 16, 16]
    // Block 6 (ResBlocks x5): [1, 128, 1, 16, 16]
    // last modulation → conv_out(128→48) → depth_to_space(r=4): [1, 3, 1, 64, 64]
    let vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let decoder = build_decoder(&(vs.root() / "decoder"), NormLayerType::Group, 32, false);
    let latent = tch::Tensor::randn([1, 128, 1, 2, 2], (tch::Kind::Float, tch::Device::Cpu));
    let timestep = tch::Tensor::from_slice(&[1000.0f32]);
    let output = decoder.forward(&latent, &timestep);
    assert_eq!(output.size(), vec![1, 3, 1, 64, 64]);
}

#[test]
fn test_decoder_weight_loading() {
    let vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let _decoder = build_decoder(&(vs.root() / "decoder"), NormLayerType::Group, 32, false);
    let loaded = load_vae_weights(&vs, &weights_path(CKPT), "vae.");
    let total = vs.variables().len();
    assert_eq!(loaded as usize, total, "all decoder variables should match checkpoint");
}

// ---------------------------------------------------------------------------
// Encoder architecture shape flow (random weights)
// ---------------------------------------------------------------------------

#[test]
fn test_encoder_forward_shape_random() {
    let vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let encoder = build_encoder(&(vs.root() / "encoder"), NormLayerType::Group, 32, false);
    // [1, 3, 4, 64, 64] with random weights
    let input = tch::Tensor::randn([1, 3, 4, 64, 64], (tch::Kind::Float, tch::Device::Cpu));
    let raw = encoder.forward(&input);
    assert_eq!(raw.size(), vec![1, 129, 1, 2, 2]);
    let latent = encoder.encode_mean(&input);
    assert_eq!(latent.size(), vec![1, 128, 1, 2, 2]);
}

// ---------------------------------------------------------------------------
// Roundtrip: encoder (loaded) → decoder (random weights) shape consistency
// ---------------------------------------------------------------------------

#[test]
fn test_encoder_decoder_shape_consistency() {
    // Verify encoder output can feed into decoder input by shape.
    let vs_e = tch::nn::VarStore::new(tch::Device::Cpu);
    let encoder = build_encoder(&(vs_e.root() / "encoder"), NormLayerType::Group, 32, false);

    let vs_d = tch::nn::VarStore::new(tch::Device::Cpu);
    let decoder = build_decoder(&(vs_d.root() / "decoder"), NormLayerType::Group, 32, false);

    // Tiny input → encoder → latent [1, 128, 1, 1, 1]
    let input = tch::Tensor::randn([1, 3, 4, 32, 32], (tch::Kind::Float, tch::Device::Cpu));
    let latent = encoder.encode_mean(&input);
    assert_eq!(latent.size()[1], 128, "encoder outputs 128 channels");

    // Decoder accepts [B, 128, T', H', W']
    let timestep = tch::Tensor::from_slice(&[1000.0f32]);
    let output = decoder.forward(&latent, &timestep);
    assert_eq!(output.size()[0], 1);
    assert_eq!(output.size()[1], 3);
    // depth_to_space(r=4) upsamples spatial by 4x
    assert!(output.size()[3] >= 4 && output.size()[4] >= 4);
}
