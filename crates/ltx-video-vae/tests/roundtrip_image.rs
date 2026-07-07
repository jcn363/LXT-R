/// Full encode-decode roundtrip with a synthetic RGB image.
use ltx_video_vae::configurator::{build_encoder, build_decoder};
use ltx_video_vae::load_vae_weights;
use ltx_types::NormLayerType;
use serial_test::serial;

fn weights_path(name: &str) -> String {
    let root = format!("{}/../..", env!("CARGO_MANIFEST_DIR"));
    format!("{root}/weights/{name}")
}

const CKPT: &str = "ltx-video-2b-v0.9.1.safetensors";

#[test]
#[serial]
fn test_encode_decode_roundtrip_with_image() {
    let no_grad = tch::no_grad_guard();

    // Load encoder
    let vs_e = tch::nn::VarStore::new(tch::Device::Cpu);
    let encoder = build_encoder(&(vs_e.root() / "encoder"), NormLayerType::Group, 32, false);
    let loaded_e = load_vae_weights(&vs_e, &weights_path(CKPT), "vae.");
    eprintln!("encoder weights: {loaded_e}");

    // Load decoder
    let vs_d = tch::nn::VarStore::new(tch::Device::Cpu);
    let decoder = build_decoder(&(vs_d.root() / "decoder"), NormLayerType::Group, 32, false);
    let loaded_d = load_vae_weights(&vs_d, &weights_path(CKPT), "vae.");
    eprintln!("decoder weights: {loaded_d}");

    // Generate synthetic 256x256 RGB image as [1, 3, 1, 256, 256]
    let input = tch::Tensor::randn([1, 3, 1, 256, 256], (tch::Kind::Float, tch::Device::Cpu));

    let input_min = input.min().double_value(&[]);
    let input_max = input.max().double_value(&[]);
    eprintln!("input: {:?} range=[{input_min:.3}, {input_max:.3}]", input.size());

    // Encode
    let latent = encoder.encode_mean(&input);
    let lat_min = latent.min().double_value(&[]);
    let lat_max = latent.max().double_value(&[]);
    eprintln!("latent: {:?} range=[{lat_min:.3}, {lat_max:.3}]", latent.size());
    assert_eq!(latent.size(), vec![1, 128, 1, 8, 8]);

    // Decode with timestep=0.05 (default decode timestep)
    let timestep = tch::Tensor::from_slice(&[0.05f32]);
    let output = decoder.forward(&latent, &timestep);
    let out_min = output.min().double_value(&[]);
    let out_max = output.max().double_value(&[]);
    eprintln!("output: {:?} range=[{out_min:.3}, {out_max:.3}]", output.size());
    assert_eq!(output.size()[0], 1);
    assert_eq!(output.size()[1], 3);

    // Check no NaN/Inf
    let has_nan = output.isnan().any().double_value(&[]) > 0.0;
    let has_inf = output.isinf().any().double_value(&[]) > 0.0;
    assert!(!has_nan, "output contains NaN");
    assert!(!has_inf, "output contains Inf");

    // Check non-trivial output
    let output_std = output.std(false).double_value(&[]);
    eprintln!("output std: {output_std:.6}");
    assert!(output_std > 0.001, "output is nearly constant (std={output_std})");

    eprintln!("✓ roundtrip: {:?} → {:?} → {:?}", input.size(), latent.size(), output.size());
    drop(no_grad);
}
