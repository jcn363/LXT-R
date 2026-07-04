/// Smoke test: verify all sub-crate re-exports are accessible from ltx_core.
/// Each test just accesses a type or constant to confirm the re-export compiles.

#[test]
fn test_re_export_ltx_types() {
    let _ = ltx_core::ltx_types::NORM_EPS;
}

#[test]
fn test_re_export_ltx_norm() {
    let _ = ltx_core::ltx_norm::RMSNorm::default_eps(64, tch::Device::Cpu);
}

#[test]
fn test_re_export_ltx_attention() {
    let _ = ltx_core::ltx_attention::RopeType::Interleaved;
}

#[test]
fn test_re_export_ltx_upsampler() {
    let _ = ltx_core::ltx_upsampler::PixelShuffleND::new(2, 3);
}

#[test]
fn test_re_export_ltx_quantization() {
    let _ = ltx_core::ltx_quantization::QuantizationPolicy::default();
}

#[test]
fn test_re_export_ltx_transformer() {
    let _ = ltx_core::ltx_transformer::from_config;
}

#[test]
fn test_re_export_ltx_text_encoder() {
    let _ = ltx_core::ltx_text_encoder::prompt_enhancement::PromptEnhancer::new();
}
