use ltx_norm::RMSNorm;
use ltx_test_utils::{assert_allclose, load_golden};

/// Golden test: RMSNorm with all-ones input matches Python reference.
#[test]
fn test_golden_rms_norm_ones() {
    let input = load_golden("crates/goldens/rms_norm_ones.safetensors", "input");
    let expected = load_golden("crates/goldens/rms_norm_ones.safetensors", "output");

    let dim = input.size()[2];
    let norm = RMSNorm::new(dim, 1e-6, tch::Device::Cpu);
    let actual = norm.forward(&input);

    assert_allclose(&actual, &expected, 1e-5, 1e-5);
}

/// Golden test: RMSNorm with random input matches Python reference.
#[test]
fn test_golden_rms_norm_nontrivial() {
    let input = load_golden("crates/goldens/rms_norm_nontrivial.safetensors", "input");
    let expected = load_golden("crates/goldens/rms_norm_nontrivial.safetensors", "output");

    let dim = input.size()[2];
    let norm = RMSNorm::new(dim, 1e-6, tch::Device::Cpu);
    let actual = norm.forward(&input);

    assert_allclose(&actual, &expected, 1e-5, 1e-5);
}
