use ltx_norm::RMSNorm;
use tch::Tensor;

/// Golden test for RMSNorm.
///
/// This test verifies the Rust RMSNorm implementation matches hand-computed
/// expected values. When Python golden files are available, replace the
/// expected values with `load_golden("crates/goldens/rms_norm.safetensors", "output")`.
#[test]
fn test_golden_rms_norm_values() {
    let dim = 4;
    let norm = RMSNorm::new(dim, 1e-6, tch::Device::Cpu);

    // Hand-computed input: all ones
    let x = Tensor::ones([1, 2, dim], (tch::Kind::Float, tch::Device::Cpu));
    let out = norm.forward(&x);

    // RMSNorm: x / sqrt(mean(x^2) + eps) * weight
    // mean(ones^2) = 1.0, sqrt(1.0 + 1e-6) ≈ 1.0, so output ≈ ones
    assert_eq!(out.size(), vec![1, 2, dim]);
    let max_diff = (&out - &x).abs().max().double_value(&[]);
    assert!(
        max_diff < 1e-4,
        "RMSNorm golden test: max diff = {max_diff}, expected < 1e-4"
    );
}

/// Verify RMSNorm with non-trivial input matches expected output.
#[test]
fn test_golden_rms_norm_nontrivial() {
    let dim = 4;
    let norm = RMSNorm::new(dim, 0.0, tch::Device::Cpu);

    // Input: [1, 2, 3, 4] — RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
    let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]).reshape([1, 1, dim]);
    let out = norm.forward(&x);

    // Expected: [1, 2, 3, 4] / 2.7386 * weight(≈1.0)
    let expected = Tensor::from_slice(&[1.0 / 2.7386, 2.0 / 2.7386, 3.0 / 2.7386, 4.0 / 2.7386])
        .reshape([1, 1, dim]);

    let max_diff = (&out - &expected).abs().max().double_value(&[]);
    assert!(
        max_diff < 1e-3,
        "RMSNorm nontrivial golden test: max diff = {max_diff}"
    );
}
