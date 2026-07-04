use ltx_types::{FP8_MAX, FP8_MIN, STABILITY_EPS};
use tch::Tensor;

/// Quantize weight to FP8 E4M3FN per tensor.
///
/// Returns `(quantized_weight, inverse_scale)` where:
/// - `quantized_weight` has values clamped to `[FP8_MIN, FP8_MAX]`
/// - `inverse_scale` can be multiplied with the quantized weight to recover the original range
///
/// THE ONLY FP8 quantization implementation in the LTX codebase.
pub fn quantize_weight_to_fp8_per_tensor(weight: &Tensor) -> (Tensor, Tensor) {
    let f32 = weight.to_kind(tch::Kind::Float);

    // Per-tensor max absolute value, clamped for numerical stability
    let max_abs = f32.abs().max().clamp_min(STABILITY_EPS);

    // Scale: maps the tensor's dynamic range into FP8 range
    let scale = max_abs.full_like(FP8_MAX) / &max_abs;

    // Quantize: scale, clamp to FP8 range, store as BF16 (FP8 storage proxy)
    let q = (&f32 * &scale)
        .clamp(FP8_MIN, FP8_MAX)
        .to_kind(tch::Kind::BFloat16);

    // Inverse scale for dequantization
    let inv_scale = max_abs.full_like(1.0) / &scale;

    (q, inv_scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn test_quantize_round_trip() {
        let w = Tensor::randn([64, 128], (tch::Kind::Float, Device::Cpu));
        let (q, inv_scale) = quantize_weight_to_fp8_per_tensor(&w);

        // Dequantize should approximate original
        let recovered = q.to_kind(tch::Kind::Float) * &inv_scale;
        let original = w.to_kind(tch::Kind::Float);

        // Max error bounded by FP8 quantization step
        let max_err = (&original - &recovered).abs().max();
        assert!(
            max_err.double_value(&[]) < 1.0,
            "quantization error too large"
        );
    }

    #[test]
    fn test_quantize_clamps_to_fp8_range() {
        // Create a tensor with values exceeding FP8 range
        let w = Tensor::from_slice(&[-1000.0, FP8_MIN, 0.0, FP8_MAX, 1000.0]);
        let (q, _) = quantize_weight_to_fp8_per_tensor(&w);
        let q_f32 = q.to_kind(tch::Kind::Float);

        // All values should be within FP8 range
        assert!(q_f32.max().double_value(&[]) <= FP8_MAX + 0.01);
        assert!(q_f32.min().double_value(&[]) >= FP8_MIN - 0.01);
    }

    #[test]
    fn test_quantize_uses_sot_constants() {
        let w = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        let (q, inv_scale) = quantize_weight_to_fp8_per_tensor(&w);

        // The max value in quantized output should be <= FP8_MAX
        let q_f32 = q.to_kind(tch::Kind::Float);
        assert!(q_f32.max().double_value(&[]) <= FP8_MAX + 0.01);

        // Inv scale should be positive
        assert!(inv_scale.double_value(&[]) > 0.0);
    }

    // ── Numerical sanity tests ──────────────────────────────────────────

    /// Quantize all-zero tensor should not produce NaN.
    #[test]
    fn test_quantize_all_zeros() {
        let w = Tensor::zeros([16, 32], (tch::Kind::Float, Device::Cpu));
        let (q, inv_scale) = quantize_weight_to_fp8_per_tensor(&w);
        assert_eq!(q.to_kind(tch::Kind::Float).isnan().any().double_value(&[]), 0.0);
        assert!(inv_scale.double_value(&[]) > 0.0);
    }

    /// Quantize very large values should clamp, not produce Inf.
    #[test]
    fn test_quantize_extreme_values() {
        let w = Tensor::from_slice(&[1e6, -1e6, 1e-6, -1e-6]);
        let (q, inv_scale) = quantize_weight_to_fp8_per_tensor(&w);
        let q_f32 = q.to_kind(tch::Kind::Float);
        assert!(q_f32.isfinite().all().double_value(&[]) > 0.0);
        assert!(inv_scale.isfinite().all().double_value(&[]) > 0.0);
    }

    /// Quantize single element should work.
    #[test]
    fn test_quantize_single_element() {
        let w = Tensor::from_slice(&[3.14]);
        let (q, inv_scale) = quantize_weight_to_fp8_per_tensor(&w);
        assert_eq!(q.size(), vec![1]);
        assert!(inv_scale.double_value(&[]) > 0.0);
    }
}
