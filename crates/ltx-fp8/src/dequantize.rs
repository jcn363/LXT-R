use tch::Tensor;

/// Dequantize FP8 weight back to full precision.
///
/// `scale` is the inverse scale from `quantize_weight_to_fp8_per_tensor`.
/// `target` specifies the output dtype (e.g., `tch::Kind::Float`, `tch::Kind::BFloat16`).
///
/// THE ONLY FP8 dequantization implementation in the LTX codebase.
pub fn dequantize_fp8(weight: &Tensor, scale: &Tensor, target: tch::Kind) -> Tensor {
    (weight.to_kind(tch::Kind::Float) * scale.to_kind(tch::Kind::Float))
        .to_kind(target)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ltx_types::{FP8_MAX, FP8_MIN};

    #[test]
    fn test_dequantize_recovers_original() {
        let original = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]);
        let scale = Tensor::from_slice(&[0.5f32]);
        let inv_scale = Tensor::from_slice(&[2.0f32]); // 1.0 / 0.5
        let quantized = (&original * &scale).clamp(FP8_MIN, FP8_MAX);

        let recovered = dequantize_fp8(&quantized, &inv_scale, tch::Kind::Float);
        assert!(original.allclose(&recovered, 1e-6, 1e-6, false));
    }

    #[test]
    fn test_dequantize_to_bf16() {
        let w = Tensor::from_slice(&[1.0f32, 2.0]);
        let s = Tensor::from_slice(&[1.0f32]);
        let out = dequantize_fp8(&w, &s, tch::Kind::BFloat16);
        assert_eq!(out.kind(), tch::Kind::BFloat16);
    }
}
