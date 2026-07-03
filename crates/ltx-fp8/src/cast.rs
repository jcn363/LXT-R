use tch::Tensor;

/// Calculate FP8 weight with stochastic rounding.
///
/// CPU implementation: simple add + clamp (no stochastic rounding on CPU).
/// THE ONLY FP8 stochastic rounding cast in the LTX codebase.
pub fn calculate_weight_float8(target: &Tensor, original: &Tensor) -> Tensor {
    // CPU fallback: simple add + clamp (no stochastic rounding on CPU)
    let added = target + original;
    added.clamp(ltx_types::FP8_MIN, ltx_types::FP8_MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cast_clamps_to_fp8_range() {
        let target = Tensor::from_slice(&[100.0f32, 200.0, 300.0]);
        let original = Tensor::from_slice(&[50.0f32, 50.0, 50.0]);
        let result = calculate_weight_float8(&target, &original);
        let r = result.to_kind(tch::Kind::Float);

        assert!(r.max().double_value(&[]) <= ltx_types::FP8_MAX + 0.01);
        assert!(r.min().double_value(&[]) >= ltx_types::FP8_MIN - 0.01);
    }

    #[test]
    fn test_cast_adds_target_and_original() {
        let target = Tensor::from_slice(&[1.0f32, 2.0]);
        let original = Tensor::from_slice(&[3.0f32, 4.0]);
        let result = calculate_weight_float8(&target, &original);
        let expected = Tensor::from_slice(&[4.0f32, 6.0]);
        assert!(expected.allclose(&result, 1e-6, 1e-6, false));
    }
}
