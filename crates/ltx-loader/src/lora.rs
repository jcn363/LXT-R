use std::collections::HashMap;
use tch::Tensor;

use ltx_fp8::quantize_weight_to_fp8_per_tensor;
use ltx_types::{DType, LORA_DELTAS_DTYPE_IF_FP8};

/// Apply LoRA deltas to a state dict.
///
/// `base_weights` - the base model state dict
/// `lora_deltas` - LoRA weight deltas keyed by parameter name
/// `alpha` - LoRA scaling factor
/// `target_dtype` - whether the base model is FP8
///
/// Returns a new state dict with LoRA deltas applied.
pub fn apply_loras(
    base_weights: &mut HashMap<String, Tensor>,
    lora_deltas: &HashMap<String, Tensor>,
    alpha: f64,
    is_fp8: bool,
) {
    let delta_dtype = if is_fp8 {
        DType::parse(LORA_DELTAS_DTYPE_IF_FP8)
            .unwrap_or(DType::BFloat16)
            .to_tch_kind()
    } else {
        tch::Kind::Float
    };

    for (name, delta) in lora_deltas {
        if let Some(weight) = base_weights.get_mut(name) {
            let delta_f = delta.to_kind(delta_dtype);
            let scale = Tensor::from_slice(&[alpha as f32])
                .to_kind(delta_dtype)
                .to_device(weight.device());

            if is_fp8 {
                // For FP8 models: add scaled delta then requantize
                let w_f8 = weight.to_kind(tch::Kind::BFloat16);
                let updated = w_f8 + &delta_f * &scale;
                let (q, inv_scale) = quantize_weight_to_fp8_per_tensor(&updated);
                *weight = q;
                // Store the inverse scale alongside (convention: weight + "_inv_scale")
                let inv_key = format!("{name}_inv_scale");
                base_weights.insert(inv_key, inv_scale);
            } else {
                let w_f = weight.to_kind(tch::Kind::Float);
                *weight = (w_f + &delta_f * &scale).to_kind(weight.kind());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_apply_loras_fp32() {
        let mut base = HashMap::new();
        base.insert(
            "layer.weight".to_string(),
            Tensor::ones([4, 4], (tch::Kind::Float, tch::Device::Cpu)),
        );

        let mut deltas = HashMap::new();
        deltas.insert(
            "layer.weight".to_string(),
            Tensor::ones([4, 4], (tch::Kind::Float, tch::Device::Cpu)) * 0.1,
        );

        apply_loras(&mut base, &deltas, 1.0, false);

        let w = base.get("layer.weight").unwrap();
        // Should be 1.0 + 0.1 * 1.0 = 1.1
        assert!((w.mean(tch::Kind::Float).double_value(&[]) - 1.1).abs() < 1e-5);
    }

    #[test]
    fn test_apply_loras_skips_missing() {
        let mut base = HashMap::new();
        base.insert(
            "a.weight".to_string(),
            Tensor::ones([2, 2], (tch::Kind::Float, tch::Device::Cpu)),
        );

        let mut deltas = HashMap::new();
        deltas.insert(
            "b.weight".to_string(),
            Tensor::ones([2, 2], (tch::Kind::Float, tch::Device::Cpu)),
        );

        apply_loras(&mut base, &deltas, 1.0, false);
        // base should be unchanged
        assert_eq!(base.len(), 1);
    }
}
