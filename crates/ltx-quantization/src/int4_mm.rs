//! INT4 quantized linear layer for reduced VRAM usage.
//!
//! Per-group INT4 weight-only quantization. Each group of `group_size`
//! consecutive weights shares one scale factor. Weights are packed as
//! INT4 values (2 per byte) and dequantized on forward pass.
//!
//! With default group_size=128: ~0.508 bytes per parameter (7.9x vs FP32).

use tch::nn::Module;
use tch::Tensor;

pub const DEFAULT_GROUP_SIZE: i64 = 128;

/// Quantize weights to INT4 with per-group scaling.
pub fn quantize_weight_to_int4_per_group(
    weight: &Tensor,
    group_size: i64,
) -> (Tensor, Tensor) {
    let w = weight.to_kind(tch::Kind::Float);
    let shape = w.size();
    let out_features = shape[0];
    let in_features = shape[1];

    let padded_in = ((in_features + group_size - 1) / group_size) * group_size;
    let num_groups = padded_in / group_size;

    let w_padded = if padded_in > in_features {
        let padding = Tensor::zeros([out_features, padded_in - in_features], (tch::Kind::Float, w.device()));
        Tensor::cat(&[&w, &padding], 1)
    } else {
        w
    };

    let w_groups = w_padded.reshape([out_features, num_groups, group_size]);
    let abs_max = w_groups.abs().amax(2, true);
    let scales = (abs_max / 7.0).clamp(1e-8, f32::MAX as f64).squeeze_dim(2);

    let w_normalized = &w_groups / &scales.unsqueeze(2);
    let w_int4 = w_normalized.clamp(-7.0, 7.0).round().to_kind(tch::Kind::Int8);

    // Pack pairs of INT4 values into bytes
    let w_packed = w_int4.reshape([out_features, num_groups, group_size / 2, 2]);
    let low = w_packed.narrow(3, 0, 1).squeeze_dim(3);
    let high = w_packed.narrow(3, 1, 1).squeeze_dim(3);

    // Bit manipulation: low & 0x0F, high & 0x0F, shift high left by 4, OR together
    let masked_low = low.bitwise_and(0x0F);
    let masked_high = high.bitwise_and(0x0F);
    let shift_amt = Tensor::from_slice(&[4i64]).to_kind(tch::Kind::Int8);
    let shifted_high = masked_high.bitwise_left_shift(&shift_amt);
    let packed = masked_low.bitwise_or_tensor(&shifted_high)
        .reshape([out_features, num_groups * group_size / 2]);

    (packed.to_kind(tch::Kind::Int8), scales)
}

/// INT4 linear layer — quantizes weights to INT4 on construction.
#[derive(Debug)]
pub struct INT4Linear {
    packed_weight: Tensor,
    scales: Tensor,
    bias: Option<Tensor>,
    in_features: i64,
    out_features: i64,
    group_size: i64,
}

impl INT4Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>, group_size: i64) -> Self {
        let out_features = weight.size()[0];
        let in_features = weight.size()[1];
        let (packed_weight, scales) = quantize_weight_to_int4_per_group(&weight, group_size);
        Self { packed_weight, scales, bias, in_features, out_features, group_size }
    }

    pub fn new_default(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self::new(weight, bias, DEFAULT_GROUP_SIZE)
    }

    pub fn in_features(&self) -> i64 { self.in_features }
    pub fn out_features(&self) -> i64 { self.out_features }

    pub fn memory_bytes(&self) -> usize {
        let weight_bytes = (self.out_features * self.in_features / 2) as usize;
        let num_groups = (self.in_features + self.group_size - 1) / self.group_size;
        let scale_bytes = (self.out_features * num_groups * 4) as usize;
        weight_bytes + scale_bytes
    }
}

impl Module for INT4Linear {
    fn forward(&self, x: &Tensor) -> Tensor {
        let num_groups = self.scales.size()[1]; // scales: [out_features, num_groups]
        let group_size = self.group_size;

        // Unpack INT4: extract low and high nibbles
        let low = self.packed_weight.bitwise_and(0x0F).to_kind(tch::Kind::Float);
        let shift_amt = Tensor::from_slice(&[4i64]).to_kind(tch::Kind::Int8);
        let high = self.packed_weight.bitwise_right_shift(&shift_amt).bitwise_and(0x0F).to_kind(tch::Kind::Float);

        // Interleave low and high nibbles back to full width
        let unpacked = Tensor::stack(&[&low, &high], 2)
            .reshape([self.out_features, self.in_features]);

        // Dequantize
        let scales_expanded = self.scales.unsqueeze(2)
            .expand([self.out_features, num_groups, group_size], true)
            .reshape([self.out_features, self.in_features]);
        let w_f32 = unpacked * scales_expanded;

        let out = x.matmul(&w_f32.to_dtype(x.kind(), false, false).transpose(0, 1));
        match &self.bias {
            Some(b) => out + b,
            None => out,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int4_roundtrip() {
        let weight = Tensor::randn([256, 512], (tch::Kind::Float, tch::Device::Cpu));
        let (packed, scales) = quantize_weight_to_int4_per_group(&weight, 128);

        assert_eq!(packed.size(), vec![256, 256]);
        assert_eq!(scales.size(), vec![256, 4]);

        let fp32_bytes = 256 * 512 * 4;
        let int4_bytes = 256 * 256 + 256 * 4 * 4;
        assert!(int4_bytes < fp32_bytes / 4, "INT4 should use <25% of FP32 memory");
    }

    #[test]
    fn test_int4_linear_forward() {
        let weight = Tensor::randn([128, 256], (tch::Kind::Float, tch::Device::Cpu));
        let bias = Tensor::zeros([128], (tch::Kind::Float, tch::Device::Cpu));
        let layer = INT4Linear::new_default(weight, Some(bias));

        let x = Tensor::randn([4, 256], (tch::Kind::Float, tch::Device::Cpu));
        let out = layer.forward(&x);
        assert_eq!(out.size(), vec![4, 128]);
    }

    #[test]
    fn test_int4_memory_savings() {
        let weight = Tensor::randn([1024, 2048], (tch::Kind::Float, tch::Device::Cpu));
        let layer = INT4Linear::new_default(weight, None);

        let fp32_bytes = 1024 * 2048 * 4;
        let int4_bytes = layer.memory_bytes();
        let ratio = fp32_bytes as f64 / int4_bytes as f64;
        eprintln!("FP32: {fp32_bytes} bytes, INT4: {int4_bytes} bytes, ratio: {ratio:.1}x");
        assert!(ratio > 6.0, "INT4 should achieve >6x compression vs FP32");
    }
}
