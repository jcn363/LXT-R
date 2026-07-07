//! INT8 per-tensor symmetric weight-only quantization.
//!
//! Quantizes a float weight tensor to INT8 with a single scale factor per tensor.
//! Dequantization multiplies INT8 values by the scale to recover approximate FP32 weights.
//!
//! Compression: 4× vs FP32 (1 byte per param + 4 bytes for scale).

use tch::Tensor;

/// Quantize a float tensor to INT8 with per-tensor symmetric scaling.
///
/// Returns (quantized_weights, scale) where:
/// - quantized_weights: Int8 tensor of same shape
/// - scale: Float scalar = max(abs(weight)) / 127.0
pub fn quantize_to_int8_per_tensor(weight: &Tensor) -> (Tensor, Tensor) {
    let w = weight.to_kind(tch::Kind::Float);
    let abs_max = w.abs().amax(0, false); // scalar
    let scale = (&abs_max / 127.0).clamp(1e-8, f32::MAX as f64);
    let w_q = (&w / &scale).clamp(-128.0, 127.0).round().to_kind(tch::Kind::Int8);
    (w_q, scale)
}

/// Dequantize INT8 weights back to float.
pub fn dequantize_int8(weight_q: &Tensor, scale: &Tensor, _target_kind: tch::Kind) -> Tensor {
    weight_q.to_kind(tch::Kind::Float) * scale
}

/// Compute memory usage for INT8-quantized weight tensor.
/// Returns (int4_bytes, scale_bytes).
pub fn int8_memory_bytes(shape: &[i64]) -> (usize, usize) {
    let num_elements: usize = shape.iter().map(|&s| s as usize).product();
    (num_elements, 4) // 1 byte per element + 4 bytes for scalar scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int8_roundtrip() {
        let w = Tensor::randn([256, 512], (tch::Kind::Float, tch::Device::Cpu));
        let (w_q, scale) = quantize_to_int8_per_tensor(&w);
        assert_eq!(w_q.kind(), tch::Kind::Int8);
        assert_eq!(w_q.size(), w.size());

        let w_deq = dequantize_int8(&w_q, &scale, tch::Kind::Float);
        let max_err = (&w - &w_deq).abs().max().double_value(&[]);
        eprintln!("INT8 max error: {max_err:.6}");
        assert!(max_err < 0.05, "INT8 roundtrip error too large: {max_err}");
    }

    #[test]
    fn test_int8_memory_savings() {
        let shape = [1024, 2048];
        let (int8_bytes, scale_bytes) = int8_memory_bytes(&shape);
        let fp32_bytes: usize = shape.iter().map(|&s| s as usize).product::<usize>() * 4;
        let total = int8_bytes + scale_bytes;
        let ratio = fp32_bytes as f64 / total as f64;
        eprintln!("FP32: {fp32_bytes} bytes, INT8: {total} bytes, ratio: {ratio:.1}x");
        assert!(ratio > 3.5, "INT8 should achieve >3.5x compression vs FP32");
    }
}
