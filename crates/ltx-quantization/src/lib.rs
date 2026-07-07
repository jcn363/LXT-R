//! Quantization for the LTX-2.3 Rust rewrite.
//!
//! Provides quantized linear layers (FP8, INT4) and QuantizationPolicy
//! for controlling model quantization behavior.
//!
//! # Quantization Levels
//!
//! | Level | Bits | Compression | Quality | Use Case |
//! |-------|------|-------------|---------|----------|
//! | FP32  | 32   | 1x          | Best    | Training, reference |
//! | BF16  | 16   | 2x          | Excellent | Default inference |
//! | FP8   | 8    | 4x          | Good    | GPU inference |
//! | INT4  | 4    | ~8x         | Fair    | Edge/low-VRAM inference |
//!
//! INT4 uses per-group quantization with default group_size=128, giving
//! ~0.508 bytes per parameter (vs 4 bytes for FP32).

pub mod fp8_mm;
pub mod int4_mm;
pub mod int8_mm;
pub mod policy;

pub use fp8_mm::FP8Linear;
pub use int4_mm::INT4Linear;
pub use int8_mm::{quantize_to_int8_per_tensor, dequantize_int8};
pub use policy::QuantizationPolicy;
