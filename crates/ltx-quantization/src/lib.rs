//! Quantization for the LTX-2.3 Rust rewrite.
//!
//! Provides FP8Linear (quantized linear layer) and QuantizationPolicy
//! for controlling model quantization behavior.

pub mod fp8_mm;
pub mod policy;

pub use fp8_mm::FP8Linear;
pub use policy::QuantizationPolicy;
