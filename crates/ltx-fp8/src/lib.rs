//! FP8 (8-bit floating point) operations for the LTX-2.3 Rust rewrite.
//!
//! Provides FP8 quantization, dequantization, stochastic rounding, and cuBLAS FFI
//! for efficient low-precision computation.

pub mod cast;
pub mod cublas;
pub mod dequantize;
pub mod quantize;

pub use cast::calculate_weight_float8;
pub use cublas::CublasFp8Handle;
pub use dequantize::dequantize_fp8;
pub use quantize::quantize_weight_to_fp8_per_tensor;
