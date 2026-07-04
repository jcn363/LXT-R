//! Normalization layers for the LTX-2.3 Rust rewrite.
//!
//! Provides RMSNorm, PixelNorm, GroupNorm, and a factory function
//! to instantiate the appropriate norm layer based on configuration.

pub mod factory;
pub mod group_norm;
pub mod pixel_norm;
pub mod rms_norm;

pub use factory::build_norm_layer;
pub use group_norm::GroupNorm;
pub use pixel_norm::PixelNorm;
pub use rms_norm::RMSNorm;
