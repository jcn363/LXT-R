/// Re-export of `CausalityAxis` from `ltx-conv`.
///
/// The Audio VAE uses the same causality axis selector as the conv primitives.
/// This module re-exports it so downstream code can import from either
/// `ltx_audio_vae::causality::CausalityAxis` or `ltx_conv::CausalityAxis`.
pub use ltx_conv::CausalityAxis;
