//! Audio VAE for the LTX-2.3 Rust rewrite.
//!
//! Provides AudioEncoder and AudioDecoder for converting between
//! waveform and latent-space audio representations.

pub mod causality;
pub mod configurator;
pub mod downsample;
pub mod ops;
pub mod upsample;
pub mod vocoder;

pub use causality::CausalityAxis;
pub use configurator::{from_config, AudioDecoder, AudioEncoder, AudioVAEConfig};
pub use downsample::{build_downsampling_path, downsample_forward, DownsampleStage};
pub use ops::{AudioProcessor, PerChannelStatistics};
pub use upsample::{build_upsampling_path, upsample_forward, UpsampleStage};
pub use vocoder::Vocoder;
