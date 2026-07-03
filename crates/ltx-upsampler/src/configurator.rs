use serde::Deserialize;

use crate::LatentUpsampler;

/// Configuration for the latent upsampler, deserializable from JSON.
#[derive(Debug, Clone, Deserialize)]
pub struct LatentUpsamplerConfig {
    /// Number of input channels from the latent space.
    pub in_channels: i64,
    /// Number of output channels after upsampling.
    pub out_channels: i64,
    /// Number of ResNet blocks in the refinement stage.
    pub num_res_blocks: i64,
    /// Channels per ResNet block (grows from in_channels to this).
    pub hidden_channels: i64,
    /// Spatial upscale factor (2 = double resolution).
    pub upscale_factor: i64,
    /// Number of normalization groups for GroupNorm.
    #[serde(default = "default_norm_groups")]
    pub norm_groups: i64,
    /// Number of refinement ResNet blocks after pixel shuffle.
    #[serde(default = "default_refine_blocks")]
    pub refine_blocks: i64,
}

fn default_norm_groups() -> i64 {
    ltx_types::VAE_NORM_NUM_GROUPS
}

fn default_refine_blocks() -> i64 {
    2
}

/// Build a `LatentUpsampler` from a configuration struct.
///
/// This is the single entry point for constructing an upsampler from
/// a config — all model construction goes through here.
pub fn from_config(config: &LatentUpsamplerConfig) -> LatentUpsampler {
    LatentUpsampler::new(
        tch::Device::Cpu,
        config.in_channels,
        config.out_channels,
        config.num_res_blocks,
        config.hidden_channels,
        config.upscale_factor,
        config.norm_groups,
        config.refine_blocks,
    )
}
