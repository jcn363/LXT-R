use std::borrow::Borrow;

use serde::Deserialize;
use tch::nn::Path;

use ltx_types::{NormLayerType, VAE_NORM_NUM_GROUPS};

use crate::{VideoDecoder, VideoEncoder, VideoVAE};

#[derive(Debug, Clone, Deserialize)]
pub struct VideoVAEConfig {
    pub in_channels: i64,
    pub base_channels: i64,
    pub channel_multipliers: Vec<i64>,
    pub num_res_blocks: i64,
    pub latent_channels: i64,
    #[serde(default = "default_norm_groups")]
    pub norm_num_groups: i64,
    #[serde(default)]
    pub causal: bool,
    #[serde(default = "default_norm_type")]
    pub norm_type: NormLayerType,
    #[serde(default = "default_spatial_downsample")]
    pub spatial_downsample_factor: i64,
}

fn default_norm_groups() -> i64 {
    VAE_NORM_NUM_GROUPS
}

fn default_norm_type() -> NormLayerType {
    NormLayerType::Group
}

fn default_spatial_downsample() -> i64 {
    8
}

/// Build `VideoVAE` from a config. This is the ONLY configurator for the
/// video VAE — it wires the encoder and decoder from the same config.
pub fn from_config<'a>(config: &VideoVAEConfig, vs: impl Borrow<Path<'a>>) -> VideoVAE {
    let vs = vs.borrow();

    let encoder = VideoEncoder::new(
        vs / "encoder",
        config.in_channels,
        config.base_channels,
        &config.channel_multipliers,
        config.num_res_blocks,
        config.latent_channels,
        config.norm_num_groups,
        config.causal,
        config.norm_type,
    );

    let decoder = VideoDecoder::new(
        vs / "decoder",
        config.latent_channels,
        config.base_channels,
        &config.channel_multipliers,
        config.num_res_blocks,
        config.in_channels,
        config.norm_num_groups,
        config.causal,
        config.norm_type,
    );

    VideoVAE {
        encoder,
        decoder,
        spatial_downsample_factor: config.spatial_downsample_factor,
    }
}
