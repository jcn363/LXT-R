use std::borrow::Borrow;

use serde::Deserialize;
use tch::nn::Path;

use ltx_types::{NormLayerType, VAE_NORM_NUM_GROUPS};

use crate::{
    EncoderBlockDesc, EncoderBlockKind, VideoDecoder, VideoEncoder, VideoVAE,
};

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

/// Default encoder block descriptors matching the Python LTX-Video VAE.
///
/// 10 blocks:
///   0: ResBlocks x4, ch=128
///   1: DownsampleConv, in=128, out=128
///   2: ChannelChangeDownsample, in=128, out=256
///   3: ResBlocks x3, ch=256
///   4: DownsampleConv, in=256, out=256
///   5: ChannelChangeDownsample, in=256, out=512
///   6: ResBlocks x3, ch=512
///   7: DownsampleConv, in=512, out=512
///   8: ResBlocks x3, ch=512
///   9: ResBlocks x4, ch=512
pub fn default_encoder_block_descs() -> Vec<EncoderBlockDesc> {
    use EncoderBlockKind::*;
    vec![
        EncoderBlockDesc { kind: ResBlocks(4), in_ch: 128, out_ch: 128 },
        EncoderBlockDesc { kind: DownsampleConv, in_ch: 128, out_ch: 128 },
        EncoderBlockDesc { kind: ChannelChangeDownsample, in_ch: 128, out_ch: 256 },
        EncoderBlockDesc { kind: ResBlocks(3), in_ch: 256, out_ch: 256 },
        EncoderBlockDesc { kind: DownsampleConv, in_ch: 256, out_ch: 256 },
        EncoderBlockDesc { kind: ChannelChangeDownsample, in_ch: 256, out_ch: 512 },
        EncoderBlockDesc { kind: ResBlocks(3), in_ch: 512, out_ch: 512 },
        EncoderBlockDesc { kind: DownsampleConv, in_ch: 512, out_ch: 512 },
        EncoderBlockDesc { kind: ResBlocks(3), in_ch: 512, out_ch: 512 },
        EncoderBlockDesc { kind: ResBlocks(4), in_ch: 512, out_ch: 512 },
    ]
}

/// conv_in_channels after space_to_depth with r=4 on 3-channel RGB.
pub const SPACE_TO_DEPTH_R: i64 = 4;
pub const CONV_IN_CHANNELS: i64 = 3 * SPACE_TO_DEPTH_R * SPACE_TO_DEPTH_R; // 48
/// conv_out_channels for the encoder (129 = 128 sampled latent + 1 scale).
pub const ENCODER_CONV_OUT_CHANNELS: i64 = 129;
/// Latent channels after sampling (mean from the first 128 of 129).
pub const SAMPLED_LATENT_CHANNELS: i64 = 128;

/// Build `VideoVAE` from a config. This is the ONLY configurator for the
/// video VAE — it wires the encoder and decoder from the same config.
pub fn from_config<'a>(config: &VideoVAEConfig, vs: impl Borrow<Path<'a>>) -> VideoVAE {
    let vs = vs.borrow();

    let block_descs = default_encoder_block_descs();

    let encoder = VideoEncoder::new(
        &(vs / "encoder"),
        CONV_IN_CHANNELS,       // 48 — from space_to_depth(r=4)
        config.base_channels,
        &block_descs,
        ENCODER_CONV_OUT_CHANNELS, // 129 — raw conv_out before sampling
        config.norm_type,
        config.norm_num_groups,
        config.causal,
    );

    let decoder = VideoDecoder::new(
        vs / "decoder",
        SAMPLED_LATENT_CHANNELS, // 128 — decoder takes sampled latent, not raw conv_out
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
