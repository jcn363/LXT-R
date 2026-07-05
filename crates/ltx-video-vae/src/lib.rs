//! Video VAE (Variational Autoencoder) for the LTX-2.3 Rust rewrite.
//!
//! Provides VideoEncoder and VideoDecoder for converting between
//! pixel-space and latent-space video representations.

pub mod configurator;
pub mod decoder_blocks;
pub mod encoder_blocks;
pub mod sampling;

use std::borrow::Borrow;

use tch::nn::{ModuleT, Path};
use tch::Tensor;

use ltx_conv::make_conv_nd;
use ltx_norm::build_norm_layer;
use ltx_resblock::{make_resblock, UNetMidBlock3D};
use ltx_types::NormLayerType;

use sampling::{depth_to_space, space_to_depth};

pub use configurator::{from_config, default_encoder_block_descs, VideoVAEConfig};
pub use decoder_blocks::make_decoder_block;

use encoder_blocks::EncoderStage;

// ---------------------------------------------------------------------------
// VideoEncoder — matches Python LTX-Video checkpoint architecture
// ---------------------------------------------------------------------------

/// Video VAE encoder: pixel-space `(B,3,T,H,W)` → latent distribution.
///
/// Architecture (from `ltx-video-2b-v0.9.1.safetensors`):
/// - `space_to_depth(r=4)`: 3 → 48 channels
/// - `conv_in`: 48 → 128
/// - 10 heterogeneous `down_blocks`:
///   - ResBlock stages (0, 3, 6, 8, 9)
///   - Stride-2 convs (1, 4, 7)
///   - Channel-change downsamples (2, 5)
/// - `conv_out`: 512 → 129 (mean 64 + logvar 64 + scale 1)
/// - No mid block, no conv_norm_out, no timestep conditioning
pub struct VideoEncoder {
    conv_in: Box<dyn ModuleT>,
    blocks: Vec<EncoderStage>,
    conv_out: Box<dyn ModuleT>,
}

impl std::fmt::Debug for VideoEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VideoEncoder").finish()
    }
}

/// Descriptor for one encoder down-block, used by the config.
pub struct EncoderBlockDesc {
    pub kind: EncoderBlockKind,
    pub in_ch: i64,
    pub out_ch: i64,
}

pub enum EncoderBlockKind {
    ResBlocks(i64),           // num resblocks
    DownsampleConv,           // stride-2 conv
    ChannelChangeDownsample,  // stride-2 + channel doubling + shortcut + norm
}

impl VideoEncoder {
    pub fn new(
        vs: &tch::nn::Path,
        conv_in_channels: i64,   // 48 (from space_to_depth r=4)
        base_channels: i64,      // 128
        block_descs: &[EncoderBlockDesc],
        conv_out_channels: i64,  // 129
        norm_type: NormLayerType,
        norm_groups: i64,
        causal: bool,
    ) -> Self {
        let conv_in = make_conv_nd(
            vs / "conv_in",
            3,
            conv_in_channels,
            base_channels,
            3, 1, 1, causal, "zeros",
        );

        let mut blocks = Vec::new();
        for (i, desc) in block_descs.iter().enumerate() {
            let block_vs = vs / format!("down_blocks.{i}");
            let stage = match &desc.kind {
                EncoderBlockKind::ResBlocks(n) => {
                    EncoderStage::ResBlocks(
                        encoder_blocks::make_resblock_stage(
                            &block_vs, desc.out_ch, *n,
                            norm_type, norm_groups, causal,
                        )
                    )
                }
                EncoderBlockKind::DownsampleConv => {
                    EncoderStage::DownsampleConv(
                        encoder_blocks::make_downsample_conv(
                            &block_vs, desc.in_ch, causal,
                        )
                    )
                }
                EncoderBlockKind::ChannelChangeDownsample => {
                    EncoderStage::ChannelChange(
                        encoder_blocks::ChannelChangeDownsample::new(
                            &block_vs, desc.in_ch, desc.out_ch,
                            norm_type, norm_groups, causal,
                        )
                    )
                }
            };
            blocks.push(stage);
        }

        // conv_out: last block's output channels → conv_out_channels
        let last_out = block_descs.last().map(|d| d.out_ch).unwrap_or(base_channels);
        let conv_out = make_conv_nd(
            vs / "conv_out",
            3,
            last_out,
            conv_out_channels,
            3, 1, 1, causal, "zeros",
        );

        Self { conv_in, blocks, conv_out }
    }

    /// Encode pixel-space video to distribution parameters.
    ///
    /// Returns raw conv_out output of shape `(B, 129, T', H', W')`.
    /// Callers split into mean/logvar/scale as needed.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let x = space_to_depth(x, 4);
        let mut h = self.conv_in.forward_t(&x, false);
        for block in &self.blocks {
            h = block.forward(&h);
        }
        self.conv_out.forward_t(&h, false)
    }

    /// Encode and sample: returns `(B, 128, T', H', W')` latent.
    ///
    /// Splits the 129-channel output into mean(64) + logvar(64) + scale(1),
    /// then reparameterizes: `latent = mean + exp(0.5 * logvar) * noise`.
    /// The scale channel is discarded.
    pub fn encode(&self, x: &Tensor) -> Tensor {
        let raw = self.forward(x);
        let mean = raw.narrow(1, 0, 64);
        let logvar = raw.narrow(1, 64, 64);
        let std = (logvar * 0.5).exp();
        let noise = Tensor::randn_like(&mean);
        mean + std * noise
    }

    /// Encode and return the mean only (deterministic, for img2img).
    pub fn encode_mean(&self, x: &Tensor) -> Tensor {
        let raw = self.forward(x);
        raw.narrow(1, 0, 64)
    }
}

impl ModuleT for VideoEncoder {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        self.forward(xs)
    }
}

// ---------------------------------------------------------------------------
// VideoDecoder
// ---------------------------------------------------------------------------

/// Video VAE decoder: latent-space 5D tensor → pixel-space 5D tensor.
pub struct VideoDecoder {
    conv_in: Box<dyn ModuleT>,
    mid: UNetMidBlock3D,
    up_convs: Vec<Box<dyn ModuleT>>,
    up_resblocks: Vec<Vec<Box<dyn ModuleT>>>,
    conv_norm_out: Box<dyn ModuleT>,
    conv_out: Box<dyn ModuleT>,
    spatial_upsample_factor: i64,
}

impl std::fmt::Debug for VideoDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VideoDecoder").finish()
    }
}

impl VideoDecoder {
    #[allow(clippy::too_many_arguments)]
    pub fn new<'a>(
        vs: impl Borrow<Path<'a>>,
        latent_channels: i64,
        base_channels: i64,
        channel_multipliers: &[i64],
        num_res_blocks: i64,
        in_channels: i64,
        norm_num_groups: i64,
        causal: bool,
        norm_type: NormLayerType,
    ) -> Self {
        let vs = vs.borrow();

        let last_mult = *channel_multipliers
            .last()
            .expect("channel_multipliers should not be empty");
        let mid_channels = base_channels * last_mult;

        let conv_in = make_conv_nd(
            vs / "conv_in",
            3,
            latent_channels,
            mid_channels,
            3,
            1,
            1,
            causal,
            "zeros",
        );

        let mid = UNetMidBlock3D::new(
            vs / "mid",
            mid_channels,
            norm_type,
            norm_num_groups,
            causal,
            1,
        );

        let mut up_convs = Vec::new();
        let mut up_resblocks = Vec::new();

        for (i, &mult) in channel_multipliers.iter().enumerate().rev() {
            let ch = base_channels * mult;
            let next_ch = if i == 0 {
                base_channels
            } else {
                base_channels * channel_multipliers[i - 1]
            };

            let block = decoder_blocks::make_decoder_block(
                vs / format!("up_{i}"),
                ch,
                next_ch,
                1,
                norm_type,
                norm_num_groups,
                causal,
            );

            let mut resblock_group = Vec::new();
            for j in 0..num_res_blocks {
                resblock_group.push(make_resblock(
                    3,
                    next_ch,
                    next_ch,
                    norm_type,
                    norm_num_groups,
                    causal,
                    vs / format!("up_{i}") / "resblocks" / j,
                ));
            }

            up_convs.push(block.conv);
            up_resblocks.push(resblock_group);
        }

        let conv_norm_out = build_norm_layer(norm_type, base_channels, norm_num_groups);
        let conv_out = make_conv_nd(
            vs / "conv_out",
            3,
            base_channels,
            in_channels * 4,
            3,
            1,
            1,
            causal,
            "zeros",
        );

        Self {
            conv_in,
            mid,
            up_convs,
            up_resblocks,
            conv_norm_out,
            conv_out,
            spatial_upsample_factor: 2,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut x = self.conv_in.forward_t(x, false);

        x = self.mid.forward(&x);

        for (i, conv) in self.up_convs.iter().enumerate() {
            x = conv.forward_t(&x, false);
            for resblock in &self.up_resblocks[i] {
                x = resblock.forward_t(&x, false);
            }
            x = depth_to_space(&x, self.spatial_upsample_factor);
        }

        let h = self.conv_norm_out.forward_t(&x, false).silu();
        self.conv_out.forward_t(&h, false)
    }
}

impl ModuleT for VideoDecoder {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        self.forward(xs)
    }
}

// ---------------------------------------------------------------------------
// VideoVAE
// ---------------------------------------------------------------------------

/// Complete Video VAE — encoder + decoder with a spatial downsample factor.
pub struct VideoVAE {
    pub(crate) encoder: VideoEncoder,
    pub(crate) decoder: VideoDecoder,
    pub(crate) spatial_downsample_factor: i64,
}

impl std::fmt::Debug for VideoVAE {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VideoVAE")
            .field("spatial_downsample_factor", &self.spatial_downsample_factor)
            .finish()
    }
}

impl VideoVAE {
    #[allow(clippy::too_many_arguments)]
    pub fn new<'a>(
        vs: impl Borrow<Path<'a>>,
        conv_in_channels: i64,   // 48 — from space_to_depth(r=4)
        base_channels: i64,
        block_descs: &[EncoderBlockDesc],
        conv_out_channels: i64,  // 129 — mean(64) + logvar(64) + scale(1)
        norm_type: NormLayerType,
        norm_groups: i64,
        causal: bool,
        // Decoder still uses the old channel_multipliers layout
        channel_multipliers: &[i64],
        num_res_blocks: i64,
        latent_channels: i64,
        in_channels: i64,
        spatial_downsample_factor: i64,
    ) -> Self {
        let vs = vs.borrow();

        let encoder = VideoEncoder::new(
            &(vs / "encoder"),
            conv_in_channels,
            base_channels,
            block_descs,
            conv_out_channels,
            norm_type,
            norm_groups,
            causal,
        );

        let decoder = VideoDecoder::new(
            vs / "decoder",
            latent_channels,
            base_channels,
            channel_multipliers,
            num_res_blocks,
            in_channels,
            norm_groups,
            causal,
            norm_type,
        );

        Self {
            encoder,
            decoder,
            spatial_downsample_factor,
        }
    }

    pub fn encode(&self, x: &Tensor) -> Tensor {
        self.encoder.forward(x)
    }

    pub fn decode(&self, x: &Tensor) -> Tensor {
        self.decoder.forward(x)
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let latent = self.encode(x);
        self.decode(&latent)
    }

    pub fn spatial_downsample_factor(&self) -> i64 {
        self.spatial_downsample_factor
    }
}

impl ModuleT for VideoVAE {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        self.forward(xs)
    }
}
