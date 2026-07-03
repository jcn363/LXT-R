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

pub use configurator::{from_config, VideoVAEConfig};
pub use decoder_blocks::make_decoder_block;
pub use encoder_blocks::make_encoder_block;

// ---------------------------------------------------------------------------
// VideoEncoder
// ---------------------------------------------------------------------------

/// Video VAE encoder: pixel-space 5D tensor → latent-space 5D tensor.
pub struct VideoEncoder {
    conv_in: Box<dyn ModuleT>,
    down_convs: Vec<Box<dyn ModuleT>>,
    down_resblocks: Vec<Vec<Box<dyn ModuleT>>>,
    mid: UNetMidBlock3D,
    conv_norm_out: Box<dyn ModuleT>,
    conv_out: Box<dyn ModuleT>,
}

impl std::fmt::Debug for VideoEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VideoEncoder").finish()
    }
}

impl VideoEncoder {
    #[allow(clippy::too_many_arguments)]
    pub fn new<'a>(
        vs: impl Borrow<Path<'a>>,
        in_channels: i64,
        base_channels: i64,
        channel_multipliers: &[i64],
        num_res_blocks: i64,
        latent_channels: i64,
        norm_num_groups: i64,
        causal: bool,
        norm_type: NormLayerType,
    ) -> Self {
        let vs = vs.borrow();

        let conv_in = make_conv_nd(vs / "conv_in", 3, in_channels, base_channels, 3, 1, 1, causal, "zeros");

        let mut down_convs = Vec::new();
        let mut down_resblocks = Vec::new();

        for (i, &mult) in channel_multipliers.iter().enumerate() {
            let ch = base_channels * mult;
            let prev_ch = if i == 0 {
                base_channels
            } else {
                base_channels * channel_multipliers[i - 1]
            };
            let is_last = i == channel_multipliers.len() - 1;
            let stride = if is_last { 1 } else { 2 };

            let block = encoder_blocks::make_encoder_block(
                vs / format!("down_{i}"),
                prev_ch,
                ch,
                stride,
                norm_type,
                norm_num_groups,
                causal,
            );

            let mut resblock_group = Vec::new();
            for j in 0..num_res_blocks {
                resblock_group.push(make_resblock(
                    3,
                    ch,
                    ch,
                    norm_type,
                    norm_num_groups,
                    causal,
                    vs / format!("down_{i}") / "resblocks" / j,
                ));
            }

            down_convs.push(block.conv);
            down_resblocks.push(resblock_group);
        }

        let last_mult = *channel_multipliers.last().unwrap();
        let mid_channels = base_channels * last_mult;

        let mid = UNetMidBlock3D::new(
            vs / "mid",
            mid_channels,
            norm_type,
            norm_num_groups,
            causal,
            1,
        );

        let conv_norm_out = build_norm_layer(norm_type, mid_channels, norm_num_groups);
        let conv_out = make_conv_nd(vs / "conv_out", 3, mid_channels, latent_channels, 3, 1, 1, causal, "zeros");

        Self {
            conv_in,
            down_convs,
            down_resblocks,
            mid,
            conv_norm_out,
            conv_out,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let x = space_to_depth(x, 2);

        let mut x = self.conv_in.forward_t(&x, false);

        for (i, conv) in self.down_convs.iter().enumerate() {
            x = conv.forward_t(&x, false);
            for resblock in &self.down_resblocks[i] {
                x = resblock.forward_t(&x, false);
            }
        }

        x = self.mid.forward(&x);

        let h = self.conv_norm_out.forward_t(&x, false).silu();
        self.conv_out.forward_t(&h, false)
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

        let last_mult = *channel_multipliers.last().unwrap();
        let mid_channels = base_channels * last_mult;

        let conv_in = make_conv_nd(vs / "conv_in", 3, latent_channels, mid_channels, 3, 1, 1, causal, "zeros");

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
        let conv_out = make_conv_nd(vs / "conv_out", 3, base_channels, in_channels * 4, 3, 1, 1, causal, "zeros");

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
        in_channels: i64,
        base_channels: i64,
        channel_multipliers: &[i64],
        num_res_blocks: i64,
        latent_channels: i64,
        norm_num_groups: i64,
        causal: bool,
        norm_type: NormLayerType,
        spatial_downsample_factor: i64,
    ) -> Self {
        let vs = vs.borrow();

        let encoder = VideoEncoder::new(
            vs / "encoder",
            in_channels,
            base_channels,
            channel_multipliers,
            num_res_blocks,
            latent_channels,
            norm_num_groups,
            causal,
            norm_type,
        );

        let decoder = VideoDecoder::new(
            vs / "decoder",
            latent_channels,
            base_channels,
            channel_multipliers,
            num_res_blocks,
            in_channels,
            norm_num_groups,
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
