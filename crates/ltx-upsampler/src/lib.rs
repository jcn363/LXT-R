pub mod blur_downsample;
pub mod configurator;
pub mod pixel_shuffle;
pub mod rational_resampler;

use tch::nn::ModuleT;
use tch::Tensor;

use ltx_conv::make_conv_nd;
use ltx_resblock::make_resblock;
use ltx_types::NormLayerType;

pub use blur_downsample::BlurDownsample;
pub use configurator::{from_config, LatentUpsamplerConfig};
pub use pixel_shuffle::PixelShuffleND;
pub use rational_resampler::SpatialRationalResampler;

/// Latent upsampler that increases spatial resolution of video latents.
///
/// Architecture:
/// 1. Conv projection: `in_channels` → `hidden_channels`
/// 2. ResNet refinement blocks at low resolution
/// 3. Channel expansion: `hidden_channels` → `hidden_channels * r^2`
/// 4. PixelShuffleND: spatial upscale by `upscale_factor`
/// 5. Post-shuffle ResNet refinement blocks
/// 6. Conv projection: `hidden_channels` → `out_channels`
///
/// All convolutions, norm layers, and ResNet blocks are sourced from
/// shared primitives (`ltx_conv`, `ltx_norm`, `ltx_resblock`) —
/// no reimplementation.
pub struct LatentUpsampler {
    #[allow(dead_code)]
    vs: tch::nn::VarStore,
    conv_in: Box<dyn ModuleT>,
    pre_shuffle_blocks: Vec<Box<dyn ModuleT>>,
    channel_expand: Box<dyn ModuleT>,
    pixel_shuffle: PixelShuffleND,
    post_shuffle_blocks: Vec<Box<dyn ModuleT>>,
    conv_out: Box<dyn ModuleT>,
}

impl std::fmt::Debug for LatentUpsampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LatentUpsampler").finish()
    }
}

impl LatentUpsampler {
    /// Create a new LatentUpsampler on the given device.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: tch::Device,
        in_channels: i64,
        out_channels: i64,
        num_res_blocks: i64,
        hidden_channels: i64,
        upscale_factor: i64,
        norm_groups: i64,
        refine_blocks: i64,
    ) -> Self {
        let vs = tch::nn::VarStore::new(device);
        Self::new_at_path(
            vs,
            in_channels,
            out_channels,
            num_res_blocks,
            hidden_channels,
            upscale_factor,
            norm_groups,
            refine_blocks,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_at_path(
        vs: tch::nn::VarStore,
        in_channels: i64,
        out_channels: i64,
        num_res_blocks: i64,
        hidden_channels: i64,
        upscale_factor: i64,
        norm_groups: i64,
        refine_blocks: i64,
    ) -> Self {
        let root = vs.root();

        // Conv in: project to hidden channels
        let conv_in = make_conv_nd(
            root.clone() / "conv_in",
            3,
            in_channels,
            hidden_channels,
            3,
            1,
            1,
            false,
            "zeros",
        );

        // Pre-shuffle ResNet blocks
        let mut pre_shuffle_blocks: Vec<Box<dyn ModuleT>> = Vec::new();
        for i in 0..num_res_blocks {
            let block = make_resblock(
                3,
                hidden_channels,
                hidden_channels,
                NormLayerType::Group,
                norm_groups,
                false,
                root.clone() / format!("pre_shuffle_{i}"),
            );
            pre_shuffle_blocks.push(block);
        }

        // Channel expansion before pixel shuffle: hidden → hidden * r^2
        let channel_expand = make_conv_nd(
            root.clone() / "channel_expand",
            3,
            hidden_channels,
            hidden_channels * upscale_factor * upscale_factor,
            1,
            1,
            0,
            false,
            "zeros",
        );

        // Pixel shuffle for spatial upsampling
        let pixel_shuffle = PixelShuffleND::new(upscale_factor, 3);

        // Post-shuffle refinement blocks (input channels = hidden after pixel shuffle)
        let mut post_shuffle_blocks: Vec<Box<dyn ModuleT>> = Vec::new();
        for i in 0..refine_blocks {
            let block = make_resblock(
                3,
                hidden_channels,
                hidden_channels,
                NormLayerType::Group,
                norm_groups,
                false,
                root.clone() / format!("post_shuffle_{i}"),
            );
            post_shuffle_blocks.push(block);
        }

        // Conv out: project to output channels
        let conv_out = make_conv_nd(
            root / "conv_out",
            3,
            hidden_channels,
            out_channels,
            3,
            1,
            1,
            false,
            "zeros",
        );

        Self {
            vs,
            conv_in,
            pre_shuffle_blocks,
            channel_expand,
            pixel_shuffle,
            post_shuffle_blocks,
            conv_out,
        }
    }

    /// Forward pass: upsample a 5D video latent `(B, C, T, H, W)`.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // 1. Project input to hidden channels
        let h = self.conv_in.forward_t(x, false);

        // 2. Pre-shuffle refinement
        let mut h = h;
        for block in &self.pre_shuffle_blocks {
            h = block.forward_t(&h, false);
        }

        // 3. Channel expansion + pixel shuffle
        let h = self.channel_expand.forward_t(&h, false);
        let h = self.pixel_shuffle.forward(&h);

        // 4. Post-shuffle refinement
        let mut h = h;
        for block in &self.post_shuffle_blocks {
            h = block.forward_t(&h, false);
        }

        // 5. Project to output channels
        self.conv_out.forward_t(&h, false)
    }
}

impl ModuleT for LatentUpsampler {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        self.forward(xs)
    }
}
