use tch::nn::{Linear, ModuleT, Path};
use tch::Tensor;

use crate::sampling::depth_to_space_3d;
use ltx_conv::make_conv_nd;
use ltx_norm::build_norm_layer;
use ltx_types::NormLayerType;

// ---------------------------------------------------------------------------
// DecoderResBlock — timestep-conditioned residual block with AdaLN + noise
// ---------------------------------------------------------------------------

/// Residual block with AdaLN modulation and optional noise injection.
///
/// Matches Python `ResnetBlock3D` with `timestep_conditioning=True`.
/// Blocks in up_blocks 2, 4, 6 also have `inject_noise=True` which adds
/// `per_channel_scale1/2` learned parameters.
pub struct DecoderResBlock {
    norm1: Box<dyn ModuleT>,
    conv1: Box<dyn ModuleT>,
    norm2: Box<dyn ModuleT>,
    conv2: Box<dyn ModuleT>,
    // Optional noise injection scales (per_channel_scale1/2 from checkpoint)
    per_channel_scale1: Option<Tensor>, // [C, 1, 1]
    per_channel_scale2: Option<Tensor>, // [C, 1, 1]
}

impl std::fmt::Debug for DecoderResBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecoderResBlock").finish()
    }
}

impl DecoderResBlock {
    pub fn new(
        vs: &Path,
        channels: i64,
        norm_type: NormLayerType,
        norm_groups: i64,
        causal: bool,
        inject_noise: bool,
    ) -> Self {
        let norm1 = build_norm_layer(norm_type, channels, norm_groups);
        let conv1 = make_conv_nd(
            vs / "conv1",
            3,
            channels,
            channels,
            3,
            1,
            1,
            causal,
            "zeros",
        );
        let norm2 = build_norm_layer(norm_type, channels, norm_groups);
        let conv2 = make_conv_nd(
            vs / "conv2",
            3,
            channels,
            channels,
            3,
            1,
            1,
            causal,
            "zeros",
        );

        let per_channel_scale1 = if inject_noise {
            Some(vs.var(
                "per_channel_scale1",
                &[channels, 1, 1],
                tch::nn::init::Init::Const(0.0),
            ))
        } else {
            None
        };
        let per_channel_scale2 = if inject_noise {
            Some(vs.var(
                "per_channel_scale2",
                &[channels, 1, 1],
                tch::nn::init::Init::Const(0.0),
            ))
        } else {
            None
        };

        Self {
            norm1,
            conv1,
            norm2,
            conv2,
            per_channel_scale1,
            per_channel_scale2,
        }
    }

    /// Forward with pre-computed modulation tensor.
    ///
    /// `modulated`: `[B, 4, C]` — scale_shift_table + t_emb, already summed.
    pub fn forward_modulated(&self, x: &Tensor, modulated: &Tensor) -> Tensor {
        let bsz = modulated.size()[0];
        let c = modulated.size()[2];
        let target: Vec<i64> = vec![bsz, c, 1, 1, 1];

        let flat = modulated.reshape([bsz * 4 * c]);
        let shift1 = flat.narrow(0, 0, c).reshape(&target);
        let scale1 = flat.narrow(0, c, c).reshape(&target);
        let shift2 = flat.narrow(0, 2 * c, c).reshape(&target);
        let scale2 = flat.narrow(0, 3 * c, c).reshape(&target);

        let h = self.norm1.forward_t(x, false);
        let h: Tensor = h * (1.0 + scale1) + shift1;
        let h = h.silu();
        let h = self.conv1.forward_t(&h, false);

        // Optional noise injection (StyleGAN-style)
        let h = if let Some(ref pcs) = self.per_channel_scale1 {
            let h_shape = h.size();
            let noise = Tensor::randn([h_shape[3], h_shape[4]], (h.kind(), h.device()));
            // pcs: [C, 1, 1], noise: [H, W] → broadcast to [C, H, W] → [1, C, 1, H, W]
            let scaled_noise = (noise * pcs).unsqueeze(0).unsqueeze(2);
            h + scaled_noise
        } else {
            h
        };

        let h2 = self.norm2.forward_t(&h, false);
        let h2: Tensor = h2 * (1.0 + scale2) + shift2;
        let h2 = h2.silu();
        let h2 = self.conv2.forward_t(&h2, false);

        let h2 = if let Some(ref pcs) = self.per_channel_scale2 {
            let h2_shape = h2.size();
            let noise = Tensor::randn([h2_shape[3], h2_shape[4]], (h2.kind(), h2.device()));
            let scaled_noise = (noise * pcs).unsqueeze(0).unsqueeze(2);
            h2 + scaled_noise
        } else {
            h2
        };

        x + h2
    }
}

// ---------------------------------------------------------------------------
// CompressAllUpsample — 3D depth_to_space(r=2) upsampling with residual
// ---------------------------------------------------------------------------

/// `compress_all` block matching the Python `DepthToSpaceUpsample`.
///
/// Conv: `in_ch → prod(stride) * in_ch / multiplier` = `8*in_ch/2 = 4*in_ch`
/// Then `depth_to_space_3d(r=2)`: channels become `in_ch / multiplier` = `in_ch/2`,
/// spatial dims double in all 3 axes.
///
/// With `residual=True`: the input is rearranged to match output shape and added.
pub struct CompressAllUpsample {
    conv: Box<dyn ModuleT>,
    residual: bool,
}

impl std::fmt::Debug for CompressAllUpsample {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompressAllUpsample").finish()
    }
}

impl CompressAllUpsample {
    /// `in_channels`: input feature channels
    /// `multiplier`: channel reduction factor (2 for LTX-Video)
    /// `residual`: whether to add a skip connection
    pub fn new(vs: &Path, in_channels: i64, multiplier: i64, causal: bool, residual: bool) -> Self {
        // Conv output: prod(stride) * in_ch / multiplier = 8 * in_ch / multiplier
        let conv_out = 8 * in_channels / multiplier;
        let conv = make_conv_nd(
            vs / "conv",
            3,
            in_channels,
            conv_out,
            3,
            1,
            1,
            causal,
            "zeros",
        );
        Self { conv, residual }
    }

    /// Forward pass.
    ///
    /// Input:  `[B, C, T, H, W]`
    /// Output: `[B, C/2, 2T, 2H, 2W]`
    pub fn forward(&self, x: &Tensor) -> Tensor {
        if self.residual {
            // Rearrange input to match output shape:
            // [B, C, T, H, W] → [B, C/8, 2T, 2H, 2W] → repeat 8/(C/C') times
            // With multiplier=2: C' = C/2, repeat = 8/2 = 4 times
            let (b, c, t, h, w) = x.size5().expect("CompressAllUpsample input must be 5D");
            // Unpack: treat C as (C/8)*2*2*2, rearrange to [B, C/8, 2T, 2H, 2W]
            let c_inner = c / 8;
            let x_in = x
                .reshape([b, c_inner, 2, 2, 2, t, h, w])
                .permute([0, 1, 5, 2, 6, 3, 7, 4])
                .reshape([b, c_inner, t * 2, h * 2, w * 2]);
            // Repeat 4 times along channel dim: C/8 * 4 = C/2
            let x_in = x_in.repeat_interleave_self_int(4, 1, None);
            // Trim first temporal frame (stride[0]=2 causal handling)
            let x_in = x_in.slice(2, 1, t * 2, 1);

            let h = self.conv.forward_t(x, false);
            let h = depth_to_space_3d(&h, 2);
            // Trim first temporal frame
            let h = h.slice(2, 1, t * 2, 1);
            h + x_in
        } else {
            let (_, _, t, _, _) = x.size5().expect("CompressAllUpsample input must be 5D");
            let h = self.conv.forward_t(x, false);
            let h = depth_to_space_3d(&h, 2);
            h.slice(2, 1, t * 2, 1)
        }
    }
}

// ---------------------------------------------------------------------------
// TimestepEmbedding — 2-layer MLP: linear_2(silu(linear_1(t)))
// ---------------------------------------------------------------------------

/// Two-layer MLP with SiLU activation for timestep embedding.
pub struct TimestepEmbedding {
    linear_1: Linear,
    linear_2: Linear,
}

impl TimestepEmbedding {
    pub fn new(vs: &Path, input_dim: i64, output_dim: i64) -> Self {
        let linear_1 = tch::nn::linear(
            vs / "timestep_embedder" / "linear_1",
            input_dim,
            output_dim,
            Default::default(),
        );
        let linear_2 = tch::nn::linear(
            vs / "timestep_embedder" / "linear_2",
            output_dim,
            output_dim,
            Default::default(),
        );
        Self { linear_1, linear_2 }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.linear_1.forward_t(x, false).silu();
        self.linear_2.forward_t(&h, false)
    }
}
