use tch::nn::{Linear, ModuleT, Path};
use tch::Tensor;

use ltx_conv::make_conv_nd;
use ltx_norm::build_norm_layer;
use ltx_types::NormLayerType;

// ---------------------------------------------------------------------------
// DecoderResBlock — timestep-conditioned residual block with AdaLN
// ---------------------------------------------------------------------------

/// Residual block with AdaLN modulation for the decoder.
///
/// Modulation: `scale_shift_table [4, C] + t_emb [B, 4, C]` →
/// `shift1, scale1, shift2, scale2` via `chunk(4)`.
///
/// Forward:
/// ```text
/// h = norm1(x) * (1 + scale1) + shift1
/// h = silu(h); h = conv1(h)
/// h2 = norm2(h) * (1 + scale2) + shift2
/// h2 = silu(h2); h2 = conv2(h2)
/// out = x + h2
/// ```
pub struct DecoderResBlock {
    norm1: Box<dyn ModuleT>,
    conv1: Box<dyn ModuleT>,
    norm2: Box<dyn ModuleT>,
    conv2: Box<dyn ModuleT>,
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
    ) -> Self {
        let norm1 = build_norm_layer(norm_type, channels, norm_groups);
        let conv1 = make_conv_nd(vs / "conv1", 3, channels, channels, 3, 1, 1, causal, "zeros");
        let norm2 = build_norm_layer(norm_type, channels, norm_groups);
        let conv2 = make_conv_nd(vs / "conv2", 3, channels, channels, 3, 1, 1, causal, "zeros");
        Self { norm1, conv1, norm2, conv2 }
    }

    /// Forward with pre-computed modulation tensor.
    ///
    /// `modulated`: `[B, 4, C]` — scale_shift_table + t_emb, already summed.
    pub fn forward_modulated(&self, x: &Tensor, modulated: &Tensor) -> Tensor {
        let chunks = modulated.chunk(4, 1);
        let shift1 = &chunks[0];
        let scale1 = &chunks[1];
        let shift2 = &chunks[2];
        let scale2 = &chunks[3];

        let h = self.norm1.forward_t(x, false);
        let h: Tensor = h * (1.0 + scale1) + shift1;
        let h = h.silu();
        let h = self.conv1.forward_t(&h, false);

        let h2 = self.norm2.forward_t(&h, false);
        let h2: Tensor = h2 * (1.0 + scale2) + shift2;
        let h2 = h2.silu();
        let h2 = self.conv2.forward_t(&h2, false);

        x + h2
    }
}

// ---------------------------------------------------------------------------
// ConvUpsample — depth_to_space(r=2) spatial upsampling
// ---------------------------------------------------------------------------

/// Conv + GroupNorm + SiLU upsampling block.
///
/// The conv expands channels by 4×, then depth_to_space(r=2) halves channels
/// and doubles spatial resolution.
pub struct ConvUpsample {
    conv: Box<dyn ModuleT>,
    norm: Box<dyn ModuleT>,
}

impl std::fmt::Debug for ConvUpsample {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConvUpsample").finish()
    }
}

impl ConvUpsample {
    pub fn new(
        vs: &Path,
        in_channels: i64,
        out_channels: i64,
        norm_type: NormLayerType,
        norm_groups: i64,
        causal: bool,
    ) -> Self {
        let conv = make_conv_nd(vs / "conv", 3, in_channels, out_channels, 3, 1, 1, causal, "zeros");
        let norm = build_norm_layer(norm_type, out_channels, norm_groups);
        Self { conv, norm }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.conv.forward_t(x, false);
        let h = self.norm.forward_t(&h, false);
        h.silu()
    }
}

// ---------------------------------------------------------------------------
// TimestepEmbedding — 2-layer MLP: linear_2(silu(linear_1(t)))
// ---------------------------------------------------------------------------

/// Two-layer MLP with SiLU activation for timestep embedding.
///
/// Matches checkpoint naming: `timestep_embedder.timestep_embedder.linear_{1,2}`.
pub struct TimestepEmbedding {
    linear_1: Linear,
    linear_2: Linear,
}

impl TimestepEmbedding {
    pub fn new(vs: &Path, input_dim: i64, output_dim: i64) -> Self {
        let linear_1 = tch::nn::linear(vs / "timestep_embedder" / "linear_1", input_dim, output_dim, Default::default());
        let linear_2 = tch::nn::linear(vs / "timestep_embedder" / "linear_2", output_dim, output_dim, Default::default());
        Self { linear_1, linear_2 }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.linear_1.forward_t(x, false).silu();
        self.linear_2.forward_t(&h, false)
    }
}

// ---------------------------------------------------------------------------
// Block descriptors
// ---------------------------------------------------------------------------

pub struct DecoderBlockDesc {
    pub kind: DecoderBlockKind,
    pub channels: i64,
    pub num_resblocks: i64,
}

pub enum DecoderBlockKind {
    /// ResBlock stage with timestep conditioning
    ResBlockStage,
    /// Conv upsampling (no resblocks, no timestep)
    ConvUpsample { in_ch: i64, out_ch: i64 },
}
