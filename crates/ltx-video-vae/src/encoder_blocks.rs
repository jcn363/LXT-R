use std::borrow::Borrow;

use tch::nn::{ModuleT, Path};
use tch::Tensor;

use ltx_conv::make_conv_nd;
use ltx_norm::build_norm_layer;
use ltx_resblock::make_resblock;
use ltx_types::NormLayerType;

// ---------------------------------------------------------------------------
// Block 0/3/6/8/9: Sequential ResBlocks (no spatial change)
// ---------------------------------------------------------------------------

pub fn make_resblock_stage<'a>(
    vs: impl Borrow<Path<'a>>,
    channels: i64,
    num_blocks: i64,
    norm_type: NormLayerType,
    norm_groups: i64,
    causal: bool,
) -> Vec<Box<dyn ModuleT>> {
    let vs = vs.borrow();
    (0..num_blocks)
        .map(|j| {
            make_resblock(
                3,
                channels,
                channels,
                norm_type,
                norm_groups,
                causal,
                vs / "res_blocks" / j,
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Block 1/4/7: Stride-2 Downsample Conv (no channel change)
// ---------------------------------------------------------------------------

pub fn make_downsample_conv(
    vs: &Path,
    channels: i64,
    causal: bool,
) -> Box<dyn ModuleT> {
    make_conv_nd(vs / "conv", 3, channels, channels, 3, 2, 1, causal, "zeros")
}

// ---------------------------------------------------------------------------
// Block 2/5: Channel-change downsample with shortcut + GroupNorm
// ---------------------------------------------------------------------------

pub struct ChannelChangeDownsample {
    norm: Box<dyn ModuleT>,
    conv1: Box<dyn ModuleT>,
    conv2: Box<dyn ModuleT>,
    shortcut: Box<dyn ModuleT>,
}

impl std::fmt::Debug for ChannelChangeDownsample {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChannelChangeDownsample").finish()
    }
}

impl ChannelChangeDownsample {
    pub fn new<'a>(
        vs: impl Borrow<Path<'a>>,
        in_ch: i64,
        out_ch: i64,
        norm_type: NormLayerType,
        norm_groups: i64,
        causal: bool,
    ) -> Self {
        let vs = vs.borrow();
        let norm = build_norm_layer(norm_type, in_ch, norm_groups);
        // conv1: in_ch → out_ch, stride 2 (downsample + channel change)
        let conv1 = make_conv_nd(vs / "conv1", 3, in_ch, out_ch, 3, 2, 1, causal, "zeros");
        // conv2: out_ch → out_ch, stride 1
        let conv2 = make_conv_nd(vs / "conv2", 3, out_ch, out_ch, 3, 1, 1, causal, "zeros");
        // 1×1×1 shortcut: in_ch → out_ch
        let shortcut = make_conv_nd(vs / "conv_shortcut", 3, in_ch, out_ch, 1, 1, 0, causal, "zeros");
        Self { norm, conv1, conv2, shortcut }
    }
}

impl ModuleT for ChannelChangeDownsample {
    fn forward_t(&self, x: &Tensor, _train: bool) -> Tensor {
        let h = self.norm.forward_t(x, false);
        let h = h.silu();
        let residual = self.shortcut.forward_t(&h, false);
        let h = self.conv1.forward_t(&h, false);
        let h = self.conv2.forward_t(&h, false);
        h + residual
    }
}

// ---------------------------------------------------------------------------
// Enum dispatch for all 10 blocks
// ---------------------------------------------------------------------------

pub enum EncoderStage {
    ResBlocks(Vec<Box<dyn ModuleT>>),
    DownsampleConv(Box<dyn ModuleT>),
    ChannelChange(ChannelChangeDownsample),
}

impl EncoderStage {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        match self {
            Self::ResBlocks(blocks) => {
                let mut h = x.shallow_clone();
                for block in blocks {
                    h = block.forward_t(&h, false);
                }
                h
            }
            Self::DownsampleConv(conv) => conv.forward_t(x, false),
            Self::ChannelChange(block) => block.forward_t(x, false),
        }
    }
}
