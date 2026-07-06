use tch::nn::{ModuleT, Path};
use tch::Tensor;

use ltx_norm::build_norm_layer;
use ltx_resblock::make_resblock;
use ltx_types::NormLayerType;

// ---------------------------------------------------------------------------
// Block 0/3/6/8/9: Sequential ResBlocks (no spatial change)
// ---------------------------------------------------------------------------

pub fn make_resblock_stage(
    vs: &Path,
    channels: i64,
    num_blocks: i64,
    norm_type: NormLayerType,
    norm_groups: i64,
    causal: bool,
) -> Vec<Box<dyn ModuleT>> {
    (0..num_blocks)
        .map(|j| {
            make_resblock(
                3, channels, channels, norm_type, norm_groups, causal,
                vs / "res_blocks" / j,
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// SpatialConv3d: Conv3d with per-axis stride (1, s, s) to preserve time
// ---------------------------------------------------------------------------

/// Conv3d wrapper that applies spatial-only downsampling.
///
/// `tch::nn::conv3d` only takes scalar `stride`, but the Python LTX-Video
/// encoder uses `stride=(1,2,2)` to preserve the temporal dimension while
/// halving spatial resolution. This module stores weight/bias as VarStore
/// parameters and calls `Tensor::conv3d` with per-axis stride in forward.
pub struct SpatialConv3d {
    weight: Tensor,
    bias: Tensor,
}

impl std::fmt::Debug for SpatialConv3d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpatialConv3d").field("weight", &self.weight.size()).finish()
    }
}

impl SpatialConv3d {
    pub fn new(vs: &Path, in_ch: i64, out_ch: i64, kernel: i64) -> Self {
        let weight = vs.var("weight", &[out_ch, in_ch, kernel, kernel, kernel], tch::nn::init::Init::Const(0.0));
        let bias = vs.var("bias", &[out_ch], tch::nn::init::Init::Const(0.0));
        Self { weight, bias }
    }

    pub fn forward_spatial2(&self, x: &Tensor) -> Tensor {
        let k = self.weight.size();
        let pad_t = (k[2] - 1) / 2;
        let pad_h = (k[3] - 1) / 2;
        let pad_w = (k[4] - 1) / 2;
        x.conv3d(&self.weight, Some(&self.bias), [1, 2, 2], [pad_t, pad_h, pad_w], [1, 1, 1], 1)
    }

    pub fn forward_spatial1(&self, x: &Tensor) -> Tensor {
        let k = self.weight.size();
        let pad_t = (k[2] - 1) / 2;
        let pad_h = (k[3] - 1) / 2;
        let pad_w = (k[4] - 1) / 2;
        x.conv3d(&self.weight, Some(&self.bias), [1, 1, 1], [pad_t, pad_h, pad_w], [1, 1, 1], 1)
    }
}

// ---------------------------------------------------------------------------
// Block 1/4/7: Stride-2 Downsample Conv (no channel change)
// ---------------------------------------------------------------------------

pub type DownsampleConv = SpatialConv3d;

impl DownsampleConv {
    pub fn new_block(vs: &Path, channels: i64) -> Self {
        Self::new(vs, channels, channels, 3)
    }
}

// ---------------------------------------------------------------------------
// Block 2/5: Channel-change downsample with shortcut + GroupNorm
// ---------------------------------------------------------------------------

pub struct ChannelChangeDownsample {
    norm: Box<dyn ModuleT>,
    conv1: SpatialConv3d,   // stride (1,2,2)
    conv2: SpatialConv3d,   // stride (1,1,1)
    shortcut: SpatialConv3d, // stride (1,2,2) to match conv1
}

impl std::fmt::Debug for ChannelChangeDownsample {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChannelChangeDownsample").finish()
    }
}

impl ChannelChangeDownsample {
    pub fn new(vs: &Path, in_ch: i64, out_ch: i64, norm_type: NormLayerType, norm_groups: i64) -> Self {
        let norm = build_norm_layer(norm_type, in_ch, norm_groups);
        let conv1 = SpatialConv3d::new(&(vs / "conv1"), in_ch, out_ch, 3);
        let conv2 = SpatialConv3d::new(&(vs / "conv2"), out_ch, out_ch, 3);
        let shortcut = SpatialConv3d::new(&(vs / "conv_shortcut"), in_ch, out_ch, 1);
        Self { norm, conv1, conv2, shortcut }
    }
}

impl ModuleT for ChannelChangeDownsample {
    fn forward_t(&self, x: &Tensor, _train: bool) -> Tensor {
        let h = self.norm.forward_t(x, false).silu();
        let residual = self.shortcut.forward_spatial2(&h);
        let h = self.conv1.forward_spatial2(&h);
        let h = self.conv2.forward_spatial1(&h);
        h + residual
    }
}

// ---------------------------------------------------------------------------
// Enum dispatch for all 10 blocks
// ---------------------------------------------------------------------------

pub enum EncoderStage {
    ResBlocks(Vec<Box<dyn ModuleT>>),
    DownsampleConv(DownsampleConv),
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
            Self::DownsampleConv(c) => c.forward_spatial2(x),
            Self::ChannelChange(c) => c.forward_t(x, false),
        }
    }
}
