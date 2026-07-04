use tch::nn::ModuleT;

use ltx_types::NormLayerType;

use crate::resblock_2d::ResnetBlock2D;
use crate::resblock_3d::ResnetBlock3D;

/// THE ONLY factory for creating residual blocks in the codebase.
///
/// Returns a boxed `ModuleT` ready for use in model construction.
///
/// # Arguments
/// * `dims` — 2 for ResnetBlock2D, 3 for ResnetBlock3D.
/// * `in_channels` — input channel count.
/// * `out_channels` — output channel count.
/// * `norm_type` — `Group` or `Pixel` normalization.
/// * `norm_groups` — number of groups for GroupNorm (ignored for PixelNorm).
/// * `causal` — whether convolutions use causal temporal padding.
pub fn make_resblock(
    dims: i64,
    in_channels: i64,
    out_channels: i64,
    norm_type: NormLayerType,
    norm_groups: i64,
    causal: bool,
    vs: tch::nn::Path,
) -> Box<dyn ModuleT> {
    match dims {
        3 => Box::new(ResnetBlock3D::new(vs, in_channels, out_channels, norm_type, norm_groups, causal)),
        2 => Box::new(ResnetBlock2D::new(vs, in_channels, out_channels, norm_type, norm_groups, causal)),
        _ => panic!("make_resblock: unsupported dims={dims} (only 2 or 3 supported)"),
    }
}
