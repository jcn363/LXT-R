use std::borrow::Borrow;
use tch::nn::Path;

use crate::causal_conv2d::{CausalConv2d, CausalityAxis};
use crate::causal_conv3d::CausalConv3d;
use crate::dual_conv3d::DualConv3d;

/// THE ONLY factory for creating convolution modules in the codebase.
///
/// Returns a boxed `ModuleT` ready for use in model construction.
///
/// # Arguments
/// * `dims` — 2 for Conv2d, 3 for Conv3d / CausalConv3d.
/// * `in_channels` / `out_channels` — channel dimensions.
/// * `kernel_size` — spatial/temporal kernel size (uniform across dims).
/// * `stride` — convolution stride (uniform).
/// * `padding` — built-in padding; use 0 when you handle padding externally.
/// * `causal` — whether to use causal (left-only) temporal padding.
/// * `spatial_padding` — `"replicate"` / `"reflect"` / `"zeros"` (currently
///   only `"zeros"` is wired through; extend as needed).
#[allow(clippy::too_many_arguments)]
pub fn make_conv_nd<'a>(
    vs: impl Borrow<Path<'a>>,
    dims: i64,
    in_channels: i64,
    out_channels: i64,
    kernel_size: i64,
    stride: i64,
    padding: i64,
    causal: bool,
    _spatial_padding: &str,
) -> Box<dyn tch::nn::ModuleT> {
    match dims {
        2 => Box::new(tch::nn::conv2d(
            vs,
            in_channels,
            out_channels,
            kernel_size,
            tch::nn::ConvConfig {
                stride,
                padding,
                ..Default::default()
            },
        )),
        3 if causal => Box::new(CausalConv3d::new(
            vs,
            in_channels,
            out_channels,
            kernel_size,
            stride,
        )),
        3 => Box::new(tch::nn::conv3d(
            vs,
            in_channels,
            out_channels,
            kernel_size,
            tch::nn::ConvConfig {
                stride,
                padding,
                ..Default::default()
            },
        )),
        _ => panic!("make_conv_nd: unsupported dims={dims}"),
    }
}

/// Create a `CausalConv2d` with the given causality axis.
pub fn make_causal_conv2d<'a>(
    vs: impl Borrow<Path<'a>>,
    in_channels: i64,
    out_channels: i64,
    kernel_size: i64,
    stride: i64,
    causal_axis: CausalityAxis,
) -> CausalConv2d {
    CausalConv2d::new(vs, in_channels, out_channels, kernel_size, stride, causal_axis)
}

/// Create a `DualConv3d` (factorised 2D+1D) convolution.
pub fn make_dual_conv3d<'a>(
    vs_spatial: impl Borrow<Path<'a>>,
    vs_temporal: impl Borrow<Path<'a>>,
    in_channels: i64,
    out_channels: i64,
    kernel_size: i64,
    stride: i64,
) -> DualConv3d {
    DualConv3d::new(
        vs_spatial,
        vs_temporal,
        in_channels,
        out_channels,
        kernel_size,
        stride,
    )
}
