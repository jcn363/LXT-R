use std::borrow::Borrow;
use tch::nn::Path;
use tch::Tensor;

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
        _ => panic!("make_conv_nd: unsupported dims={dims} (only 2 or 3 supported)"),
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
    CausalConv2d::new(
        vs,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        causal_axis,
    )
}

/// Create a 2D transposed convolution (ConvTranspose2d) for upsampling.
pub fn make_conv_transpose2d<'a>(
    vs: impl Borrow<Path<'a>>,
    in_channels: i64,
    out_channels: i64,
    kernel_size: i64,
    stride: i64,
    padding: i64,
) -> tch::nn::ConvTranspose2D {
    tch::nn::conv_transpose2d(
        vs,
        in_channels,
        out_channels,
        kernel_size,
        tch::nn::ConvTransposeConfig {
            stride,
            padding,
            ..Default::default()
        },
    )
}

/// Raw asymmetric transposed conv2d that only upsamples along one axis.
///
/// `kernel_time` / `stride_time` — kernel and stride along the time (height) axis.
/// `kernel_freq` / `stride_freq` — kernel and stride along the freq (width) axis.
/// Use kernel_freq=1, stride_freq=1 to keep the freq dimension unchanged.
pub struct AsymConvTranspose2d {
    weight: Tensor,
    bias: Tensor,
    stride: [i64; 2],
    padding: [i64; 2],
    output_padding: [i64; 2],
}

impl std::fmt::Debug for AsymConvTranspose2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsymConvTranspose2d")
            .field("stride", &self.stride)
            .field("padding", &self.padding)
            .finish()
    }
}

impl AsymConvTranspose2d {
    pub fn new<'a>(
        vs: impl Borrow<Path<'a>>,
        in_channels: i64,
        out_channels: i64,
        kernel_time: i64,
        kernel_freq: i64,
        stride_time: i64,
        stride_freq: i64,
    ) -> Self {
        let vs = vs.borrow();
        let fan_in = in_channels * kernel_time * kernel_freq;
        let std = (2.0 / fan_in as f64).sqrt();
        // Weight shape for transposed conv: [in_channels, out_channels, kH, kW]
        let weight = vs.var(
            "weight",
            &[in_channels, out_channels, kernel_time, kernel_freq],
            tch::nn::init::Init::Randn {
                mean: 0.0,
                stdev: std,
            },
        );
        let bias = vs.var("bias", &[out_channels], tch::nn::init::Init::Const(0.0));
        Self {
            weight,
            bias,
            stride: [stride_time, stride_freq],
            padding: [(kernel_time - 1) / 2, (kernel_freq - 1) / 2],
            output_padding: [0, 0],
        }
    }
}

impl tch::nn::ModuleT for AsymConvTranspose2d {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        xs.conv_transpose2d(
            &self.weight,
            Some(&self.bias),
            self.stride,
            self.padding,
            self.output_padding,
            1,      // groups
            [1, 1], // dilation
        )
    }
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
