use std::borrow::Borrow;
use tch::nn::{Module, Path};
use tch::Tensor;

/// Selects which spatial axis receives causal padding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CausalityAxis {
    Time,
    Width,
    None,
}

/// THE ONLY CausalConv2d implementation in the codebase.
///
/// Wraps a `tch::nn::Conv2D` with zero internal padding.
/// The caller controls whether causal padding is applied and on which axis.
pub struct CausalConv2d {
    conv: tch::nn::Conv2D,
    causal_axis: CausalityAxis,
    kernel_size: i64,
    stride: i64,
}

impl std::fmt::Debug for CausalConv2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CausalConv2d")
            .field("causal_axis", &self.causal_axis)
            .field("kernel_size", &self.kernel_size)
            .field("stride", &self.stride)
            .finish()
    }
}

impl CausalConv2d {
    pub fn new<'a>(
        vs: impl Borrow<Path<'a>>,
        in_channels: i64,
        out_channels: i64,
        kernel_size: i64,
        stride: i64,
        causal_axis: CausalityAxis,
    ) -> Self {
        let conv = tch::nn::conv2d(
            vs,
            in_channels,
            out_channels,
            kernel_size,
            tch::nn::ConvConfig {
                stride,
                padding: 0,
                ..Default::default()
            },
        );
        Self {
            conv,
            causal_axis,
            kernel_size,
            stride,
        }
    }

    /// Forward pass with explicit causality control.
    ///
    /// When `causal = true`, pads the selected axis by repeating the first
    /// element along that dimension.  When `causal = false` or the axis is
    /// `CausalityAxis::None`, runs the raw convolution (no padding).
    pub fn forward(&self, x: &Tensor, causal: bool) -> Tensor {
        if causal {
            let pad_size = self.kernel_size - 1;
            match self.causal_axis {
                CausalityAxis::Time => {
                    let first = x
                        .narrow(2, 0, 1)
                        .expand([1, 1, pad_size, x.size()[3]], true);
                    self.conv.forward(&Tensor::cat(&[&first, x], 2))
                }
                CausalityAxis::Width => {
                    let first = x
                        .narrow(3, 0, 1)
                        .expand([1, 1, x.size()[2], pad_size], true);
                    self.conv.forward(&Tensor::cat(&[&first, x], 3))
                }
                CausalityAxis::None => self.conv.forward(x),
            }
        } else {
            self.conv.forward(x)
        }
    }
}

impl tch::nn::ModuleT for CausalConv2d {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        self.forward(xs, true)
    }
}
