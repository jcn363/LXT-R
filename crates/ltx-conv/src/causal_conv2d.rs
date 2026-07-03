use std::borrow::Borrow;
use tch::nn::{ModuleT, Path};
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
/// Wraps a convolution with zero internal padding.
/// The caller controls whether causal padding is applied and on which axis.
pub struct CausalConv2d {
    conv: Box<dyn ModuleT>,
    causal_axis: CausalityAxis,
    kernel_time: i64,
    kernel_freq: i64,
}

impl std::fmt::Debug for CausalConv2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CausalConv2d")
            .field("causal_axis", &self.causal_axis)
            .field("kernel", &[self.kernel_time, self.kernel_freq])
            .finish()
    }
}

/// Raw conv2d wrapper that allows asymmetric kernel/stride.
#[derive(Debug)]
struct AsymConv2d {
    weight: Tensor,
    bias: Tensor,
    stride: [i64; 2],
}

impl AsymConv2d {
    fn new(vs: &Path, in_c: i64, out_c: i64, k_h: i64, k_w: i64, s_h: i64, s_w: i64) -> Self {
        let fan_in = in_c * k_h * k_w;
        let std = (2.0 / fan_in as f64).sqrt();
        let weight = vs.var("weight", &[out_c, in_c, k_h, k_w], tch::nn::init::Init::Randn { mean: 0.0, stdev: std });
        let bias = vs.var("bias", &[out_c], tch::nn::init::Init::Const(0.0));
        Self { weight, bias, stride: [s_h, s_w] }
    }
}

impl ModuleT for AsymConv2d {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        xs.conv2d(&self.weight, Some(&self.bias), self.stride, [0, 0], [1, 1], 1)
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
        let conv: tch::nn::Conv2D = tch::nn::conv2d(
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
            conv: Box::new(conv),
            causal_axis,
            kernel_time: kernel_size,
            kernel_freq: kernel_size,
        }
    }

    /// Create with per-axis kernel and stride sizes.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_axes<'a>(
        vs: impl Borrow<Path<'a>>,
        in_channels: i64,
        out_channels: i64,
        kernel_time: i64,
        kernel_freq: i64,
        stride_time: i64,
        stride_freq: i64,
        causal_axis: CausalityAxis,
    ) -> Self {
        let conv: Box<dyn ModuleT> = Box::new(AsymConv2d::new(
            vs.borrow(), in_channels, out_channels, kernel_time, kernel_freq, stride_time, stride_freq,
        ));
        Self { conv, causal_axis, kernel_time, kernel_freq }
    }

    /// Forward pass with explicit causality control.
    pub fn forward(&self, x: &Tensor, causal: bool) -> Tensor {
        if causal {
            match self.causal_axis {
                CausalityAxis::Time => {
                    let b = x.size()[0];
                    let c = x.size()[1];
                    let f = x.size()[3];
                    let causal_pad = self.kernel_time - 1;
                    let first = x.narrow(2, 0, 1).expand([b, c, causal_pad, f], true);
                    let padded = Tensor::cat(&[&first, x], 2);
                    let freq_pad = (self.kernel_freq - 1) / 2;
                    let padded = if freq_pad > 0 {
                        padded.constant_pad_nd([freq_pad, freq_pad, 0, 0])
                    } else {
                        padded
                    };
                    self.conv.forward_t(&padded, false)
                }
                CausalityAxis::Width => {
                    let b = x.size()[0];
                    let c = x.size()[1];
                    let t = x.size()[2];
                    let causal_pad = self.kernel_freq - 1;
                    let first = x.narrow(3, 0, 1).expand([b, c, t, causal_pad], true);
                    let padded = Tensor::cat(&[&first, x], 3);
                    let time_pad = (self.kernel_time - 1) / 2;
                    let padded = if time_pad > 0 {
                        padded.constant_pad_nd([0, 0, time_pad, time_pad])
                    } else {
                        padded
                    };
                    self.conv.forward_t(&padded, false)
                }
                CausalityAxis::None => self.conv.forward_t(x, false),
            }
        } else {
            let time_pad = (self.kernel_time - 1) / 2;
            let freq_pad = (self.kernel_freq - 1) / 2;
            if time_pad > 0 || freq_pad > 0 {
                let padded = x.constant_pad_nd([freq_pad, freq_pad, time_pad, time_pad]);
                self.conv.forward_t(&padded, false)
            } else {
                self.conv.forward_t(x, false)
            }
        }
    }
}

impl ModuleT for CausalConv2d {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        self.forward(xs, true)
    }
}
