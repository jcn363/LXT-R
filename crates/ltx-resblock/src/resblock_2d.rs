use std::borrow::Borrow;
use tch::nn::{ModuleT, Path};
use tch::Tensor;

use ltx_conv::CausalConv2d;
use ltx_types::{NormLayerType, LRELU_SLOPE};

/// Apply leaky ReLU with a custom negative slope (tch 0.16 `leaky_relu()` takes no args).
fn leaky_relu(x: &Tensor, negative_slope: f64) -> Tensor {
    let positive = x.clamp_min(0.0);
    let negative = x.clamp_max(0.0) * negative_slope;
    positive + negative
}

/// THE ONLY ResnetBlock2D in the codebase.
///
/// Residual block for the audio VAE encoder/decoder.
/// Uses LeakyReLU activation and CausalConv2d convolutions.
pub struct ResnetBlock2D {
    conv1: CausalConv2d,
    conv2: CausalConv2d,
    shortcut: Option<Box<dyn ModuleT>>,
}

impl std::fmt::Debug for ResnetBlock2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResnetBlock2D").finish()
    }
}

impl ResnetBlock2D {
    pub fn new<'a>(
        vs: impl Borrow<Path<'a>>,
        in_channels: i64,
        out_channels: i64,
        _norm_type: NormLayerType,
        _norm_groups: i64,
        _causal: bool,
    ) -> Self {
        let vs = vs.borrow();

        let conv1 = CausalConv2d::new(
            vs / "conv1",
            in_channels,
            out_channels,
            3,
            1,
            ltx_conv::CausalityAxis::Time,
        );
        let conv2 = CausalConv2d::new(
            vs / "conv2",
            out_channels,
            out_channels,
            3,
            1,
            ltx_conv::CausalityAxis::Time,
        );

        let shortcut: Option<Box<dyn ModuleT>> = if in_channels != out_channels {
            Some(Box::new(tch::nn::conv2d(
                vs / "shortcut",
                in_channels,
                out_channels,
                1,
                tch::nn::ConvConfig::default(),
            )))
        } else {
            None
        };

        Self {
            conv1,
            conv2,
            shortcut,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let h = leaky_relu(x, LRELU_SLOPE);
        let h = self.conv1.forward(&h, false);
        let h = leaky_relu(&h, LRELU_SLOPE);
        let h = self.conv2.forward(&h, false);
        match &self.shortcut {
            Some(s) => s.forward_t(x, true) + &h,
            None => x + &h,
        }
    }
}

impl ModuleT for ResnetBlock2D {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        self.forward(xs)
    }
}
