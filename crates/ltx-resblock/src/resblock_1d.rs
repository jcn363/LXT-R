use std::borrow::Borrow;
use tch::nn::{Module, ModuleT, Path};
use tch::Tensor;

use ltx_norm::build_norm_layer;
use ltx_types::NormLayerType;

/// THE ONLY ResBlock1 in the codebase.
///
/// 1D residual block for the vocoder (ConvTranspose1d-based decoder).
/// Architecture: `x + activation(norm(conv(x)))`
pub struct ResBlock1 {
    conv: tch::nn::Conv1D,
    norm: Box<dyn ModuleT>,
    use_silu: bool,
    negative_slope: f64,
}

impl std::fmt::Debug for ResBlock1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResBlock1").finish()
    }
}

impl ResBlock1 {
    /// Create a new ResBlock1.
    ///
    /// # Arguments
    /// * `vs` — parameter namespace.
    /// * `channels` — number of channels (in == out).
    /// * `kernel_size` — convolution kernel size.
    /// * `norm_type` — `Group` or `Pixel` normalization.
    /// * `norm_groups` — number of groups for GroupNorm (ignored for PixelNorm).
    /// * `negative_slope` — slope for LeakyReLU; use 0.0 for SiLU instead.
    pub fn new<'a>(
        vs: impl Borrow<Path<'a>>,
        channels: i64,
        kernel_size: i64,
        norm_type: NormLayerType,
        norm_groups: i64,
        negative_slope: f64,
    ) -> Self {
        let vs = vs.borrow();
        let padding = (kernel_size - 1) / 2;

        let conv = tch::nn::conv1d(
            vs / "conv",
            channels,
            channels,
            kernel_size,
            tch::nn::ConvConfig {
                padding,
                ..Default::default()
            },
        );

        let norm = build_norm_layer(norm_type, channels, norm_groups);
        let use_silu = negative_slope <= 0.0;

        Self {
            conv,
            norm,
            use_silu,
            negative_slope,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.norm.forward_t(x, false);
        let h = if self.use_silu {
            h.silu()
        } else {
            let positive = h.clamp_min(0.0);
            let negative = h.clamp_max(0.0) * self.negative_slope;
            positive + negative
        };
        let h = self.conv.forward(&h);
        x + h
    }
}

impl ModuleT for ResBlock1 {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        self.forward(xs)
    }
}
