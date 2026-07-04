use tch::nn::ModuleT;
use tch::Tensor;

use ltx_conv::make_conv_nd;
use ltx_norm::build_norm_layer;
use ltx_types::NormLayerType;

/// Rational-factor spatial resampler for latent tensors.
///
/// Resamples 3D video latents `(B, C, T, H, W)` by an arbitrary rational
/// factor `(num / den)` in the spatial dimensions. The implementation:
///
/// 1. If `num > den`: upsample by `num` via nearest-interpolation, then
///    downsample by `den` via average pooling.
/// 2. If `num < den`: downsample by `den` via average pooling, then
///    upsample by `num` via nearest-interpolation.
/// 3. A lightweight conv-refine stage cleans up interpolation artifacts.
///
/// The temporal axis is always preserved unchanged.
pub struct SpatialRationalResampler {
    num: i64,
    den: i64,
    refine_conv: Box<dyn ModuleT>,
    refine_norm: Box<dyn ModuleT>,
}

impl std::fmt::Debug for SpatialRationalResampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpatialRationalResampler")
            .field("num", &self.num)
            .field("den", &self.den)
            .finish()
    }
}

impl SpatialRationalResampler {
    pub fn new(vs: tch::nn::Path, channels: i64, num: i64, den: i64, norm_groups: i64) -> Self {
        assert!(num > 0 && den > 0, "num and den must be positive");
        assert!(num != den, "num == den means no resampling needed");

        let norm = build_norm_layer(NormLayerType::Group, channels, norm_groups);
        let conv = make_conv_nd(vs / "conv", 3, channels, channels, 3, 1, 1, false, "zeros");

        Self {
            num,
            den,
            refine_conv: conv,
            refine_norm: norm,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let (_b, _c, _t, h, w) = x.size5().expect("forward: input tensor must be 5D");

        let upsampled = if self.num > self.den {
            // Upsample by num, then downsample by den
            let up = Self::nearest_upsample(x, self.num);
            let (_, _, _, h_up, w_up) = up.size5().expect("nearest_upsample: output must be 5D");
            Self::downsample_by_factor(&up, self.den, h_up, w_up)
        } else {
            // Downsample by den, then upsample by num
            let down = Self::downsample_by_factor(x, self.den, h, w);
            Self::nearest_upsample(&down, self.num)
        };

        // Refine: norm → SiLU → conv
        let refined = self.refine_norm.forward_t(&upsampled, false).silu();
        let refined = self.refine_conv.forward_t(&refined, false);

        // Ensure output matches expected spatial size
        let (_, _, _, h_out, w_out) = refined.size5().expect("refine_conv: output must be 5D");
        let h_expected = h * self.num / self.den;
        let w_expected = w * self.num / self.den;

        if h_out != h_expected || w_out != w_expected {
            refined.narrow(3, 0, h_expected).narrow(4, 0, w_expected)
        } else {
            refined
        }
    }

    /// Nearest-neighbor upsample by factor `r` on spatial dims only.
    fn nearest_upsample(x: &Tensor, r: i64) -> Tensor {
        let (b, c, t, h, w) = x.size5().expect("nearest_upsample: input must be 5D");
        x.reshape([b, c, t, h, 1, w, 1])
            .expand([b, c, t, h, r, w, r], true)
            .reshape([b, c, t, h * r, w * r])
    }

    /// Downsample by factor `r` using average pooling on spatial dims.
    fn downsample_by_factor(x: &Tensor, r: i64, h: i64, w: i64) -> Tensor {
        let (b, c, t, _, _) = x.size5().expect("downsample_by_factor: input must be 5D");
        let h_out = h / r;
        let w_out = w / r;
        // Reshape to 4D for adaptive_avg_pool2d, process per-frame
        let x_4d = x.reshape([b * t, c, h, w]);
        let pooled = x_4d.adaptive_avg_pool2d([h_out, w_out]);
        let (_, _, h_out, w_out) = pooled
            .size4()
            .expect("adaptive_avg_pool2d: output must be 4D");
        pooled
            .reshape([b, t, c, h_out, w_out])
            .permute([0, 2, 1, 3, 4])
    }
}

impl ModuleT for SpatialRationalResampler {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        self.forward(xs)
    }
}
