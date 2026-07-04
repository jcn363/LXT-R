use tch::Tensor;

use ltx_conv::make_conv_nd;

/// Anti-aliased downsampling using Gaussian blur + strided convolution.
///
/// Implements the blur-downsample from "Making GANs More Robust" (Zhang 2019):
/// a fixed Gaussian kernel low-pass-filters the input before strided conv
/// reduces the spatial resolution. This avoids aliasing artifacts that
/// naive strided convolutions introduce.
///
/// For 3D (video latent) inputs the blur is applied per-frame (2D kernel
/// across H/W), and the strided conv operates across all dims.
pub struct BlurDownsample {
    conv: Box<dyn tch::nn::ModuleT>,
    blur_kernel: Tensor,
    stride: i64,
}

impl std::fmt::Debug for BlurDownsample {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlurDownsample")
            .field("stride", &self.stride)
            .finish()
    }
}

impl BlurDownsample {
    /// Create a new blur-downsample module.
    ///
    /// # Arguments
    /// * `vs` — parameter scope.
    /// * `in_channels` — input channels.
    /// * `out_channels` — output channels.
    /// * `stride` — spatial downsampling factor (must be 2 or 4).
    /// * `kernel_size` — Gaussian kernel size (default 3 for stride-2, 5 for stride-4).
    pub fn new(
        vs: tch::nn::Path,
        in_channels: i64,
        out_channels: i64,
        stride: i64,
        kernel_size: Option<i64>,
    ) -> Self {
        assert!(
            stride == 2 || stride == 4,
            "stride must be 2 or 4, got {stride}"
        );

        let k = kernel_size.unwrap_or(if stride == 2 { 3 } else { 5 });
        let blur_kernel = Self::make_gaussian_kernel(k);

        // Learnable strided conv via shared primitive
        let conv = make_conv_nd(
            vs / "conv",
            2,
            in_channels,
            out_channels,
            k,
            stride,
            (k - 1) / 2,
            false,
            "zeros",
        );

        Self {
            conv,
            blur_kernel,
            stride,
        }
    }

    /// Forward pass — handles both 4D (B,C,H,W) and 5D (B,C,T,H,W) inputs.
    ///
    /// For 5D inputs, the blur kernel is applied per-frame by reshaping
    /// to 4D, blurring, then striding via the conv.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let ndims = x.dim();
        assert!(
            ndims == 4 || ndims == 5,
            "BlurDownsample expects 4D or 5D input, got {ndims}D"
        );

        if ndims == 5 {
            self.forward_5d(x)
        } else {
            self.forward_4d(x)
        }
    }

    fn forward_4d(&self, x: &Tensor) -> Tensor {
        let (_b, c, _h, _w) = x.size4().expect("forward_4d: tensor must be 4D");
        let k = self.blur_kernel.size()[0];

        // Blur: depthwise conv with fixed Gaussian kernel (no learnable params)
        let gaussian = Self::make_gaussian_kernel_4d(k, c);
        let groups = c;
        let blurred =
            tch::Tensor::conv2d(x, &gaussian, None::<&Tensor>, [1, 1], [1, 1], [1, 1], groups);

        // Strided conv with learnable weights
        self.conv.forward_t(&blurred, false)
    }

    fn forward_5d(&self, x: &Tensor) -> Tensor {
        let (b, c, t, h, w) = x.size5().expect("forward_5d: tensor must be 5D");
        // Reshape to (B*T, C, H, W) for per-frame blur + conv
        let x_4d = x.reshape([b * t, c, h, w]);
        let out = self.forward_4d(&x_4d);
        let (_, c_out, h_out, w_out) = out.size4().expect("forward_5d: output must be 4D");
        out.reshape([b, t, c_out, h_out, w_out])
            .permute([0, 2, 1, 3, 4])
    }

    /// Create a 1D Gaussian kernel of the given size, then outer-product to 2D.
    fn make_gaussian_kernel(size: i64) -> Tensor {
        assert!(size % 2 == 1, "kernel size must be odd, got {size}");
        let sigma = size as f64 / 6.0;
        let half = size / 2;
        let coords =
            Tensor::arange_start(-half, half + 1, (tch::Kind::Float, tch::Device::Cpu));
        let gaussian_1d = (&coords * &coords * (-0.5 / (sigma * sigma))).exp();
        let gaussian_1d = &gaussian_1d / gaussian_1d.sum(tch::Kind::Float);
        // Outer product for 2D kernel: [3,1] * [1,3] = [3,3]
        gaussian_1d.unsqueeze(1) * gaussian_1d.unsqueeze(0)
    }

    /// Create a 4D Gaussian kernel for depthwise conv: (C, 1, K, K).
    fn make_gaussian_kernel_4d(size: i64, channels: i64) -> Tensor {
        let kernel_2d = Self::make_gaussian_kernel(size);
        kernel_2d
            .reshape([1, 1, size, size])
            .repeat([channels, 1, 1, 1])
    }
}

impl tch::nn::ModuleT for BlurDownsample {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        self.forward(xs)
    }
}
