use std::borrow::Borrow;
use tch::nn::{Module, Path};
use tch::Tensor;

/// THE ONLY DualConv3d implementation in the codebase.
///
/// Factorised 3D convolution: a spatial 2D convolution (kernel `[1, k, k]`)
/// followed by a temporal 1D convolution (kernel `[k, 1, 1]`).  This gives
/// the receptive-field coverage of a full 3D kernel with far fewer parameters.
pub struct DualConv3d {
    conv_spatial: tch::nn::Conv3D,
    conv_temporal: tch::nn::Conv3D,
    time_kernel_size: i64,
}

impl std::fmt::Debug for DualConv3d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DualConv3d")
            .field("time_kernel_size", &self.time_kernel_size)
            .finish()
    }
}

impl DualConv3d {
    pub fn new<'a>(
        vs_spatial: impl Borrow<Path<'a>>,
        vs_temporal: impl Borrow<Path<'a>>,
        in_channels: i64,
        out_channels: i64,
        kernel_size: i64,
        stride: i64,
    ) -> Self {
        let spatial_config = tch::nn::ConvConfigND {
            stride: [1, stride, stride],
            padding: [0, 0, 0],
            dilation: [1, 1, 1],
            groups: 1,
            bias: true,
            ws_init: tch::nn::init::DEFAULT_KAIMING_UNIFORM,
            bs_init: tch::nn::init::Init::Const(0.0),
            padding_mode: tch::nn::PaddingMode::Zeros,
        };
        let conv_spatial = tch::nn::conv(
            vs_spatial,
            in_channels,
            out_channels,
            [1, kernel_size, kernel_size],
            spatial_config,
        );

        let temporal_config = tch::nn::ConvConfigND {
            stride: [stride, 1, 1],
            padding: [0, 0, 0],
            dilation: [1, 1, 1],
            groups: 1,
            bias: true,
            ws_init: tch::nn::init::DEFAULT_KAIMING_UNIFORM,
            bs_init: tch::nn::init::Init::Const(0.0),
            padding_mode: tch::nn::PaddingMode::Zeros,
        };
        let conv_temporal = tch::nn::conv(
            vs_temporal,
            out_channels,
            out_channels,
            [kernel_size, 1, 1],
            temporal_config,
        );

        Self {
            conv_spatial,
            conv_temporal,
            time_kernel_size: kernel_size,
        }
    }

    /// Forward pass.
    ///
    /// When `causal = true`, the temporal convolution receives left-only
    /// padding (first frame repeated).  The spatial convolution always uses
    /// symmetric padding.
    pub fn forward(&self, x: &Tensor, causal: bool) -> Tensor {
        let h = self.conv_spatial.forward(x);
        let b = h.size()[0];
        let c = h.size()[1];
        let d4 = h.size()[3];
        let d5 = h.size()[4];

        if causal {
            let first = h
                .narrow(2, 0, 1)
                .expand([b, c, self.time_kernel_size - 1, d4, d5], true);
            self.conv_temporal.forward(&Tensor::cat(&[&first, &h], 2))
        } else {
            let half = (self.time_kernel_size - 1) / 2;
            let first = h
                .narrow(2, 0, 1)
                .expand([b, c, half, d4, d5], true);
            let last = h
                .narrow(2, h.size()[2] - 1, 1)
                .expand([b, c, half, d4, d5], true);
            self.conv_temporal.forward(&Tensor::cat(&[&first, &h, &last], 2))
        }
    }
}

impl tch::nn::ModuleT for DualConv3d {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        self.forward(xs, true)
    }
}
