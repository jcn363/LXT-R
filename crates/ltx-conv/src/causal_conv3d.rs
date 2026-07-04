use std::borrow::Borrow;
use tch::nn::{Module, Path};
use tch::Tensor;

/// THE ONLY CausalConv3d implementation in the codebase.
///
/// Wraps a `tch::nn::Conv3D` with zero internal padding.
/// Caller-controlled causal or symmetric temporal padding is applied before
/// the underlying convolution.
pub struct CausalConv3d {
    conv: tch::nn::Conv3D,
    time_kernel_size: i64,
}

impl std::fmt::Debug for CausalConv3d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CausalConv3d")
            .field("time_kernel_size", &self.time_kernel_size)
            .finish()
    }
}

impl CausalConv3d {
    pub fn new<'a>(
        vs: impl Borrow<Path<'a>>,
        in_channels: i64,
        out_channels: i64,
        kernel_size: i64,
        stride: i64,
    ) -> Self {
        let conv = tch::nn::conv3d(
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
            time_kernel_size: kernel_size,
        }
    }

    /// Forward pass with explicit causality control.
    ///
    /// - `causal = true`: pads the time axis on the left by repeating the
    ///   first frame, preserving temporal causality.
    /// - `causal = false`: symmetrically pads with the first and last frames.
    ///
    /// Spatial dimensions (H, W) are zero-padded by `(kernel-1)/2` on each
    /// side to preserve spatial resolution.
    pub fn forward(&self, x: &Tensor, causal: bool) -> Tensor {
        let b = x.size()[0];
        let c = x.size()[1];
        let h = x.size()[3];
        let w = x.size()[4];
        let spatial_pad = (self.time_kernel_size - 1) / 2; // e.g. 1 for kernel=3
        if causal {
            let first_frame = x.narrow(2, 0, 1);
            let pad = first_frame.expand(
                [b, c, self.time_kernel_size - 1, h, w],
                true,
            );
            let x_padded = Tensor::cat(&[&pad, x], 2);
            // Zero-pad spatial dims: [W_left, W_right, H_left, H_right, T_left, T_right]
            let x_padded = x_padded.pad(
                [spatial_pad, spatial_pad, spatial_pad, spatial_pad, 0, 0],
                "constant",
                0.0,
            );
            self.conv.forward(&x_padded)
        } else {
            let half = (self.time_kernel_size - 1) / 2;
            let first = x
                .narrow(2, 0, 1)
                .expand([b, c, half, h, w], true);
            let last = x
                .narrow(2, x.size()[2] - 1, 1)
                .expand([b, c, half, h, w], true);
            let x_padded = Tensor::cat(&[&first, x, &last], 2);
            // Zero-pad spatial dims: [W_left, W_right, H_left, H_right, T_left, T_right]
            let x_padded = x_padded.pad(
                [spatial_pad, spatial_pad, spatial_pad, spatial_pad, 0, 0],
                "constant",
                0.0,
            );
            self.conv.forward(&x_padded)
        }
    }
}

impl tch::nn::ModuleT for CausalConv3d {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        self.forward(xs, true)
    }
}
