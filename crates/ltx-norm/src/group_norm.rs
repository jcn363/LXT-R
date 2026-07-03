use tch::Tensor;
use ltx_types::NORM_EPS;

/// Thin wrapper around `Tensor::group_norm` that implements `ModuleT`.
///
/// Unlike `tch::nn::group_norm`, this does not require a `Path` for parameter
/// initialization — the affine weights are stored directly on the struct.
pub struct GroupNorm {
    num_groups: i64,
    num_channels: i64,
    eps: f64,
    cudnn_enabled: bool,
    weight: Option<Tensor>,
    bias: Option<Tensor>,
}

impl std::fmt::Debug for GroupNorm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GroupNorm")
            .field("num_groups", &self.num_groups)
            .field("num_channels", &self.num_channels)
            .field("eps", &self.eps)
            .finish()
    }
}

impl GroupNorm {
    pub fn new(num_groups: i64, num_channels: i64, eps: f64, affine: bool) -> Self {
        let device = tch::Device::Cpu;
        let (weight, bias) = if affine {
            (
                Some(Tensor::ones(&[num_channels], (tch::Kind::Float, device))),
                Some(Tensor::zeros(&[num_channels], (tch::Kind::Float, device))),
            )
        } else {
            (None, None)
        };
        Self {
            num_groups,
            num_channels,
            eps,
            cudnn_enabled: true,
            weight,
            bias,
        }
    }

    /// Convenience constructor using `NORM_EPS` from SSOT.
    pub fn with_defaults(num_groups: i64, num_channels: i64) -> Self {
        Self::new(num_groups, num_channels, NORM_EPS, true)
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        Tensor::group_norm(
            x,
            self.num_groups,
            self.weight.as_ref(),
            self.bias.as_ref(),
            self.eps,
            self.cudnn_enabled,
        )
    }
}

impl tch::nn::Module for GroupNorm {
    fn forward(&self, xs: &Tensor) -> Tensor {
        self.forward(xs)
    }
}
