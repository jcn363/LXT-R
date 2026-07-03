use tch::Tensor;
use ltx_types::NORM_EPS;

pub struct RMSNorm {
    weight: Tensor,
    eps: f64,
}

impl std::fmt::Debug for RMSNorm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RMSNorm")
            .field("weight", &self.weight)
            .field("eps", &self.eps)
            .finish()
    }
}

impl RMSNorm {
    pub fn new(dim: i64, eps: f64, device: tch::Device) -> Self {
        Self {
            weight: Tensor::ones([dim], (tch::Kind::Float, device)),
            eps,
        }
    }

    /// Create with default epsilon from SSOT constants.
    pub fn default_eps(dim: i64, device: tch::Device) -> Self {
        Self::new(dim, NORM_EPS, device)
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let x_f32 = x.to_kind(tch::Kind::Float);
        let rms = (&x_f32 * &x_f32).mean_dim([-1i64].as_slice(), true, tch::Kind::Float);
        (x_f32 / (rms + self.eps).sqrt()).to_kind(x.kind()) * &self.weight
    }
}

impl tch::nn::Module for RMSNorm {
    fn forward(&self, xs: &Tensor) -> Tensor {
        self.forward(xs)
    }
}
