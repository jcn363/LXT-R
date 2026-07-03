use tch::Tensor;
use ltx_types::NORM_EPS;

pub struct PixelNorm {
    eps: f64,
}

impl std::fmt::Debug for PixelNorm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PixelNorm")
            .field("eps", &self.eps)
            .finish()
    }
}

impl PixelNorm {
    pub fn new(eps: f64) -> Self {
        Self { eps }
    }

    pub fn default() -> Self {
        Self { eps: NORM_EPS }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mean_sq = (x * x).mean_dim([1i64].as_slice(), true, tch::Kind::Float);
        x / (mean_sq + self.eps).sqrt()
    }
}

impl tch::nn::Module for PixelNorm {
    fn forward(&self, xs: &Tensor) -> Tensor {
        self.forward(xs)
    }
}
