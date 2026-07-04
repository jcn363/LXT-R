use tch::nn::{Linear, Module};
use tch::Tensor;

pub struct FeedForward {
    linear_1: Linear,
    linear_2: Linear,
}

impl FeedForward {
    pub fn new(vs: &tch::nn::Path, dim: i64) -> Self {
        let ff_dim = dim * 4;
        Self {
            linear_1: tch::nn::linear(vs / "net_0", dim, ff_dim, Default::default()),
            linear_2: tch::nn::linear(vs / "net_2", ff_dim, dim, Default::default()),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.linear_2
            .forward(&self.linear_1.forward(x).gelu("none"))
    }
}

impl std::fmt::Debug for FeedForward {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FeedForward").finish_non_exhaustive()
    }
}
