use tch::nn::{Linear, Module};
use tch::Tensor;

pub struct PixArtAlphaTextProjection {
    linear_1: Linear,
    linear_2: Linear,
}

impl PixArtAlphaTextProjection {
    pub fn new(vs: &tch::nn::Path, text_dim: i64, hidden_dim: i64) -> Self {
        Self {
            linear_1: tch::nn::linear(vs / "linear_1", text_dim, hidden_dim, Default::default()),
            linear_2: tch::nn::linear(vs / "linear_2", hidden_dim, hidden_dim, Default::default()),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.linear_2.forward(&self.linear_1.forward(x).silu())
    }
}

impl std::fmt::Debug for PixArtAlphaTextProjection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PixArtAlphaTextProjection").finish_non_exhaustive()
    }
}
