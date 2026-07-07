use tch::nn::{Linear, Module};
use tch::Tensor;

/// Timestep MLP embedding. THE ONLY implementation.
pub struct TimestepEmbedding {
    linear_1: Linear,
    linear_2: Linear,
}

impl TimestepEmbedding {
    pub fn new(vs: &tch::nn::Path, time_embed_dim: i64) -> Self {
        let linear_1 = tch::nn::linear(
            vs / "linear_1",
            time_embed_dim,
            time_embed_dim,
            Default::default(),
        );
        let linear_2 = tch::nn::linear(
            vs / "linear_2",
            time_embed_dim,
            time_embed_dim,
            Default::default(),
        );
        Self { linear_1, linear_2 }
    }

    /// Create with separate input dimension (for sinusoidal embedding input).
    pub fn new_with_input_dim(vs: &tch::nn::Path, input_dim: i64, output_dim: i64) -> Self {
        let linear_1 = tch::nn::linear(vs / "linear_1", input_dim, output_dim, Default::default());
        let linear_2 = tch::nn::linear(vs / "linear_2", output_dim, output_dim, Default::default());
        Self { linear_1, linear_2 }
    }

    pub fn forward(&self, sample: &Tensor) -> Tensor {
        let h = self.linear_1.forward(sample).silu();
        self.linear_2.forward(&h)
    }
}
