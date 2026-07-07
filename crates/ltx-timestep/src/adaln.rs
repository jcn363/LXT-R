use tch::nn::{Linear, Module};
use tch::Tensor;

use crate::combined::CombinedTimestepSizeEmbeddings;

/// Adaptive Layer Norm single-head. THE ONLY implementation.
pub struct AdaLayerNormSingle {
    emb: CombinedTimestepSizeEmbeddings,
    linear: Linear,
}

impl AdaLayerNormSingle {
    pub fn new(vs: &tch::nn::Path, dim: i64) -> Self {
        Self {
            emb: CombinedTimestepSizeEmbeddings::new(vs, dim),
            linear: tch::nn::linear(vs / "linear", dim, dim * 6, Default::default()),
        }
    }

    /// Create with separate sinusoidal dim (matches Python model architecture).
    pub fn new_with_input_dim(vs: &tch::nn::Path, hidden_dim: i64, sinusoidal_dim: i64) -> Self {
        Self {
            emb: CombinedTimestepSizeEmbeddings::new_with_input_dim(vs, hidden_dim, sinusoidal_dim),
            linear: tch::nn::linear(
                vs / "linear",
                hidden_dim,
                hidden_dim * 6,
                Default::default(),
            ),
        }
    }

    pub fn forward(&self, timestep: &Tensor, hidden_dtype: tch::Kind) -> (Tensor, Tensor) {
        let embedded = self.emb.forward(timestep, hidden_dtype);
        let output = self.linear.forward(&embedded.silu());
        (output, embedded)
    }
}
