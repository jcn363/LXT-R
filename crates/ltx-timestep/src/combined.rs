use tch::Tensor;

use crate::mlp::TimestepEmbedding;
use crate::sinusoidal::SinusoidalTimesteps;

/// Combined sinusoidal + MLP timestep + size embeddings. THE ONLY implementation.
pub struct CombinedTimestepSizeEmbeddings {
    time_proj: SinusoidalTimesteps,
    embedder: TimestepEmbedding,
}

impl CombinedTimestepSizeEmbeddings {
    pub fn new(vs: &tch::nn::Path, dim: i64) -> Self {
        Self {
            time_proj: SinusoidalTimesteps::new(dim),
            embedder: TimestepEmbedding::new(vs, dim),
        }
    }

    /// Create with separate input dim for sinusoidal embedding (matches Python model).
    pub fn new_with_input_dim(vs: &tch::nn::Path, hidden_dim: i64, sinusoidal_dim: i64) -> Self {
        Self {
            time_proj: SinusoidalTimesteps::new(sinusoidal_dim),
            embedder: TimestepEmbedding::new_with_input_dim(vs, sinusoidal_dim, hidden_dim),
        }
    }

    pub fn forward(&self, timestep: &Tensor, hidden_dtype: tch::Kind) -> Tensor {
        let proj = self.time_proj.forward(timestep);
        self.embedder.forward(&proj.to_kind(hidden_dtype))
    }
}
