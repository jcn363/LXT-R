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

    pub fn forward(&self, timestep: &Tensor, hidden_dtype: tch::Kind) -> Tensor {
        let proj = self.time_proj.forward(timestep);
        self.embedder.forward(&proj.to_kind(hidden_dtype))
    }
}
