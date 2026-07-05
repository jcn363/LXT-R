use std::borrow::Borrow;

use tch::nn::{Linear, ModuleT, Path};
use tch::Tensor;

use ltx_norm::RMSNorm;

pub struct EmbeddingsProcessor {
    projection: Linear,
    norm: RMSNorm,
    hidden_size: i64,
}

impl EmbeddingsProcessor {
    pub fn new<'a>(vs: impl Borrow<Path<'a>>, input_size: i64, hidden_size: i64) -> Self {
        let vs = vs.borrow();
        let linear_cfg = tch::nn::LinearConfig {
            bias: true,
            ..Default::default()
        };
        Self {
            projection: tch::nn::linear(vs / "projection", input_size, hidden_size, linear_cfg),
            norm: RMSNorm::default_eps_with_path(vs / "norm", hidden_size),
            hidden_size,
        }
    }

    pub fn forward(&self, embeddings: &Tensor) -> Tensor {
        let projected = self.projection.forward_t(embeddings, false);
        self.norm.forward(&projected)
    }

    pub fn extract_cls(&self, embeddings: &Tensor) -> Tensor {
        embeddings.narrow(1, 0, 1).squeeze_dim(1)
    }

    pub fn mean_pool(&self, embeddings: &Tensor) -> Tensor {
        embeddings.mean_dim(&[1i64][..], false, tch::Kind::Float)
    }

    pub fn hidden_size(&self) -> i64 {
        self.hidden_size
    }
}
