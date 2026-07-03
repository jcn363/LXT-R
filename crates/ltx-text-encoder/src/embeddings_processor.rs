use tch::nn::Linear;
use tch::nn::ModuleT;
use tch::Tensor;

use ltx_norm::RMSNorm;
use ltx_types::NORM_EPS;

pub struct EmbeddingsProcessor {
    projection: Linear,
    norm: RMSNorm,
    hidden_size: i64,
}

impl EmbeddingsProcessor {
    pub fn new(input_size: i64, hidden_size: i64) -> Self {
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let root = vs.root();
        let linear_cfg = tch::nn::LinearConfig {
            bias: true,
            ..Default::default()
        };
        Self {
            projection: tch::nn::linear(&root / "projection", input_size, hidden_size, linear_cfg),
            norm: RMSNorm::new(hidden_size, NORM_EPS, tch::Device::Cpu),
            hidden_size,
        }
    }

    /// Project and normalize combined embeddings.
    pub fn forward(&self, embeddings: &Tensor) -> Tensor {
        let projected = self.projection.forward_t(embeddings, false);
        self.norm.forward(&projected)
    }

    /// Extract the CLS token embedding from a sequence.
    pub fn extract_cls(&self, embeddings: &Tensor) -> Tensor {
        embeddings.narrow(1, 0, 1).squeeze_dim(1)
    }

    /// Apply mean pooling over the sequence dimension.
    pub fn mean_pool(&self, embeddings: &Tensor) -> Tensor {
        embeddings.mean_dim(&[1i64][..], false, tch::Kind::Float)
    }

    pub fn hidden_size(&self) -> i64 {
        self.hidden_size
    }
}
