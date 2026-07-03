use tch::nn::{Linear, Module};
use tch::Tensor;
use ltx_norm::RMSNorm;

use crate::block::BasicAVTransformerBlock;

pub struct LTXModel {
    blocks: Vec<BasicAVTransformerBlock>,
    norm_out: RMSNorm,
    proj_out: Linear,
}

impl LTXModel {
    pub fn new(
        blocks: Vec<BasicAVTransformerBlock>,
        norm_out: RMSNorm,
        proj_out: Linear,
    ) -> Self {
        Self { blocks, norm_out, proj_out }
    }

    pub fn forward(
        &self,
        latent: &Tensor,
        timestep: &Tensor,
        context: &Tensor,
        mask: Option<&Tensor>,
        pe: Option<(&Tensor, &Tensor)>,
    ) -> Tensor {
        let mut x = latent.shallow_clone();
        for block in &self.blocks {
            x = block.forward(&x, timestep, context, mask, pe);
        }
        let x = self.norm_out.forward(&x);
        self.proj_out.forward(&x)
    }
}

impl std::fmt::Debug for LTXModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LTXModel")
            .field("num_blocks", &self.blocks.len())
            .finish_non_exhaustive()
    }
}
