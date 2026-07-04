use ltx_attention::{RopeType, TransformerAttention};
use ltx_norm::RMSNorm;
use ltx_timestep::AdaLayerNormSingle;
use ltx_types::DEFAULT_SINUSOIDAL_DIM;
use tch::Tensor;

use crate::feed_forward::FeedForward;

pub struct BasicAVTransformerBlock {
    adaln: AdaLayerNormSingle,
    self_attn: TransformerAttention,
    cross_attn: TransformerAttention,
    norm1: RMSNorm,
    norm_cross: RMSNorm,
    norm2: RMSNorm,
    ff: FeedForward,
}

impl BasicAVTransformerBlock {
    pub fn new(
        vs: &tch::nn::Path,
        dim: i64,
        num_heads: i64,
        head_dim: i64,
        context_dim: Option<i64>,
        rope_type: RopeType,
    ) -> Self {
        let adaln =
            AdaLayerNormSingle::new_with_input_dim(&(vs / "adaln"), dim, DEFAULT_SINUSOIDAL_DIM);
        let self_attn = TransformerAttention::new(
            &(vs / "self_attn"),
            dim,
            num_heads,
            head_dim,
            None,
            rope_type,
        );
        let cross_attn = TransformerAttention::new(
            &(vs / "cross_attn"),
            dim,
            num_heads,
            head_dim,
            context_dim,
            rope_type,
        );
        let norm1 = RMSNorm::default_eps_with_path(vs / "norm1", dim);
        let norm_cross = RMSNorm::default_eps_with_path(vs / "norm_cross", dim);
        let norm2 = RMSNorm::default_eps_with_path(vs / "norm2", dim);
        let ff = FeedForward::new(&(vs / "ff"), dim);

        Self {
            adaln,
            self_attn,
            cross_attn,
            norm1,
            norm_cross,
            norm2,
            ff,
        }
    }

    pub fn forward(
        &self,
        x: &Tensor,
        timestep: &Tensor,
        context: &Tensor,
        mask: Option<&Tensor>,
        pe: Option<(&Tensor, &Tensor)>,
    ) -> Tensor {
        let (modulation, _) = self.adaln.forward(timestep, x.kind());
        let chunks: Vec<Tensor> = modulation.chunk(6, -1);
        // Unsqueeze to (B, 1, dim) for broadcasting with (B, seq, dim)
        let (shift_msa, scale_msa, gate_msa) = (
            chunks[0].unsqueeze(1),
            chunks[1].unsqueeze(1),
            chunks[2].unsqueeze(1),
        );
        let (shift_mlp, scale_mlp, gate_mlp) = (
            chunks[3].unsqueeze(1),
            chunks[4].unsqueeze(1),
            chunks[5].unsqueeze(1),
        );

        let h = self.norm1.forward(x) * (Tensor::ones_like(&scale_msa) + &scale_msa) + &shift_msa;
        let h = self.self_attn.forward(&h, None, mask, pe);
        let x = x + &gate_msa * h;

        let h = self.norm_cross.forward(&x);
        let h = self.cross_attn.forward(&h, Some(context), mask, None);
        let x = x + h;

        let h = self.norm2.forward(&x) * (Tensor::ones_like(&scale_mlp) + &scale_mlp) + &shift_mlp;
        let h = self.ff.forward(&h);
        x + &gate_mlp * h
    }
}

impl std::fmt::Debug for BasicAVTransformerBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BasicAVTransformerBlock")
            .finish_non_exhaustive()
    }
}
