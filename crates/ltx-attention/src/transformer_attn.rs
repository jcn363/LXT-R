use ltx_norm::RMSNorm;
use std::borrow::Borrow;
use tch::nn::{Linear, ModuleT, Path};
use tch::Tensor;

use crate::rope::{self, RopeType};
use crate::sdpa;

pub struct TransformerAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    num_heads: i64,
    head_dim: i64,
    rope_type: RopeType,
    gate_logits: Option<Linear>,
}

impl std::fmt::Debug for TransformerAttention {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransformerAttention")
            .field("num_heads", &self.num_heads)
            .field("head_dim", &self.head_dim)
            .field("rope_type", &self.rope_type)
            .field("has_gate", &self.gate_logits.is_some())
            .finish()
    }
}

impl TransformerAttention {
    pub fn new<'a>(
        vs: impl Borrow<Path<'a>>,
        dim: i64,
        heads: i64,
        head_dim: i64,
        context_dim: Option<i64>,
        rope_type: RopeType,
    ) -> Self {
        let vs = vs.borrow();
        let context_dim = context_dim.unwrap_or(dim);
        Self {
            to_q: tch::nn::linear(vs / "to_q", dim, dim, Default::default()),
            to_k: tch::nn::linear(vs / "to_k", context_dim, dim, Default::default()),
            to_v: tch::nn::linear(vs / "to_v", context_dim, dim, Default::default()),
            to_out: tch::nn::linear(vs / "to_out", dim, dim, Default::default()),
            q_norm: RMSNorm::default_eps(dim, vs.device()),
            k_norm: RMSNorm::default_eps(dim, vs.device()),
            num_heads: heads,
            head_dim,
            rope_type,
            gate_logits: None,
        }
    }

    pub fn new_gated<'a>(
        vs: impl Borrow<Path<'a>>,
        dim: i64,
        heads: i64,
        head_dim: i64,
        context_dim: Option<i64>,
        rope_type: RopeType,
    ) -> Self {
        let vs = vs.borrow();
        let context_dim = context_dim.unwrap_or(dim);
        Self {
            to_q: tch::nn::linear(vs / "to_q", dim, dim, Default::default()),
            to_k: tch::nn::linear(vs / "to_k", context_dim, dim, Default::default()),
            to_v: tch::nn::linear(vs / "to_v", context_dim, dim, Default::default()),
            to_out: tch::nn::linear(vs / "to_out", dim, dim, Default::default()),
            q_norm: RMSNorm::default_eps(dim, vs.device()),
            k_norm: RMSNorm::default_eps(dim, vs.device()),
            num_heads: heads,
            head_dim,
            rope_type,
            gate_logits: Some(tch::nn::linear(vs / "gate_logits", dim, dim, Default::default())),
        }
    }

    pub fn forward(
        &self,
        x: &Tensor,
        context: Option<&Tensor>,
        mask: Option<&Tensor>,
        pe: Option<(&Tensor, &Tensor)>,
    ) -> Tensor {
        let context = context.unwrap_or(x);
        let mut q = self.q_norm.forward(&self.to_q.forward_t(x, false));
        let mut k = self.k_norm.forward(&self.to_k.forward_t(context, false));
        let v = self.to_v.forward_t(context, false);

        if let Some((cos, sin)) = pe {
            let (q_rot, k_rot) = rope::apply_rotary_emb(&q, &k, cos, sin, self.rope_type);
            q = q_rot;
            k = k_rot;
        }

        let b = x.size()[0];
        let q = q
            .reshape([b, -1, self.num_heads, self.head_dim])
            .transpose(1, 2);
        let k = k
            .reshape([b, -1, self.num_heads, self.head_dim])
            .transpose(1, 2);
        let v = v
            .reshape([b, -1, self.num_heads, self.head_dim])
            .transpose(1, 2);

        let attn = sdpa::scaled_dot_product_attention(&q, &k, &v, mask, false);
        let attn = attn
            .transpose(1, 2)
            .reshape([b, -1, self.num_heads * self.head_dim]);
        let out = self.to_out.forward_t(&attn, false);

        if let Some(ref gate) = self.gate_logits {
            let gate = gate.forward_t(x, false).sigmoid();
            out * gate
        } else {
            out
        }
    }
}

impl ModuleT for TransformerAttention {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let _ = train;
        self.forward(xs, None, None, None)
    }
}
