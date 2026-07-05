use safetensors::SafeTensors;
use tch::nn::{Linear, ModuleT};
use tch::Tensor;

use crate::config::T5ConfigData;

/// Load a tensor from checkpoint, keeping original dtype (FP16/BF16).
fn load_tensor(st: &SafeTensors, key: &str, shape: &[i64], device: tch::Device) -> Tensor {
    st.tensor(key)
        .map(|view| {
            let kind = match view.dtype() {
                safetensors::Dtype::F16 => tch::Kind::Half,
                safetensors::Dtype::BF16 => tch::Kind::BFloat16,
                _ => tch::Kind::Float,
            };
            Tensor::from_data_size(view.data(), shape, kind).to_device(device)
        })
        .unwrap_or_else(|_| Tensor::zeros(shape, (tch::Kind::Float, device)))
}

/// T5 LayerNorm — weight-only (no bias).
struct T5LayerNorm {
    weight: Tensor,
    eps: f64,
}

impl T5LayerNorm {
    fn forward(&self, x: &Tensor) -> Tensor {
        let x_f32 = x.to_kind(tch::Kind::Float);
        let mean = x_f32.mean_dim(-1i64, true, tch::Kind::Float);
        let var = (&x_f32 - &mean).pow_tensor_scalar(2).mean_dim(-1i64, true, tch::Kind::Float);
        let norm = (&x_f32 - &mean) / (var + self.eps).sqrt();
        (norm * &self.weight).to_kind(x.kind())
    }
}

/// T5 relative position bias — shared across all layers.
struct T5RelativePositionBias {
    weight: Tensor,
    num_buckets: i64,
    max_distance: i64,
    num_heads: i64,
}

impl T5RelativePositionBias {
    fn forward(&self, query_len: i64, key_len: i64, device: tch::Device) -> Tensor {
        let half = self.num_buckets / 2;
        let ctx_pos = Tensor::arange_start(0i64, query_len, (tch::Kind::Int64, device)).unsqueeze(1);
        let mem_pos = Tensor::arange_start(0i64, key_len, (tch::Kind::Int64, device)).unsqueeze(0);
        let relative = &mem_pos - &ctx_pos;

        let is_pos = relative.gt(0);
        let abs_rel = relative.abs().to_kind(tch::Kind::Float);
        let log_ratio = (self.max_distance as f64).ln() / (half as f64).ln();
        let neg_bucket = (half as f64 - abs_rel.log() / log_ratio)
            .to_kind(tch::Kind::Int64).clamp(0, half - 1);
        let pos_bucket = (half as f64 + &abs_rel / self.max_distance as f64 * (half as f64 - 1.0))
            .to_kind(tch::Kind::Int64).clamp(half, self.num_buckets - 1);
        let buckets = pos_bucket.where_self(&is_pos, &neg_bucket);

        let bias = self.weight.index_select(0, &buckets.flatten(0, -1));
        bias.reshape([query_len, key_len, self.num_heads])
            .permute([2, 0, 1]).unsqueeze(0)
    }
}

/// T5 multi-head self-attention.
struct T5Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    num_heads: i64,
    head_dim: i64,
    inner_dim: i64,
}

impl T5Attention {
    fn forward(&self, x: &Tensor, position_bias: &Tensor) -> Tensor {
        let b = x.size()[0];
        let seq_len = x.size()[1];

        let q = self.q.forward_t(x, false).reshape([b, seq_len, self.num_heads, self.head_dim]).transpose(1, 2);
        let k = self.k.forward_t(x, false).reshape([b, seq_len, self.num_heads, self.head_dim]).transpose(1, 2);
        let v = self.v.forward_t(x, false).reshape([b, seq_len, self.num_heads, self.head_dim]).transpose(1, 2);

        let scale = (self.head_dim as f64).powf(-0.5);
        let scores = q.matmul(&k.transpose(-2, -1)) * scale + position_bias;
        let attn = scores.softmax(-1, tch::Kind::Float).matmul(&v);
        let attn = attn.transpose(1, 2).reshape([b, seq_len, self.inner_dim]);
        self.o.forward_t(&attn, false)
    }
}

/// T5 gated feed-forward.
struct T5DenseGatedGELU {
    wi_0: Linear,
    wi_1: Linear,
    wo: Linear,
}

impl T5DenseGatedGELU {
    fn forward(&self, x: &Tensor) -> Tensor {
        let gate = self.wi_0.forward_t(x, false).gelu("none");
        let up = self.wi_1.forward_t(x, false);
        self.wo.forward_t(&(&gate * &up), false)
    }
}

/// T5 encoder block.
struct T5Block {
    ln_0: T5LayerNorm,
    attn: T5Attention,
    ln_1: T5LayerNorm,
    ffn: T5DenseGatedGELU,
}

impl T5Block {
    fn forward(&self, x: &Tensor, position_bias: &Tensor) -> Tensor {
        let h = x + self.attn.forward(&self.ln_0.forward(x), position_bias);
        &h + self.ffn.forward(&self.ln_1.forward(&h))
    }
}

/// T5 encoder model — loaded directly from checkpoint (no VarStore).
/// Tensors kept in original dtype (FP16/BF16), cutting memory ~50%.
pub struct T5EncoderModel {
    embed_tokens: Tensor,
    blocks: Vec<T5Block>,
    rel_bias: T5RelativePositionBias,
    final_ln: T5LayerNorm,
    d_model: i64,
}

impl T5EncoderModel {
    /// Load from checkpoint. Tensors stay in original dtype (FP16/BF16).
    /// ~9GB for FP16 checkpoint vs ~18GB with VarStore (FP32).
    pub fn from_checkpoint(st: &SafeTensors, config: &T5ConfigData, device: tch::Device) -> Self {
        let dm = config.d_model;
        let dv = config.d_kv;
        let nh = config.num_heads;
        let df = config.d_ff;
        let eps = config.layer_norm_epsilon;

        let embed_tokens = load_tensor(st, "shared.weight", &[config.vocab_size, dm], device);

        let rel_bias = T5RelativePositionBias {
            weight: load_tensor(st, "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight", &[32, nh], device),
            num_buckets: 32,
            max_distance: 128,
            num_heads: nh,
        };

        let mut blocks = Vec::with_capacity(config.num_layers as usize);
        for i in 0..config.num_layers {
            let p = format!("encoder.block.{i}");
            blocks.push(T5Block {
                ln_0: T5LayerNorm { weight: load_tensor(st, &format!("{p}.layer.0.layer_norm.weight"), &[dm], device), eps },
                attn: {
                    let inner = nh * dv;
                    T5Attention {
                        q: Linear { ws: load_tensor(st, &format!("{p}.layer.0.SelfAttention.q.weight"), &[dm, inner], device), bs: None },
                        k: Linear { ws: load_tensor(st, &format!("{p}.layer.0.SelfAttention.k.weight"), &[dm, inner], device), bs: None },
                        v: Linear { ws: load_tensor(st, &format!("{p}.layer.0.SelfAttention.v.weight"), &[dm, inner], device), bs: None },
                        o: Linear { ws: load_tensor(st, &format!("{p}.layer.0.SelfAttention.o.weight"), &[inner, dm], device), bs: None },
                        num_heads: nh, head_dim: dv, inner_dim: inner,
                    }
                },
                ln_1: T5LayerNorm { weight: load_tensor(st, &format!("{p}.layer.1.layer_norm.weight"), &[dm], device), eps },
                ffn: T5DenseGatedGELU {
                    wi_0: Linear { ws: load_tensor(st, &format!("{p}.layer.1.DenseReluDense.wi_0.weight"), &[df, dm], device), bs: None },
                    wi_1: Linear { ws: load_tensor(st, &format!("{p}.layer.1.DenseReluDense.wi_1.weight"), &[df, dm], device), bs: None },
                    wo: Linear { ws: load_tensor(st, &format!("{p}.layer.1.DenseReluDense.wo.weight"), &[dm, df], device), bs: None },
                },
            });
        }

        let final_ln = T5LayerNorm {
            weight: load_tensor(st, "encoder.final_layer_norm.weight", &[dm], device),
            eps,
        };

        Self { embed_tokens, blocks, rel_bias, final_ln, d_model: dm }
    }

    pub fn forward(&self, input_ids: &Tensor) -> Tensor {
        let ids = input_ids.to_kind(tch::Kind::Int64);
        let flat = ids.flatten(0, -1);
        let mut h = self.embed_tokens.index_select(0, &flat)
            .reshape([ids.size()[0], ids.size()[1], self.d_model]);

        let seq_len = h.size()[1];
        let pos_bias = self.rel_bias.forward(seq_len, seq_len, h.device());

        for block in &self.blocks {
            h = block.forward(&h, &pos_bias);
        }
        self.final_ln.forward(&h)
    }

    pub fn hidden_size(&self) -> i64 {
        self.d_model
    }
}
