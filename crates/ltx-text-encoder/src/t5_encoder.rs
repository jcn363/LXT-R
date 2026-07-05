use tch::nn::{Linear, ModuleT};
use tch::Tensor;

use crate::config::T5ConfigData;

/// T5 LayerNorm — weight-only (no bias).
pub struct T5LayerNorm {
    weight: Tensor,
    eps: f64,
}

impl T5LayerNorm {
    fn new(vs: &tch::nn::Path, size: i64, eps: f64) -> Self {
        Self {
            weight: vs.var("weight", &[size], tch::nn::init::Init::Const(1.0)),
            eps,
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let x_f32 = x.to_kind(tch::Kind::Float);
        let mean = x_f32.mean_dim(-1i64, true, tch::Kind::Float);
        let var = (&x_f32 - &mean).pow_tensor_scalar(2).mean_dim(-1i64, true, tch::Kind::Float);
        let norm = (&x_f32 - &mean) / (var + self.eps).sqrt();
        (norm * &self.weight).to_kind(x.kind())
    }
}

/// T5 relative position bias — shared across all layers.
pub struct T5RelativePositionBias {
    weight: Tensor,
    num_buckets: i64,
    max_distance: i64,
    num_heads: i64,
}

impl T5RelativePositionBias {
    fn new(vs: &tch::nn::Path, config: &T5ConfigData) -> Self {
        Self {
            weight: vs.var(
                "weight",
                &[config.relative_attention_num_buckets, config.num_heads],
                tch::nn::init::Init::Const(0.0),
            ),
            num_buckets: config.relative_attention_num_buckets,
            max_distance: config.relative_attention_max_distance,
            num_heads: config.num_heads,
        }
    }

    fn relative_position_bucket(&self, relative_position: &Tensor) -> Tensor {
        let half = self.num_buckets / 2;
        let is_positive = relative_position.gt(0);
        let relative_position = relative_position.abs();

        let neg_val = relative_position.to_kind(tch::Kind::Float);
        let log_ratio = (self.max_distance as f64).ln() / (half as f64).ln();
        let neg_bucket = (half as f64 - neg_val.log() / log_ratio)
            .to_kind(tch::Kind::Int64)
            .clamp(0, half - 1);

        let pos_bucket = (half as f64
            + &neg_val / self.max_distance as f64 * (half as f64 - 1.0))
            .to_kind(tch::Kind::Int64)
            .clamp(half, self.num_buckets - 1);

        pos_bucket.where_self(&is_positive, &neg_bucket)
    }

    fn forward(&self, query_len: i64, key_len: i64, device: tch::Device) -> Tensor {
        let ctx_pos = Tensor::arange_start(0i64, query_len, (tch::Kind::Int64, device)).unsqueeze(1);
        let mem_pos = Tensor::arange_start(0i64, key_len, (tch::Kind::Int64, device)).unsqueeze(0);
        let relative = &mem_pos - &ctx_pos;

        let buckets = self.relative_position_bucket(&relative);
        let bias = self.weight.index_select(0, &buckets.flatten(0, -1));
        bias.reshape([query_len, key_len, self.num_heads])
            .permute([2, 0, 1])
            .unsqueeze(0)
    }
}

/// T5 multi-head self-attention with relative position bias.
pub struct T5Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    num_heads: i64,
    head_dim: i64,
    inner_dim: i64,
}

impl T5Attention {
    fn new(vs: &tch::nn::Path, config: &T5ConfigData) -> Self {
        let linear_cfg = tch::nn::LinearConfig { bias: false, ..Default::default() };
        let inner_dim = config.num_heads * config.d_kv;
        Self {
            q: tch::nn::linear(vs / "q", config.d_model, inner_dim, linear_cfg),
            k: tch::nn::linear(vs / "k", config.d_model, inner_dim, linear_cfg),
            v: tch::nn::linear(vs / "v", config.d_model, inner_dim, linear_cfg),
            o: tch::nn::linear(vs / "o", inner_dim, config.d_model, linear_cfg),
            num_heads: config.num_heads,
            head_dim: config.d_kv,
            inner_dim,
        }
    }

    fn forward(&self, x: &Tensor, position_bias: &Tensor) -> Tensor {
        let b = x.size()[0];
        let seq_len = x.size()[1];

        let q = self.q.forward_t(x, false)
            .reshape([b, seq_len, self.num_heads, self.head_dim])
            .transpose(1, 2);
        let k = self.k.forward_t(x, false)
            .reshape([b, seq_len, self.num_heads, self.head_dim])
            .transpose(1, 2);
        let v = self.v.forward_t(x, false)
            .reshape([b, seq_len, self.num_heads, self.head_dim])
            .transpose(1, 2);

        let scale = (self.head_dim as f64).powf(-0.5);
        let scores = q.matmul(&k.transpose(-2, -1)) * scale + position_bias;
        let attn = scores.softmax(-1, tch::Kind::Float).matmul(&v);

        let attn = attn.transpose(1, 2).reshape([b, seq_len, self.inner_dim]);
        self.o.forward_t(&attn, false)
    }
}

/// T5 gated feed-forward: gelu(wi_0(x)) * wi_1(x) -> wo
pub struct T5DenseGatedGELU {
    wi_0: Linear,
    wi_1: Linear,
    wo: Linear,
}

impl T5DenseGatedGELU {
    fn new(vs: &tch::nn::Path, config: &T5ConfigData) -> Self {
        let linear_cfg = tch::nn::LinearConfig { bias: false, ..Default::default() };
        Self {
            wi_0: tch::nn::linear(vs / "wi_0", config.d_model, config.d_ff, linear_cfg),
            wi_1: tch::nn::linear(vs / "wi_1", config.d_model, config.d_ff, linear_cfg),
            wo: tch::nn::linear(vs / "wo", config.d_ff, config.d_model, linear_cfg),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let gate = self.wi_0.forward_t(x, false).gelu("none");
        let up = self.wi_1.forward_t(x, false);
        self.wo.forward_t(&(&gate * &up), false)
    }
}

/// T5 encoder block.
pub struct T5Block {
    layer_norm_0: T5LayerNorm,
    self_attn: T5Attention,
    layer_norm_1: T5LayerNorm,
    dense: T5DenseGatedGELU,
}

impl T5Block {
    fn new(vs: &tch::nn::Path, config: &T5ConfigData) -> Self {
        Self {
            layer_norm_0: T5LayerNorm::new(&(vs / "0/layer_norm"), config.d_model, config.layer_norm_epsilon),
            self_attn: T5Attention::new(&(vs / "0/SelfAttention"), config),
            layer_norm_1: T5LayerNorm::new(&(vs / "1/layer_norm"), config.d_model, config.layer_norm_epsilon),
            dense: T5DenseGatedGELU::new(&(vs / "1/DenseReluDense"), config),
        }
    }

    fn forward(&self, x: &Tensor, position_bias: &Tensor) -> Tensor {
        let residual = x;
        let x = self.layer_norm_0.forward(x);
        let x = self.self_attn.forward(&x, position_bias);
        let x = residual + x;

        let residual = &x;
        let x = self.layer_norm_1.forward(&x);
        let x = self.dense.forward(&x);
        residual + x
    }
}

/// T5 encoder model.
pub struct T5EncoderModel {
    embed_tokens: Tensor,
    blocks: Vec<T5Block>,
    relative_attention_bias: T5RelativePositionBias,
    final_layer_norm: T5LayerNorm,
    config: T5ConfigData,
}

impl T5EncoderModel {
    pub fn new(vs: &tch::nn::Path, config: &T5ConfigData) -> Self {
        let embed_tokens = vs.var(
            "embed_tokens",
            &[config.vocab_size, config.d_model],
            tch::nn::init::Init::Const(0.0),
        );

        let relative_attention_bias = T5RelativePositionBias::new(
            &(vs / "block/0/layer/0/SelfAttention"),
            config,
        );

        let mut blocks = Vec::with_capacity(config.num_layers as usize);
        for i in 0..config.num_layers {
            blocks.push(T5Block::new(&(vs / format!("block/{i}")), config));
        }

        let final_layer_norm = T5LayerNorm::new(vs, config.d_model, config.layer_norm_epsilon);

        Self {
            embed_tokens,
            blocks,
            relative_attention_bias,
            final_layer_norm,
            config: config.clone(),
        }
    }

    pub fn forward(&self, input_ids: &Tensor) -> Tensor {
        let ids = input_ids.to_kind(tch::Kind::Int64);
        let flat = ids.flatten(0, -1);
        let mut hidden = self.embed_tokens.index_select(0, &flat)
            .reshape([ids.size()[0], ids.size()[1], self.config.d_model]);

        let seq_len = hidden.size()[1];
        let position_bias = self.relative_attention_bias.forward(seq_len, seq_len, hidden.device());

        for block in &self.blocks {
            hidden = block.forward(&hidden, &position_bias);
        }

        self.final_layer_norm.forward(&hidden)
    }

    pub fn hidden_size(&self) -> i64 {
        self.config.d_model
    }
}
