use std::borrow::Borrow;

use tch::nn::{Linear, ModuleT, Path};
use tch::Tensor;

use ltx_attention::{
    apply_rotary_emb, precompute_freqs_cis, scaled_dot_product_attention, RopeType,
};
use ltx_norm::RMSNorm;

use crate::config::Gemma3ConfigData;

pub struct Gemma3MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Gemma3MLP {
    fn new<'a>(vs: impl Borrow<Path<'a>>, config: &Gemma3ConfigData) -> Self {
        let vs = vs.borrow();
        let linear_cfg = tch::nn::LinearConfig {
            bias: false,
            ..Default::default()
        };
        Self {
            gate_proj: tch::nn::linear(
                vs / "gate_proj",
                config.hidden_size,
                config.intermediate_size,
                linear_cfg,
            ),
            up_proj: tch::nn::linear(
                vs / "up_proj",
                config.hidden_size,
                config.intermediate_size,
                linear_cfg,
            ),
            down_proj: tch::nn::linear(
                vs / "down_proj",
                config.intermediate_size,
                config.hidden_size,
                linear_cfg,
            ),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let gate = self.gate_proj.forward_t(x, false).silu();
        let up = self.up_proj.forward_t(x, false);
        self.down_proj.forward_t(&(&gate * &up), false)
    }
}

pub struct Gemma3Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
}

impl Gemma3Attention {
    fn new<'a>(vs: impl Borrow<Path<'a>>, config: &Gemma3ConfigData) -> Self {
        let vs = vs.borrow();
        let linear_cfg = tch::nn::LinearConfig {
            bias: false,
            ..Default::default()
        };
        Self {
            q_proj: tch::nn::linear(
                vs / "q_proj",
                config.hidden_size,
                config.num_attention_heads * config.head_dim,
                linear_cfg,
            ),
            k_proj: tch::nn::linear(
                vs / "k_proj",
                config.hidden_size,
                config.num_key_value_heads * config.head_dim,
                linear_cfg,
            ),
            v_proj: tch::nn::linear(
                vs / "v_proj",
                config.hidden_size,
                config.num_key_value_heads * config.head_dim,
                linear_cfg,
            ),
            o_proj: tch::nn::linear(
                vs / "o_proj",
                config.num_attention_heads * config.head_dim,
                config.hidden_size,
                linear_cfg,
            ),
            q_norm: RMSNorm::default_eps_with_path(vs / "q_norm", config.head_dim),
            k_norm: RMSNorm::default_eps_with_path(vs / "k_norm", config.head_dim),
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
        }
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
        let b = x.size()[0];
        let seq_len = x.size()[1];

        let mut q = self.q_proj.forward_t(x, false);
        q = self.q_norm.forward(&q);
        let mut k = self.k_proj.forward_t(x, false);
        k = self.k_norm.forward(&k);
        let v = self.v_proj.forward_t(x, false);

        let (q_rot, k_rot) = apply_rotary_emb(&q, &k, cos, sin, RopeType::Interleaved);

        let q = q_rot
            .reshape([b, seq_len, self.num_heads, self.head_dim])
            .transpose(1, 2);
        let k = k_rot
            .reshape([b, seq_len, self.num_kv_heads, self.head_dim])
            .transpose(1, 2);
        let v = v
            .reshape([b, seq_len, self.num_kv_heads, self.head_dim])
            .transpose(1, 2);

        let repeat_factor = self.num_heads / self.num_kv_heads;
        let k = k.repeat_interleave_self_int(repeat_factor, 1, None);
        let v = v.repeat_interleave_self_int(repeat_factor, 1, None);

        let attn = scaled_dot_product_attention(&q, &k, &v, None, false);
        let attn = attn
            .transpose(1, 2)
            .reshape([b, seq_len, self.num_heads * self.head_dim]);
        self.o_proj.forward_t(&attn, false)
    }
}

pub struct Gemma3DecoderLayer {
    self_attn: Gemma3Attention,
    mlp: Gemma3MLP,
    input_norm: RMSNorm,
    post_attn_norm: RMSNorm,
}

impl Gemma3DecoderLayer {
    fn new<'a>(vs: impl Borrow<Path<'a>>, config: &Gemma3ConfigData) -> Self {
        let vs = vs.borrow();
        Self {
            self_attn: Gemma3Attention::new(vs / "self_attn", config),
            mlp: Gemma3MLP::new(vs / "mlp", config),
            input_norm: RMSNorm::default_eps_with_path(vs / "input_norm", config.hidden_size),
            post_attn_norm: RMSNorm::default_eps_with_path(vs / "post_attn_norm", config.hidden_size),
        }
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
        let residual = x;
        let hidden = self.input_norm.forward(x);
        let hidden = self.self_attn.forward(&hidden, cos, sin);
        let x = residual + hidden;

        let residual = &x;
        let hidden = self.post_attn_norm.forward(&x);
        let hidden = self.mlp.forward(&hidden);
        residual + hidden
    }
}

pub struct Gemma3TextModel {
    embed_tokens: Tensor,
    layers: Vec<Gemma3DecoderLayer>,
    norm: RMSNorm,
    config: Gemma3ConfigData,
    cos_cache: Tensor,
    sin_cache: Tensor,
}

impl Gemma3TextModel {
    pub fn new<'a>(vs: impl Borrow<Path<'a>>, config: &Gemma3ConfigData) -> Self {
        let vs = vs.borrow();
        let device = vs.device();

        let embed_tokens = vs.var(
            "embed_tokens",
            &[config.vocab_size, config.hidden_size],
            tch::nn::init::DEFAULT_KAIMING_UNIFORM,
        );

        let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
        for i in 0..config.num_hidden_layers {
            layers.push(Gemma3DecoderLayer::new(vs / format!("layers.{i}"), config));
        }

        let (cos_cache, sin_cache) = precompute_freqs_cis(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            RopeType::Interleaved,
            device,
        );

        Self {
            embed_tokens,
            layers,
            norm: RMSNorm::default_eps_with_path(vs / "norm", config.hidden_size),
            config: config.clone(),
            cos_cache,
            sin_cache,
        }
    }

    pub fn forward(&self, input_ids: &Tensor) -> Tensor {
        let seq_len = input_ids.size()[1];
        let cos = self.cos_cache.narrow(0, 0, seq_len);
        let sin = self.sin_cache.narrow(0, 0, seq_len);

        let b = input_ids.size()[0];
        let flat_ids = input_ids.to_kind(tch::Kind::Int64).flatten(0, -1);
        let hidden_states = self
            .embed_tokens
            .index_select(0, &flat_ids)
            .reshape([b, seq_len, self.config.hidden_size]);

        let mut hidden_states = hidden_states;
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, &cos, &sin);
        }

        self.norm.forward(&hidden_states)
    }

    pub fn hidden_size(&self) -> i64 {
        self.config.hidden_size
    }
}
