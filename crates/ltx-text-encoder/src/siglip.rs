use std::borrow::Borrow;

use tch::nn::{Linear, ModuleT, Path};
use tch::Tensor;

use ltx_attention::scaled_dot_product_attention;
use ltx_norm::RMSNorm;

use crate::config::SigLIPConfigData;

pub struct SigLIPVisionMLP {
    fc1: Linear,
    fc2: Linear,
}

impl SigLIPVisionMLP {
    fn new<'a>(vs: impl Borrow<Path<'a>>, config: &SigLIPConfigData) -> Self {
        let vs = vs.borrow();
        let linear_cfg = tch::nn::LinearConfig {
            bias: true,
            ..Default::default()
        };
        Self {
            fc1: tch::nn::linear(
                vs / "fc1",
                config.hidden_size,
                config.intermediate_size,
                linear_cfg,
            ),
            fc2: tch::nn::linear(
                vs / "fc2",
                config.intermediate_size,
                config.hidden_size,
                linear_cfg,
            ),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let x = self.fc1.forward_t(x, false);
        let x = x.gelu("none");
        self.fc2.forward_t(&x, false)
    }
}

pub struct SigLIPVisionAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: i64,
    head_dim: i64,
}

impl SigLIPVisionAttention {
    fn new<'a>(vs: impl Borrow<Path<'a>>, config: &SigLIPConfigData) -> Self {
        let vs = vs.borrow();
        let head_dim = config.hidden_size / config.num_attention_heads;
        let linear_cfg = tch::nn::LinearConfig {
            bias: true,
            ..Default::default()
        };
        Self {
            q_proj: tch::nn::linear(vs / "q_proj", config.hidden_size, config.hidden_size, linear_cfg),
            k_proj: tch::nn::linear(vs / "k_proj", config.hidden_size, config.hidden_size, linear_cfg),
            v_proj: tch::nn::linear(vs / "v_proj", config.hidden_size, config.hidden_size, linear_cfg),
            out_proj: tch::nn::linear(vs / "out_proj", config.hidden_size, config.hidden_size, linear_cfg),
            num_heads: config.num_attention_heads,
            head_dim,
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let b = x.size()[0];
        let n = x.size()[1];

        let q = self.q_proj.forward_t(x, false)
            .reshape([b, n, self.num_heads, self.head_dim])
            .transpose(1, 2);
        let k = self.k_proj.forward_t(x, false)
            .reshape([b, n, self.num_heads, self.head_dim])
            .transpose(1, 2);
        let v = self.v_proj.forward_t(x, false)
            .reshape([b, n, self.num_heads, self.head_dim])
            .transpose(1, 2);

        let attn = scaled_dot_product_attention(&q, &k, &v, None, false);
        let attn = attn
            .transpose(1, 2)
            .reshape([b, n, self.num_heads * self.head_dim]);
        self.out_proj.forward_t(&attn, false)
    }
}

pub struct SigLIPVisionBlock {
    norm1: RMSNorm,
    attn: SigLIPVisionAttention,
    norm2: RMSNorm,
    mlp: SigLIPVisionMLP,
}

impl SigLIPVisionBlock {
    fn new<'a>(vs: impl Borrow<Path<'a>>, config: &SigLIPConfigData) -> Self {
        let vs = vs.borrow();
        Self {
            norm1: RMSNorm::default_eps_with_path(vs / "norm1", config.hidden_size),
            attn: SigLIPVisionAttention::new(vs / "attn", config),
            norm2: RMSNorm::default_eps_with_path(vs / "norm2", config.hidden_size),
            mlp: SigLIPVisionMLP::new(vs / "mlp", config),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let residual = x;
        let hidden = self.norm1.forward(x);
        let hidden = self.attn.forward(&hidden);
        let x = residual + hidden;

        let residual = &x;
        let hidden = self.norm2.forward(&x);
        let hidden = self.mlp.forward(&hidden);
        residual + hidden
    }
}

pub struct SigLIPVisionTower {
    patch_embed_weight: Tensor,
    position_embed_weight: Tensor,
    layers: Vec<SigLIPVisionBlock>,
    post_layernorm: RMSNorm,
    hidden_size: i64,
    patch_size: i64,
}

impl SigLIPVisionTower {
    pub fn new<'a>(vs: impl Borrow<Path<'a>>, config: &SigLIPConfigData) -> Self {
        let vs = vs.borrow();
        let num_patches =
            (config.image_size / config.patch_size) * (config.image_size / config.patch_size);

        let patch_embed_weight = vs.var(
            "patch_embed_weight",
            &[config.hidden_size, 3, config.patch_size, config.patch_size],
            tch::nn::init::DEFAULT_KAIMING_UNIFORM,
        );
        let position_embed_weight = vs.var(
            "position_embed_weight",
            &[1, num_patches + 1, config.hidden_size],
            tch::nn::init::DEFAULT_KAIMING_UNIFORM,
        );

        let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
        for i in 0..config.num_hidden_layers {
            layers.push(SigLIPVisionBlock::new(vs / format!("encoder.layers.{i}"), config));
        }

        Self {
            patch_embed_weight,
            position_embed_weight,
            layers,
            post_layernorm: RMSNorm::default_eps_with_path(vs / "post_layernorm", config.hidden_size),
            hidden_size: config.hidden_size,
            patch_size: config.patch_size,
        }
    }

    fn patch_embed(&self, pixel_values: &Tensor) -> Tensor {
        let b = pixel_values.size()[0];
        let c = pixel_values.size()[1];
        let h = pixel_values.size()[2];
        let w = pixel_values.size()[3];
        let ps = self.patch_size;
        let grid_h = h / ps;
        let grid_w = w / ps;

        let patches = pixel_values
            .reshape([b, c, grid_h, ps, grid_w, ps])
            .permute([0, 2, 4, 1, 3, 5])
            .reshape([b, grid_h * grid_w, c * ps * ps]);

        let weight = self
            .patch_embed_weight
            .reshape([self.hidden_size, c * ps * ps]);
        patches.matmul(&weight.transpose(0, 1))
    }

    pub fn forward(&self, pixel_values: &Tensor) -> Tensor {
        let b = pixel_values.size()[0];
        let mut hidden = self.patch_embed(pixel_values);

        let cls_token = self.position_embed_weight.narrow(1, 0, 1);
        let pos_embed = self.position_embed_weight.narrow(1, 1, hidden.size()[1]);
        hidden = Tensor::cat(&[&cls_token.expand([b, -1, -1], false), &hidden], 1);
        hidden += pos_embed;

        for layer in &self.layers {
            hidden = layer.forward(&hidden);
        }

        self.post_layernorm.forward(&hidden)
    }

    pub fn hidden_size(&self) -> i64 {
        self.hidden_size
    }
}
