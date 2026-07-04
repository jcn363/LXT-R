use tch::nn::Linear;
use tch::nn::ModuleT;
use tch::Tensor;

use ltx_attention::scaled_dot_product_attention;
use ltx_norm::RMSNorm;
use ltx_types::NORM_EPS;

use crate::config::SigLIPConfigData;

pub struct SigLIPVisionMLP {
    fc1: Linear,
    fc2: Linear,
}

impl SigLIPVisionMLP {
    fn new(config: &SigLIPConfigData) -> Self {
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let root = vs.root();
        let linear_cfg = tch::nn::LinearConfig {
            bias: true,
            ..Default::default()
        };
        Self {
            fc1: tch::nn::linear(
                &root / "fc1",
                config.hidden_size,
                config.intermediate_size,
                linear_cfg,
            ),
            fc2: tch::nn::linear(
                &root / "fc2",
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
    fn new(config: &SigLIPConfigData) -> Self {
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let root = vs.root();
        let head_dim = config.hidden_size / config.num_attention_heads;
        let linear_cfg = tch::nn::LinearConfig {
            bias: true,
            ..Default::default()
        };
        Self {
            q_proj: tch::nn::linear(
                &root / "q_proj",
                config.hidden_size,
                config.hidden_size,
                linear_cfg,
            ),
            k_proj: tch::nn::linear(
                &root / "k_proj",
                config.hidden_size,
                config.hidden_size,
                linear_cfg,
            ),
            v_proj: tch::nn::linear(
                &root / "v_proj",
                config.hidden_size,
                config.hidden_size,
                linear_cfg,
            ),
            out_proj: tch::nn::linear(
                &root / "out_proj",
                config.hidden_size,
                config.hidden_size,
                linear_cfg,
            ),
            num_heads: config.num_attention_heads,
            head_dim,
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let b = x.size()[0];
        let n = x.size()[1];

        let q = self
            .q_proj
            .forward_t(x, false)
            .reshape([b, n, self.num_heads, self.head_dim])
            .transpose(1, 2);
        let k = self
            .k_proj
            .forward_t(x, false)
            .reshape([b, n, self.num_heads, self.head_dim])
            .transpose(1, 2);
        let v = self
            .v_proj
            .forward_t(x, false)
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
    fn new(config: &SigLIPConfigData) -> Self {
        let device = tch::Device::Cpu;
        Self {
            norm1: RMSNorm::new(config.hidden_size, NORM_EPS, device),
            attn: SigLIPVisionAttention::new(config),
            norm2: RMSNorm::new(config.hidden_size, NORM_EPS, device),
            mlp: SigLIPVisionMLP::new(config),
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
    pub fn new(config: &SigLIPConfigData) -> Self {
        let device = tch::Device::Cpu;
        let num_patches =
            (config.image_size / config.patch_size) * (config.image_size / config.patch_size);
        let patch_embed_weight = Tensor::randn(
            [config.hidden_size, 3, config.patch_size, config.patch_size],
            (tch::Kind::Float, device),
        );
        let position_embed_weight = Tensor::randn(
            [1, num_patches + 1, config.hidden_size],
            (tch::Kind::Float, device),
        );

        let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
        for _ in 0..config.num_hidden_layers {
            layers.push(SigLIPVisionBlock::new(config));
        }

        Self {
            patch_embed_weight,
            position_embed_weight,
            layers,
            post_layernorm: RMSNorm::new(config.hidden_size, NORM_EPS, device),
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
