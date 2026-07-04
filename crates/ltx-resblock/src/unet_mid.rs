use std::borrow::Borrow;
use tch::nn::{Linear, Module, ModuleT, Path};
use tch::Tensor;

use crate::resblock_3d::ResnetBlock3D;
use ltx_types::NormLayerType;

/// THE ONLY UNetMidBlock3D in the codebase.
///
/// Mid block for the video VAE UNet-style encoder/decoder.
/// Contains self-attention sandwiched between two ResnetBlock3D modules.
///
/// Architecture: `x → ResBlock3D → SelfAttn → ResBlock3D → out`
pub struct UNetMidBlock3D {
    resblock1: ResnetBlock3D,
    attention: SelfAttention3D,
    resblock2: ResnetBlock3D,
}

impl std::fmt::Debug for UNetMidBlock3D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UNetMidBlock3D").finish()
    }
}

impl UNetMidBlock3D {
    pub fn new<'a>(
        vs: impl Borrow<Path<'a>>,
        channels: i64,
        norm_type: NormLayerType,
        norm_groups: i64,
        causal: bool,
        num_heads: i64,
    ) -> Self {
        let vs = vs.borrow();

        let resblock1 = ResnetBlock3D::new(
            vs / "resblock1",
            channels,
            channels,
            norm_type,
            norm_groups,
            causal,
        );
        let attention = SelfAttention3D::new(vs / "attention", channels, num_heads);
        let resblock2 = ResnetBlock3D::new(
            vs / "resblock2",
            channels,
            channels,
            norm_type,
            norm_groups,
            causal,
        );

        Self {
            resblock1,
            attention,
            resblock2,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.resblock1.forward(x);
        let h = self.attention.forward(&h);
        self.resblock2.forward(&h)
    }
}

impl ModuleT for UNetMidBlock3D {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        self.forward(xs)
    }
}

/// Simple self-attention for 5D video tensors (B, C, T, H, W).
///
/// Reshapes to (B, T*H*W, C), applies multi-head self-attention, reshapes back.
struct SelfAttention3D {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: i64,
    scale: f64,
}

impl std::fmt::Debug for SelfAttention3D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SelfAttention3D")
            .field("num_heads", &self.num_heads)
            .finish()
    }
}

impl SelfAttention3D {
    fn new<'a>(vs: impl Borrow<Path<'a>>, channels: i64, num_heads: i64) -> Self {
        let vs = vs.borrow();
        let head_dim = channels / num_heads;
        let scale = (head_dim as f64).powf(-0.5);

        let q_proj = tch::nn::linear(vs / "q_proj", channels, channels, Default::default());
        let k_proj = tch::nn::linear(vs / "k_proj", channels, channels, Default::default());
        let v_proj = tch::nn::linear(vs / "v_proj", channels, channels, Default::default());
        let out_proj = tch::nn::linear(vs / "out_proj", channels, channels, Default::default());

        Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            scale,
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let input_shape = x.size();
        let b = input_shape[0];
        let c = input_shape[1];
        let t = input_shape[2];
        let h_dim = input_shape[3];
        let w = input_shape[4];
        let n = t * h_dim * w;

        // (B, C, T*H*W) → (B, T*H*W, C)
        let x_flat = x.view([b, c, n]).transpose(1, 2);

        // Project
        let q = self.q_proj.forward(&x_flat);
        let k = self.k_proj.forward(&x_flat);
        let v = self.v_proj.forward(&x_flat);

        // (B, N, C) → (B, heads, N, head_dim)
        let head_dim = c / self.num_heads;
        let q = q.view([b, n, self.num_heads, head_dim]).transpose(1, 2);
        let k = k.view([b, n, self.num_heads, head_dim]).transpose(1, 2);
        let v = v.view([b, n, self.num_heads, head_dim]).transpose(1, 2);

        // Scaled dot-product attention: softmax(QK^T / sqrt(d)) V
        let attn = Tensor::einsum("bhnd,bhmd->bhnm", &[&q, &k], None::<i64>) * self.scale;
        let attn = attn.softmax(-1, tch::Kind::Float);

        let out = Tensor::einsum("bhnm,bhmd->bhnd", &[&attn, &v], None::<i64>);

        // (B, heads, N, head_dim) → (B, N, C) → (B, C, T, H, W)
        let out = out.transpose(1, 2).contiguous().view([b, n, c]);
        let out = self.out_proj.forward(&out);
        let out = out.transpose(1, 2).view([b, c, t, h_dim, w]);

        x + &out
    }
}
