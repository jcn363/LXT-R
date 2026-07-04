use ltx_types::NORM_EPS;
use tch::nn::{Conv2D, ModuleT};
use tch::Tensor;

pub struct SimpleAttnBlock {
    norm: tch::nn::GroupNorm,
    q: Conv2D,
    k: Conv2D,
    v: Conv2D,
    proj_out: Conv2D,
}

impl std::fmt::Debug for SimpleAttnBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimpleAttnBlock").finish()
    }
}

impl SimpleAttnBlock {
    pub fn new(channels: i64) -> Self {
        let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        let root = vs.root();
        let conv_cfg = tch::nn::ConvConfig {
            padding: 0,
            stride: 1,
            ..Default::default()
        };
        let gn_cfg = tch::nn::GroupNormConfig {
            eps: NORM_EPS,
            affine: true,
            ..Default::default()
        };
        Self {
            norm: tch::nn::group_norm(&root / "norm", 32, channels, gn_cfg),
            q: tch::nn::conv2d(&root / "q", channels, channels, 1, conv_cfg),
            k: tch::nn::conv2d(&root / "k", channels, channels, 1, conv_cfg),
            v: tch::nn::conv2d(&root / "v", channels, channels, 1, conv_cfg),
            proj_out: tch::nn::conv2d(&root / "proj_out", channels, channels, 1, conv_cfg),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.norm.forward_t(x, false);
        let (b, c, height, width) = h.size4().unwrap();
        let hw = height * width;

        let q = self
            .q
            .forward_t(&h, false)
            .reshape([b, c, hw])
            .transpose(1, 2);
        let k = self.k.forward_t(&h, false).reshape([b, c, hw]);
        let v = self
            .v
            .forward_t(&h, false)
            .reshape([b, c, hw])
            .transpose(1, 2);

        let scale = (c as f64).powf(-0.5);
        let w = q.matmul(&k) * scale;
        let w = w.softmax(-1, tch::Kind::Float);
        let h = w.matmul(&v).transpose(1, 2).reshape([b, c, height, width]);
        x + self.proj_out.forward_t(&h, false)
    }
}

impl ModuleT for SimpleAttnBlock {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let _ = train;
        self.forward(xs)
    }
}
