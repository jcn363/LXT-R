use tch::nn::Module;
use tch::Tensor;

use ltx_fp8::{dequantize_fp8, quantize_weight_to_fp8_per_tensor};

/// FP8 linear layer — quantizes weights to FP8 on forward pass.
#[derive(Debug)]
pub struct FP8Linear {
    weight: Tensor,
    scale: Tensor,
    bias: Option<Tensor>,
    in_features: i64,
    out_features: i64,
}

impl FP8Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        let (q_weight, scale) = quantize_weight_to_fp8_per_tensor(&weight);
        let out_features = weight.size()[0];
        let in_features = weight.size()[1];
        Self {
            weight: q_weight,
            scale,
            bias,
            in_features,
            out_features,
        }
    }

    pub fn in_features(&self) -> i64 {
        self.in_features
    }

    pub fn out_features(&self) -> i64 {
        self.out_features
    }
}

impl Module for FP8Linear {
    fn forward(&self, x: &Tensor) -> Tensor {
        let w_f32 = dequantize_fp8(&self.weight, &self.scale, tch::Kind::Float);
        let out = x.matmul(&w_f32.to_dtype(x.kind(), false, false).transpose(0, 1));
        match &self.bias {
            Some(b) => out + b,
            None => out,
        }
    }
}
