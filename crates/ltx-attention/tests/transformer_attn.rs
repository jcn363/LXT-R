use ltx_attention::{RopeType, TransformerAttention};
use tch::nn::ModuleT;
use tch::{Device, Kind, Tensor};

#[test]
fn test_transformer_attention_new() {
    // dim must equal head_dim for current RMSNorm impl
    let dim = 32;
    let attn = TransformerAttention::new(dim, 1, dim, None, RopeType::Interleaved);
    let x = Tensor::randn([1, 16, dim], (Kind::Float, Device::Cpu));
    let out = attn.forward_t(&x, false);
    assert_eq!(out.size(), vec![1, 16, dim]);
}

#[test]
fn test_transformer_attention_with_context() {
    let dim = 16;
    let context_dim = 32;
    let attn = TransformerAttention::new(dim, 1, dim, Some(context_dim), RopeType::Split);
    let x = Tensor::randn([1, 8, dim], (Kind::Float, Device::Cpu));
    let context = Tensor::randn([1, 10, context_dim], (Kind::Float, Device::Cpu));
    let out = attn.forward(&x, Some(&context), None, None);
    assert_eq!(out.size(), vec![1, 8, dim]);
}

#[test]
fn test_transformer_attention_gated() {
    let dim = 32;
    let attn = TransformerAttention::new_gated(dim, 1, dim, None, RopeType::Interleaved);
    let x = Tensor::randn([1, 8, dim], (Kind::Float, Device::Cpu));
    let out = attn.forward_t(&x, false);
    assert_eq!(out.size(), vec![1, 8, dim]);
}

#[test]
fn test_transformer_attention_small() {
    let dim = 16;
    let attn = TransformerAttention::new(dim, 1, dim, None, RopeType::Interleaved);
    let x = Tensor::randn([1, 4, dim], (Kind::Float, Device::Cpu));
    let out = attn.forward_t(&x, false);
    assert_eq!(out.size(), vec![1, 4, dim]);
}

/// Bug verification: q_norm/k_norm must use `dim`, not `head_dim`.
/// This test uses `heads=4, head_dim=16` (so `dim=64 != head_dim`).
/// Before the fix, this would panic with a shape mismatch in RMSNorm.
#[test]
fn test_multi_head_qknorm_uses_dim() {
    let dim = 64;
    let heads = 4;
    let head_dim = 16;
    let attn = TransformerAttention::new(dim, heads, head_dim, None, RopeType::Interleaved);
    let x = Tensor::randn([2, 8, dim], (Kind::Float, Device::Cpu));
    let out = attn.forward(&x, None, None, None);
    assert_eq!(out.size(), vec![2, 8, dim]);
    // Output must be non-zero (projection + norm + attention happened)
    assert!(out.abs().sum(Kind::Float).double_value(&[]) > 0.0);
}

/// Bug verification: multi-head with different context_dim and RoPE.
#[test]
fn test_multi_head_with_context_and_rope() {
    let dim = 64;
    let heads = 4;
    let head_dim = 16;
    let context_dim = 48;
    let attn = TransformerAttention::new(dim, heads, head_dim, Some(context_dim), RopeType::Split);
    let x = Tensor::randn([1, 6, dim], (Kind::Float, Device::Cpu));
    let ctx = Tensor::randn([1, 4, context_dim], (Kind::Float, Device::Cpu));
    let out = attn.forward(&x, Some(&ctx), None, None);
    assert_eq!(out.size(), vec![1, 6, dim]);
}
