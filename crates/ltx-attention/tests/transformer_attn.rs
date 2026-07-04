use ltx_attention::{RopeType, TransformerAttention};
use tch::nn::ModuleT;
use tch::{Device, Kind, Tensor};

fn make_vs() -> tch::nn::VarStore {
    tch::nn::VarStore::new(Device::Cpu)
}

#[test]
fn test_transformer_attention_new() {
    let vs = make_vs();
    let dim = 32;
    let attn = TransformerAttention::new(&vs.root(), dim, 1, dim, None, RopeType::Interleaved);
    let x = Tensor::randn([1, 16, dim], (Kind::Float, Device::Cpu));
    let out = attn.forward_t(&x, false);
    assert_eq!(out.size(), vec![1, 16, dim]);
}

#[test]
fn test_transformer_attention_with_context() {
    let vs = make_vs();
    let dim = 16;
    let context_dim = 32;
    let attn = TransformerAttention::new(&vs.root(), dim, 1, dim, Some(context_dim), RopeType::Split);
    let x = Tensor::randn([1, 8, dim], (Kind::Float, Device::Cpu));
    let context = Tensor::randn([1, 10, context_dim], (Kind::Float, Device::Cpu));
    let out = attn.forward(&x, Some(&context), None, None);
    assert_eq!(out.size(), vec![1, 8, dim]);
}

#[test]
fn test_transformer_attention_gated() {
    let vs = make_vs();
    let dim = 32;
    let attn = TransformerAttention::new_gated(&vs.root(), dim, 1, dim, None, RopeType::Interleaved);
    let x = Tensor::randn([1, 8, dim], (Kind::Float, Device::Cpu));
    let out = attn.forward_t(&x, false);
    assert_eq!(out.size(), vec![1, 8, dim]);
}

#[test]
fn test_transformer_attention_small() {
    let vs = make_vs();
    let dim = 16;
    let attn = TransformerAttention::new(&vs.root(), dim, 1, dim, None, RopeType::Interleaved);
    let x = Tensor::randn([1, 4, dim], (Kind::Float, Device::Cpu));
    let out = attn.forward_t(&x, false);
    assert_eq!(out.size(), vec![1, 4, dim]);
}

/// Bug verification: q_norm/k_norm must use `dim`, not `head_dim`.
#[test]
fn test_multi_head_qknorm_uses_dim() {
    let vs = make_vs();
    let dim = 64;
    let heads = 4;
    let head_dim = 16;
    let attn = TransformerAttention::new(&vs.root(), dim, heads, head_dim, None, RopeType::Interleaved);
    let x = Tensor::randn([2, 8, dim], (Kind::Float, Device::Cpu));
    let out = attn.forward(&x, None, None, None);
    assert_eq!(out.size(), vec![2, 8, dim]);
    assert!(out.abs().sum(Kind::Float).double_value(&[]) > 0.0);
}

/// Bug verification: multi-head with different context_dim and RoPE.
#[test]
fn test_multi_head_with_context_and_rope() {
    let vs = make_vs();
    let dim = 64;
    let heads = 4;
    let head_dim = 16;
    let context_dim = 48;
    let attn = TransformerAttention::new(&vs.root(), dim, heads, head_dim, Some(context_dim), RopeType::Split);
    let x = Tensor::randn([1, 6, dim], (Kind::Float, Device::Cpu));
    let ctx = Tensor::randn([1, 4, context_dim], (Kind::Float, Device::Cpu));
    let out = attn.forward(&x, Some(&ctx), None, None);
    assert_eq!(out.size(), vec![1, 6, dim]);
}
