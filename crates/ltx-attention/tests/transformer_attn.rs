use ltx_attention::{TransformerAttention, RopeType};
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
