use ltx_attention::{make_attention, RopeType};
use tch::{Device, Kind, Tensor};

#[test]
fn test_make_attention_transformer() {
    // dim must equal head_dim for current RMSNorm impl
    let dim = 32;
    let attn = make_attention("transformer", dim, 1, dim, None, RopeType::Interleaved).unwrap();
    let x = Tensor::randn([1, 8, dim], (Kind::Float, Device::Cpu));
    let out = attn.forward_t(&x, false);
    assert_eq!(out.size(), vec![1, 8, dim]);
}

#[test]
fn test_make_attention_simple() {
    let attn = make_attention("simple", 64, 0, 0, None, RopeType::Interleaved).unwrap();
    let x = Tensor::randn([2, 64, 8, 8], (Kind::Float, Device::Cpu));
    let out = attn.forward_t(&x, false);
    assert_eq!(out.size(), vec![2, 64, 8, 8]);
}

#[test]
fn test_make_attention_gated() {
    let dim = 32;
    let attn = make_attention("gated", dim, 1, dim, None, RopeType::Interleaved).unwrap();
    let x = Tensor::randn([1, 8, dim], (Kind::Float, Device::Cpu));
    let out = attn.forward_t(&x, false);
    assert_eq!(out.size(), vec![1, 8, dim]);
}

#[test]
fn test_make_attention_unknown_type() {
    let result = make_attention("unknown", 32, 1, 32, None, RopeType::Interleaved);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), "Unknown attention type: unknown");
}
