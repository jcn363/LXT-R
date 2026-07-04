use ltx_attention::scaled_dot_product_attention;
use tch::{Device, Kind, Tensor};

#[test]
fn test_sdpa_basic_shapes() {
    // Simple attention: [B, H, T, D]
    let q = Tensor::randn([2, 4, 8, 32], (Kind::Float, Device::Cpu));
    let k = Tensor::randn([2, 4, 8, 32], (Kind::Float, Device::Cpu));
    let v = Tensor::randn([2, 4, 8, 32], (Kind::Float, Device::Cpu));
    let out = scaled_dot_product_attention(&q, &k, &v, None, false);
    assert_eq!(out.size(), vec![2, 4, 8, 32]);
}

#[test]
fn test_sdpa_with_mask() {
    let q = Tensor::randn([1, 2, 4, 16], (Kind::Float, Device::Cpu));
    let k = Tensor::randn([1, 2, 6, 16], (Kind::Float, Device::Cpu));
    let v = Tensor::randn([1, 2, 6, 16], (Kind::Float, Device::Cpu));
    let mask = Tensor::ones([4, 6], (Kind::Float, Device::Cpu)) * f64::NEG_INFINITY;
    let out = scaled_dot_product_attention(&q, &k, &v, Some(&mask), false);
    assert_eq!(out.size(), vec![1, 2, 4, 16]);
}

#[test]
fn test_sdpa_causal() {
    let q = Tensor::randn([1, 2, 8, 16], (Kind::Float, Device::Cpu));
    let k = Tensor::randn([1, 2, 8, 16], (Kind::Float, Device::Cpu));
    let v = Tensor::randn([1, 2, 8, 16], (Kind::Float, Device::Cpu));
    let out = scaled_dot_product_attention(&q, &k, &v, None, true);
    assert_eq!(out.size(), vec![1, 2, 8, 16]);
}

#[test]
fn test_sdpa_cross_attention() {
    let q = Tensor::randn([1, 2, 4, 16], (Kind::Float, Device::Cpu));
    let k = Tensor::randn([1, 2, 10, 16], (Kind::Float, Device::Cpu));
    let v = Tensor::randn([1, 2, 10, 16], (Kind::Float, Device::Cpu));
    let out = scaled_dot_product_attention(&q, &k, &v, None, false);
    assert_eq!(out.size(), vec![1, 2, 4, 16]);
}
