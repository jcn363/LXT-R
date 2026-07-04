use ltx_conv::{CausalConv2d, CausalityAxis};
use tch::nn::ModuleT;
use tch::{Device, Kind, Tensor};

fn make_vs() -> tch::nn::VarStore {
    tch::nn::VarStore::new(Device::Cpu)
}

#[test]
fn test_causal_conv2d_preserves_shape() {
    let vs = make_vs();
    let conv = CausalConv2d::new(vs.root(), 16, 32, 3, 1, CausalityAxis::Time);
    let x = Tensor::randn([2, 16, 16, 16], (Kind::Float, Device::Cpu));
    let out = conv.forward_t(&x, false);
    assert_eq!(out.size(), vec![2, 32, 16, 16]);
}

#[test]
fn test_causal_conv2d_small() {
    let vs = make_vs();
    let conv = CausalConv2d::new(vs.root(), 4, 8, 3, 1, CausalityAxis::Time);
    let x = Tensor::randn([1, 4, 8, 8], (Kind::Float, Device::Cpu));
    let out = conv.forward_t(&x, false);
    assert_eq!(out.size(), vec![1, 8, 8, 8]);
}

#[test]
fn test_causal_conv2d_with_stride() {
    let vs = make_vs();
    let conv = CausalConv2d::new(vs.root(), 8, 16, 3, 2, CausalityAxis::Width);
    let x = Tensor::randn([1, 8, 16, 16], (Kind::Float, Device::Cpu));
    let out = conv.forward_t(&x, false);
    assert_eq!(out.size(), vec![1, 16, 8, 8]);
}

#[test]
fn test_causal_conv2d_none_axis() {
    let vs = make_vs();
    let conv = CausalConv2d::new(vs.root(), 3, 6, 3, 1, CausalityAxis::None);
    let x = Tensor::randn([1, 3, 8, 8], (Kind::Float, Device::Cpu));
    let out = conv.forward_t(&x, false);
    // No padding is applied with CausalityAxis::None when causal=true,
    // so spatial dims shrink by (kernel_size - 1).
    assert_eq!(out.size(), vec![1, 6, 6, 6]);
}
