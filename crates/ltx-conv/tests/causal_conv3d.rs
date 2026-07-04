use ltx_conv::CausalConv3d;
use tch::nn::ModuleT;
use tch::{Device, Kind, Tensor};

fn make_vs() -> tch::nn::VarStore {
    tch::nn::VarStore::new(Device::Cpu)
}

#[test]
fn test_causal_conv3d_preserves_shape() {
    let vs = make_vs();
    let conv = CausalConv3d::new(vs.root(), 8, 16, 3, 1);
    let x = Tensor::randn([1, 8, 4, 8, 8], (Kind::Float, Device::Cpu));
    let out = conv.forward_t(&x, false);
    assert_eq!(out.size(), vec![1, 16, 4, 8, 8]);
}

#[test]
fn test_causal_conv3d_small() {
    let vs = make_vs();
    let conv = CausalConv3d::new(vs.root(), 4, 8, 3, 1);
    let x = Tensor::randn([1, 4, 2, 4, 4], (Kind::Float, Device::Cpu));
    let out = conv.forward_t(&x, false);
    assert_eq!(out.size(), vec![1, 8, 2, 4, 4]);
}

#[test]
fn test_causal_conv3d_stride() {
    let vs = make_vs();
    let conv = CausalConv3d::new(vs.root(), 4, 8, 3, 2);
    let x = Tensor::randn([1, 4, 4, 16, 16], (Kind::Float, Device::Cpu));
    let out = conv.forward_t(&x, false);
    assert_eq!(out.size(), vec![1, 8, 2, 8, 8]);
}

#[test]
fn test_causal_conv3d_non_causal() {
    let vs = make_vs();
    let conv = CausalConv3d::new(vs.root(), 3, 6, 3, 1);
    let x = Tensor::randn([1, 3, 4, 8, 8], (Kind::Float, Device::Cpu));
    let out = conv.forward(&x, false);
    assert_eq!(out.size(), vec![1, 6, 4, 8, 8]);
}
