use ltx_conv::{make_conv_nd, make_causal_conv2d, CausalityAxis};
use tch::nn::ModuleT;
use tch::{Device, Kind, Tensor};

fn make_vs() -> tch::nn::VarStore {
    tch::nn::VarStore::new(Device::Cpu)
}

#[test]
fn test_make_conv_nd_2d() {
    let vs = make_vs();
    // kernel=3 needs padding=1 to preserve spatial shape
    let conv = make_conv_nd(vs.root(), 2, 8, 16, 3, 1, 1, false, "zeros");
    let x = Tensor::randn([2, 8, 16, 16], (Kind::Float, Device::Cpu));
    let out = conv.forward_t(&x, false);
    assert_eq!(out.size(), vec![2, 16, 16, 16]);
}

#[test]
fn test_make_conv_nd_3d_causal() {
    let vs = make_vs();
    let conv = make_conv_nd(vs.root(), 3, 4, 8, 3, 1, 0, true, "zeros");
    let x = Tensor::randn([1, 4, 4, 8, 8], (Kind::Float, Device::Cpu));
    let out = conv.forward_t(&x, false);
    assert_eq!(out.size(), vec![1, 8, 4, 8, 8]);
}

#[test]
fn test_make_causal_conv2d() {
    let vs = make_vs();
    let conv = make_causal_conv2d(vs.root(), 4, 8, 3, 1, CausalityAxis::Time);
    let x = Tensor::randn([1, 4, 8, 8], (Kind::Float, Device::Cpu));
    let out = conv.forward_t(&x, false);
    assert_eq!(out.size(), vec![1, 8, 8, 8]);
}
