use ltx_conv::DualConv3d;
use tch::nn::ModuleT;
use tch::{Device, Kind, Tensor};

fn make_vs() -> (tch::nn::VarStore, tch::nn::VarStore) {
    let vs_s = tch::nn::VarStore::new(Device::Cpu);
    let vs_t = tch::nn::VarStore::new(Device::Cpu);
    (vs_s, vs_t)
}

#[test]
fn test_dual_conv3d_preserves_shape() {
    let (vs_s, vs_t) = make_vs();
    let conv = DualConv3d::new(vs_s.root(), vs_t.root(), 8, 16, 3, 1);
    let x = Tensor::randn([1, 8, 4, 8, 8], (Kind::Float, Device::Cpu));
    let out = conv.forward_t(&x, false);
    assert_eq!(out.size(), vec![1, 16, 4, 8, 8]);
}

#[test]
fn test_dual_conv3d_small() {
    let (vs_s, vs_t) = make_vs();
    let conv = DualConv3d::new(vs_s.root(), vs_t.root(), 4, 8, 3, 1);
    let x = Tensor::randn([1, 4, 2, 4, 4], (Kind::Float, Device::Cpu));
    let out = conv.forward_t(&x, false);
    assert_eq!(out.size(), vec![1, 8, 2, 4, 4]);
}

#[test]
fn test_dual_conv3d_same_channels() {
    let (vs_s, vs_t) = make_vs();
    let conv = DualConv3d::new(vs_s.root(), vs_t.root(), 4, 4, 3, 1);
    let x = Tensor::randn([1, 4, 4, 8, 8], (Kind::Float, Device::Cpu));
    let out = conv.forward_t(&x, false);
    assert_eq!(out.size(), vec![1, 4, 4, 8, 8]);
}

#[test]
fn test_dual_conv3d_stride() {
    let (vs_s, vs_t) = make_vs();
    let conv = DualConv3d::new(vs_s.root(), vs_t.root(), 4, 8, 3, 2);
    let x = Tensor::randn([1, 4, 4, 16, 16], (Kind::Float, Device::Cpu));
    let out = conv.forward_t(&x, false);
    assert_eq!(out.size(), vec![1, 8, 2, 8, 8]);
}
