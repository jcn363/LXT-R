use ltx_resblock::ResnetBlock2D;
use tch::nn::ModuleT;
use tch::{Device, Kind, Tensor};

fn make_vs() -> tch::nn::VarStore {
    tch::nn::VarStore::new(Device::Cpu)
}

#[test]
fn test_resblock_2d_same_channels() {
    let vs = make_vs();
    let block = ResnetBlock2D::new(vs.root(), 8, 8);
    let x = Tensor::randn([2, 8, 16, 16], (Kind::Float, Device::Cpu));
    let out = block.forward_t(&x, false);
    assert_eq!(out.size(), vec![2, 8, 16, 16]);
}

#[test]
fn test_resblock_2d_diff_channels() {
    let vs = make_vs();
    let block = ResnetBlock2D::new(vs.root(), 8, 16);
    let x = Tensor::randn([1, 8, 8, 8], (Kind::Float, Device::Cpu));
    let out = block.forward_t(&x, false);
    assert_eq!(out.size(), vec![1, 16, 8, 8]);
}

#[test]
fn test_resblock_2d_residual_connection() {
    let vs = make_vs();
    let block = ResnetBlock2D::new(vs.root(), 8, 8);
    let x = Tensor::randn([1, 8, 8, 8], (Kind::Float, Device::Cpu));
    let out = block.forward_t(&x, false);
    // Output shape must match input shape (residual connection)
    assert_eq!(out.size(), x.size());
}
