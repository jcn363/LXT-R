use ltx_resblock::ResBlock1;
use ltx_types::NormLayerType;
use tch::nn::ModuleT;
use tch::{Device, Kind, Tensor};

fn make_vs() -> tch::nn::VarStore {
    tch::nn::VarStore::new(Device::Cpu)
}

#[test]
fn test_resblock_1d_preserves_shape() {
    let vs = make_vs();
    let block = ResBlock1::new(vs.root(), 8, 3, NormLayerType::Group, 4, 0.2);
    let x = Tensor::randn([2, 8, 32], (Kind::Float, Device::Cpu));
    let out = block.forward_t(&x, false);
    assert_eq!(out.size(), vec![2, 8, 32]);
}

#[test]
fn test_resblock_1d_silu() {
    let vs = make_vs();
    let block = ResBlock1::new(vs.root(), 4, 5, NormLayerType::Pixel, 0, 0.0);
    let x = Tensor::randn([1, 4, 16], (Kind::Float, Device::Cpu));
    let out = block.forward_t(&x, false);
    assert_eq!(out.size(), vec![1, 4, 16]);
}

#[test]
fn test_resblock_1d_kernel_1() {
    let vs = make_vs();
    let block = ResBlock1::new(vs.root(), 8, 1, NormLayerType::Group, 2, 0.1);
    let x = Tensor::randn([1, 8, 16], (Kind::Float, Device::Cpu));
    let out = block.forward_t(&x, false);
    assert_eq!(out.size(), vec![1, 8, 16]);
}
