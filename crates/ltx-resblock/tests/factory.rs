use ltx_resblock::make_resblock;
use ltx_types::NormLayerType;
use tch::{Device, Kind, Tensor};

fn make_vs() -> tch::nn::VarStore {
    tch::nn::VarStore::new(Device::Cpu)
}

#[test]
fn test_make_resblock_3d() {
    let vs = make_vs();
    let block = make_resblock(3, 8, 8, NormLayerType::Group, 4, false, vs.root());
    let x = Tensor::randn([1, 8, 4, 8, 8], (Kind::Float, Device::Cpu));
    let out = block.forward_t(&x, false);
    assert_eq!(out.size(), vec![1, 8, 4, 8, 8]);
}

#[test]
fn test_make_resblock_2d() {
    let vs = make_vs();
    let block = make_resblock(2, 4, 8, NormLayerType::Pixel, 0, false, vs.root());
    let x = Tensor::randn([1, 4, 8, 8], (Kind::Float, Device::Cpu));
    let out = block.forward_t(&x, false);
    assert_eq!(out.size(), vec![1, 8, 8, 8]);
}
