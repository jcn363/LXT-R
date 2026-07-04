use ltx_resblock::UNetMidBlock3D;
use ltx_types::NormLayerType;
use tch::nn::ModuleT;
use tch::{Device, Kind, Tensor};

fn make_vs() -> tch::nn::VarStore {
    tch::nn::VarStore::new(Device::Cpu)
}

#[test]
fn test_unet_mid_3d_preserves_shape() {
    let vs = make_vs();
    let block = UNetMidBlock3D::new(vs.root(), 8, NormLayerType::Group, 4, false, 2);
    let x = Tensor::randn([1, 8, 4, 8, 8], (Kind::Float, Device::Cpu));
    let out = block.forward_t(&x, false);
    assert_eq!(out.size(), vec![1, 8, 4, 8, 8]);
}

#[test]
fn test_unet_mid_3d_small() {
    let vs = make_vs();
    let block = UNetMidBlock3D::new(vs.root(), 4, NormLayerType::Group, 2, false, 2);
    let x = Tensor::randn([1, 4, 2, 4, 4], (Kind::Float, Device::Cpu));
    let out = block.forward_t(&x, false);
    assert_eq!(out.size(), vec![1, 4, 2, 4, 4]);
}
