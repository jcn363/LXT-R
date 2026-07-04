use ltx_resblock::ResnetBlock3D;
use ltx_types::NormLayerType;
use tch::nn::ModuleT;
use tch::{Device, Kind, Tensor};

fn make_vs() -> tch::nn::VarStore {
    tch::nn::VarStore::new(Device::Cpu)
}

#[test]
fn test_resblock_3d_same_channels() {
    let vs = make_vs();
    let block = ResnetBlock3D::new(vs.root(), 8, 8, NormLayerType::Group, 4, false);
    let x = Tensor::randn([1, 8, 4, 8, 8], (Kind::Float, Device::Cpu));
    let out = block.forward_t(&x, false);
    assert_eq!(out.size(), vec![1, 8, 4, 8, 8]);
}

#[test]
fn test_resblock_3d_diff_channels() {
    let vs = make_vs();
    let block = ResnetBlock3D::new(vs.root(), 8, 16, NormLayerType::Group, 4, false);
    let x = Tensor::randn([1, 8, 4, 8, 8], (Kind::Float, Device::Cpu));
    let out = block.forward_t(&x, false);
    assert_eq!(out.size(), vec![1, 16, 4, 8, 8]);
}

#[test]
fn test_resblock_3d_pixel_norm() {
    let vs = make_vs();
    let block = ResnetBlock3D::new(vs.root(), 4, 4, NormLayerType::Pixel, 0, false);
    let x = Tensor::randn([1, 4, 4, 8, 8], (Kind::Float, Device::Cpu));
    let out = block.forward_t(&x, false);
    assert_eq!(out.size(), vec![1, 4, 4, 8, 8]);
}
