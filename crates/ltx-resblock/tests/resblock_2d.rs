use ltx_resblock::ResnetBlock2D;
use ltx_types::NormLayerType;
use tch::nn::ModuleT;
use tch::{Device, Kind, Tensor};

fn make_vs() -> tch::nn::VarStore {
    tch::nn::VarStore::new(Device::Cpu)
}

#[test]
fn test_resblock_2d_same_channels() {
    let vs = make_vs();
    let block = ResnetBlock2D::new(vs.root(), 8, 8, NormLayerType::Group, 4, false);
    let x = Tensor::randn([2, 8, 16, 16], (Kind::Float, Device::Cpu));
    let out = block.forward_t(&x, false);
    assert_eq!(out.size(), vec![2, 8, 16, 16]);
}

#[test]
fn test_resblock_2d_diff_channels() {
    let vs = make_vs();
    let block = ResnetBlock2D::new(vs.root(), 8, 16, NormLayerType::Group, 4, false);
    let x = Tensor::randn([1, 8, 8, 8], (Kind::Float, Device::Cpu));
    let out = block.forward_t(&x, false);
    assert_eq!(out.size(), vec![1, 16, 8, 8]);
}

#[test]
fn test_resblock_2d_pixel_norm() {
    let vs = make_vs();
    let block = ResnetBlock2D::new(vs.root(), 4, 4, NormLayerType::Pixel, 0, false);
    let x = Tensor::randn([1, 4, 8, 8], (Kind::Float, Device::Cpu));
    let out = block.forward_t(&x, false);
    assert_eq!(out.size(), vec![1, 4, 8, 8]);
}
