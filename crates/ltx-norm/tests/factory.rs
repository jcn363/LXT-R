use ltx_norm::build_norm_layer;
use ltx_types::NormLayerType;
use tch::{Device, Kind, Tensor};

#[test]
fn test_build_norm_layer_group() {
    let norm = build_norm_layer(NormLayerType::Group, 32, 8);
    let x = Tensor::randn([1, 32, 8, 8], (Kind::Float, Device::Cpu));
    let out = norm.forward_t(&x, false);
    assert_eq!(out.size(), vec![1, 32, 8, 8]);
}

#[test]
fn test_build_norm_layer_pixel() {
    let norm = build_norm_layer(NormLayerType::Pixel, 16, 0);
    let x = Tensor::randn([2, 16, 16, 16], (Kind::Float, Device::Cpu));
    let out = norm.forward_t(&x, false);
    assert_eq!(out.size(), vec![2, 16, 16, 16]);
}

#[test]
fn test_build_norm_layer_group_3d() {
    let norm = build_norm_layer(NormLayerType::Group, 16, 4);
    let x = Tensor::randn([1, 16, 4, 8, 8], (Kind::Float, Device::Cpu));
    let out = norm.forward_t(&x, false);
    assert_eq!(out.size(), vec![1, 16, 4, 8, 8]);
}
