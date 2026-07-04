use ltx_norm::GroupNorm;
use tch::{Device, Kind, Tensor};

#[test]
fn test_group_norm_preserves_shape() {
    let x = Tensor::randn([2, 32, 8, 8], (Kind::Float, Device::Cpu));
    let norm = GroupNorm::with_defaults(8, 32);
    let out = norm.forward(&x);
    assert_eq!(out.size(), vec![2, 32, 8, 8]);
}

#[test]
fn test_group_norm_3d_preserves_shape() {
    let x = Tensor::randn([1, 16, 4, 8, 8], (Kind::Float, Device::Cpu));
    let norm = GroupNorm::with_defaults(4, 16);
    let out = norm.forward(&x);
    assert_eq!(out.size(), vec![1, 16, 4, 8, 8]);
}

#[test]
fn test_group_norm_no_affine() {
    let norm = GroupNorm::new(4, 16, 1e-5, false);
    let x = Tensor::randn([2, 16, 8, 8], (Kind::Float, Device::Cpu));
    let out = norm.forward(&x);
    assert_eq!(out.size(), vec![2, 16, 8, 8]);
}

#[test]
fn test_group_norm_custom_eps() {
    let norm = GroupNorm::new(2, 8, 1e-4, true);
    let x = Tensor::ones([1, 8, 4, 4], (Kind::Float, Device::Cpu));
    let out = norm.forward(&x);
    assert_eq!(out.size(), vec![1, 8, 4, 4]);
}
