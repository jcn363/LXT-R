use ltx_norm::RMSNorm;
use tch::{Device, Kind, Tensor};

#[test]
fn test_rms_norm_preserves_shape() {
    let x = Tensor::randn([2, 16, 64], (Kind::Float, Device::Cpu));
    let norm = RMSNorm::default_eps(64, Device::Cpu);
    let out = norm.forward(&x);
    assert_eq!(out.size(), vec![2, 16, 64]);
}

#[test]
fn test_rms_norm_3d_preserves_shape() {
    let x = Tensor::randn([4, 8, 32], (Kind::Float, Device::Cpu));
    let norm = RMSNorm::default_eps(32, Device::Cpu);
    let out = norm.forward(&x);
    assert_eq!(out.size(), vec![4, 8, 32]);
}

#[test]
fn test_rms_norm_keeps_dtype() {
    let x = Tensor::ones([1, 4], (Kind::Float, Device::Cpu));
    let norm = RMSNorm::default_eps(4, Device::Cpu);
    let out = norm.forward(&x);
    assert_eq!(out.kind(), Kind::Float);
}

#[test]
fn test_rms_norm_custom_eps() {
    let eps = 1e-4;
    let x = Tensor::zeros([1, 4], (Kind::Float, Device::Cpu));
    let norm = RMSNorm::new(4, eps, Device::Cpu);
    let out = norm.forward(&x);
    assert_eq!(out.size(), vec![1, 4]);
}
