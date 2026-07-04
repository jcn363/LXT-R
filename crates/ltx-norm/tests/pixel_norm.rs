use ltx_norm::PixelNorm;
use tch::{Device, Kind, Tensor};

#[test]
fn test_pixel_norm_4d_preserves_shape() {
    let x = Tensor::randn([2, 16, 32, 32], (Kind::Float, Device::Cpu));
    let norm = PixelNorm::default();
    let out = norm.forward(&x);
    assert_eq!(out.size(), vec![2, 16, 32, 32]);
}

#[test]
fn test_pixel_norm_5d_preserves_shape() {
    let x = Tensor::randn([1, 4, 8, 16, 16], (Kind::Float, Device::Cpu));
    let norm = PixelNorm::default();
    let out = norm.forward(&x);
    assert_eq!(out.size(), vec![1, 4, 8, 16, 16]);
}

#[test]
fn test_pixel_norm_default_uses_norm_eps() {
    let norm = PixelNorm::default();
    let x = Tensor::zeros([1, 3, 8, 8], (Kind::Float, Device::Cpu));
    let out = norm.forward(&x);
    assert_eq!(out.size(), vec![1, 3, 8, 8]);
}

#[test]
fn test_pixel_norm_custom_eps() {
    let norm = PixelNorm::new(1e-3);
    let x = Tensor::randn([1, 3, 8, 8], (Kind::Float, Device::Cpu));
    let out = norm.forward(&x);
    assert_eq!(out.size(), vec![1, 3, 8, 8]);
}
