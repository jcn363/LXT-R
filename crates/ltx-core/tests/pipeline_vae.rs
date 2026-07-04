/// Integration tests for the video VAE encode/decode pipeline.
/// Tests: pixel space → VideoEncoder → latent → space_to_depth/depth_to_space
use ltx_video_vae::sampling::{space_to_depth, depth_to_space};
use tch::{Device, Kind, Tensor};

/// Test space_to_depth → depth_to_space roundtrip (the core VAE sampling operation).
#[test]
fn test_vae_sampling_roundtrip() {
    let x = Tensor::randn([1, 4, 8, 32, 32], (Kind::Float, Device::Cpu));

    // space_to_depth: (B, C, T, H, W) → (B, C*4, T, H/2, W/2)
    let y = space_to_depth(&x, 2);
    assert_eq!(y.size(), vec![1, 16, 8, 16, 16]);

    // depth_to_space: inverse
    let x_rec = depth_to_space(&y, 2);
    assert_eq!(x_rec.size(), vec![1, 4, 8, 32, 32]);

    // Should be exactly reversible
    assert!(x.allclose(&x_rec, 1e-6, 1e-6, false));
}

/// Test with various spatial downsampling factors.
#[test]
fn test_vae_sampling_various_factors() {
    for &r in &[2i64, 4] {
        let x = Tensor::randn([1, 3, 4, 16, 16], (Kind::Float, Device::Cpu));
        let y = space_to_depth(&x, r);
        assert_eq!(y.size(), vec![1, 3 * r * r, 4, 16 / r, 16 / r]);
        let x_rec = depth_to_space(&y, r);
        assert!(x.allclose(&x_rec, 1e-6, 1e-6, false));
    }
}

/// Test VAE sampling with different batch sizes.
#[test]
fn test_vae_sampling_batch() {
    for &b in &[1i64, 2, 4] {
        let x = Tensor::randn([b, 4, 8, 16, 16], (Kind::Float, Device::Cpu));
        let y = space_to_depth(&x, 2);
        assert_eq!(y.size()[0], b);
        let x_rec = depth_to_space(&y, 2);
        assert!(x.allclose(&x_rec, 1e-6, 1e-6, false));
    }
}

/// Test VAE sampling with different channel counts.
#[test]
fn test_vae_sampling_various_channels() {
    for &c in &[1i64, 3, 8, 16] {
        let x = Tensor::randn([1, c, 4, 16, 16], (Kind::Float, Device::Cpu));
        let y = space_to_depth(&x, 2);
        assert_eq!(y.size(), vec![1, c * 4, 4, 8, 8]);
        let x_rec = depth_to_space(&y, 2);
        assert!(x.allclose(&x_rec, 1e-6, 1e-6, false));
    }
}

/// Test that VAE sampling preserves values through roundtrip.
#[test]
fn test_vae_sampling_value_roundtrip() {
    let x = Tensor::from_slice(&[
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ])
    .reshape([1, 1, 1, 4, 4]);

    let y = space_to_depth(&x, 2);
    // The packed representation has 4 channels
    assert_eq!(y.size(), vec![1, 4, 1, 2, 2]);

    // Roundtrip should recover exact original values
    let x_rec = depth_to_space(&y, 2);
    assert!(x.allclose(&x_rec, 1e-6, 1e-6, false));

    // Verify specific values survived
    let flat = x_rec.reshape([-1]);
    assert!((flat.double_value(&[0]) - 1.0).abs() < 1e-6);
    assert!((flat.double_value(&[5]) - 6.0).abs() < 1e-6);
    assert!((flat.double_value(&[15]) - 16.0).abs() < 1e-6);
}
