use ltx_types::utils::{projection_coef, to_denoised, to_velocity};
use tch::{Device, Kind, Tensor};

/// to_velocity with sigma=0 should use epsilon floor, not panic or produce NaN.
#[test]
fn test_to_velocity_sigma_zero() {
    let sample = Tensor::ones([2, 4], (Kind::Float, Device::Cpu));
    let denoised = Tensor::zeros([2, 4], (Kind::Float, Device::Cpu));
    let vel = to_velocity(&sample, 0.0, &denoised, Kind::Float);
    assert_eq!(
        vel.isnan().any().double_value(&[]),
        0.0,
        "to_velocity produced NaN at sigma=0"
    );
    assert_eq!(
        vel.isinf().any().double_value(&[]),
        0.0,
        "to_velocity produced Inf at sigma=0"
    );
}

/// to_velocity with very small sigma should not overflow.
#[test]
fn test_to_velocity_small_sigma() {
    let sample = Tensor::ones([2, 4], (Kind::Float, Device::Cpu));
    let denoised = Tensor::zeros([2, 4], (Kind::Float, Device::Cpu));
    let vel = to_velocity(&sample, 1e-10, &denoised, Kind::Float);
    assert!(
        vel.isfinite().all().double_value(&[]) > 0.0,
        "to_velocity overflow at small sigma"
    );
}

/// to_denoised roundtrip: sample → velocity → denoised should recover.
#[test]
fn test_to_denoised_roundtrip() {
    let sample = Tensor::randn([2, 8], (Kind::Float, Device::Cpu));
    let denoised = Tensor::randn([2, 8], (Kind::Float, Device::Cpu));
    let sigma = 0.5;
    let vel = to_velocity(&sample, sigma, &denoised, Kind::Float);
    let recovered = to_denoised(&sample, &vel, sigma, Kind::Float);
    let max_diff = (&recovered - &denoised).abs().max().double_value(&[]);
    assert!(
        max_diff < 1e-4,
        "to_denoised roundtrip: max diff = {max_diff}"
    );
}

/// projection_coef should produce finite output for normal inputs.
#[test]
fn test_projection_coef_finite() {
    let a = Tensor::randn([4, 8, 8], (Kind::Float, Device::Cpu));
    let b = Tensor::randn([4, 8, 8], (Kind::Float, Device::Cpu));
    let coef = projection_coef(&a, &b);
    assert!(
        coef.isfinite().all().double_value(&[]) > 0.0,
        "projection_coef produced NaN/Inf"
    );
    // Output should broadcast: shape [4, 1, 1]
    assert_eq!(coef.size(), vec![4, 1, 1]);
}
