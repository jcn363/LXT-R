use ltx_timestep::get_timestep_embedding;
use tch::{Device, Kind, Tensor};

#[test]
fn test_get_timestep_embedding_shape() {
    let t = Tensor::arange(4, (Kind::Int64, Device::Cpu));
    let emb = get_timestep_embedding(&t, 64, 10_000);
    assert_eq!(emb.size(), vec![4, 64]);
}

#[test]
fn test_get_timestep_embedding_single() {
    let t = Tensor::from_slice(&[0i64]);
    let emb = get_timestep_embedding(&t, 16, 10_000);
    assert_eq!(emb.size(), vec![1, 16]);
}

#[test]
fn test_get_timestep_embedding_batch() {
    let t = Tensor::arange(8, (Kind::Int64, Device::Cpu));
    let emb = get_timestep_embedding(&t, 128, 10_000);
    assert_eq!(emb.size(), vec![8, 128]);
}

#[test]
fn test_get_timestep_embedding_even_dim() {
    let t = Tensor::from_slice(&[1i64]);
    let emb = get_timestep_embedding(&t, 32, 10_000);
    assert_eq!(emb.size()[1], 32);
}

/// Regression: `(-max_period).ln()` produced NaN. Must use `-(max_period).ln()`.
/// All output values must be finite (no NaN/Inf).
#[test]
fn test_get_timestep_embedding_no_nan() {
    for &t_val in &[0i64, 1, 10, 100, 999] {
        let t = Tensor::from_slice(&[t_val]);
        let emb = get_timestep_embedding(&t, 64, 10_000);
        assert_eq!(
            emb.isnan().any().double_value(&[]),
            0.0,
            "NaN at timestep={t_val}"
        );
        assert_eq!(
            emb.isinf().any().double_value(&[]),
            0.0,
            "Inf at timestep={t_val}"
        );
    }
}

/// Verify that different timesteps produce different embeddings.
#[test]
fn test_get_timestep_embedding_varies() {
    let t0 = get_timestep_embedding(&Tensor::from_slice(&[0i64]), 32, 10_000);
    let t1 = get_timestep_embedding(&Tensor::from_slice(&[1i64]), 32, 10_000);
    let max_diff = (&t0 - &t1).abs().max().double_value(&[]);
    assert!(max_diff > 0.0, "timesteps 0 and 1 produced identical embeddings");
}
