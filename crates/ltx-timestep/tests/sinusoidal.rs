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
