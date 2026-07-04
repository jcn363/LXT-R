use ltx_timestep::TimestepEmbedding;
use tch::{Device, Kind, Tensor};

fn make_vs() -> tch::nn::VarStore {
    tch::nn::VarStore::new(Device::Cpu)
}

#[test]
fn test_timestep_embedding_preserves_shape() {
    let vs = make_vs();
    let emb = TimestepEmbedding::new(&vs.root(), 64);
    let x = Tensor::randn([2, 64], (Kind::Float, Device::Cpu));
    let out = emb.forward(&x);
    assert_eq!(out.size(), vec![2, 64]);
}

#[test]
fn test_timestep_embedding_batch() {
    let vs = make_vs();
    let emb = TimestepEmbedding::new(&vs.root(), 128);
    let x = Tensor::randn([4, 128], (Kind::Float, Device::Cpu));
    let out = emb.forward(&x);
    assert_eq!(out.size(), vec![4, 128]);
}

#[test]
fn test_timestep_embedding_single() {
    let vs = make_vs();
    let emb = TimestepEmbedding::new(&vs.root(), 32);
    let x = Tensor::randn([1, 32], (Kind::Float, Device::Cpu));
    let out = emb.forward(&x);
    assert_eq!(out.size(), vec![1, 32]);
}
