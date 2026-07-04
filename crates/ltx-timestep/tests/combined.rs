use ltx_timestep::CombinedTimestepSizeEmbeddings;
use tch::{Device, Kind, Tensor};

fn make_vs() -> tch::nn::VarStore {
    tch::nn::VarStore::new(Device::Cpu)
}

#[test]
fn test_combined_timestep_shape() {
    let vs = make_vs();
    let emb = CombinedTimestepSizeEmbeddings::new(&vs.root(), 64);
    let t = Tensor::from_slice(&[1i64]);
    let out = emb.forward(&t, Kind::Float);
    assert_eq!(out.size(), vec![1, 64]);
}

#[test]
fn test_combined_timestep_batch() {
    let vs = make_vs();
    let emb = CombinedTimestepSizeEmbeddings::new(&vs.root(), 128);
    let t = Tensor::arange(4, (Kind::Int64, Device::Cpu));
    let out = emb.forward(&t, Kind::Float);
    assert_eq!(out.size(), vec![4, 128]);
}
