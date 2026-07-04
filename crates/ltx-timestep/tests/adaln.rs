use ltx_timestep::AdaLayerNormSingle;
use tch::{Device, Kind, Tensor};

fn make_vs() -> tch::nn::VarStore {
    tch::nn::VarStore::new(Device::Cpu)
}

#[test]
fn test_adaln_shape() {
    let vs = make_vs();
    let adaln = AdaLayerNormSingle::new(&vs.root(), 64);
    let t = Tensor::from_slice(&[1i64]);
    let (output, emb) = adaln.forward(&t, Kind::Float);
    // output: dim * 6 = 384, emb: dim = 64
    assert_eq!(output.size(), vec![1, 384]);
    assert_eq!(emb.size(), vec![1, 64]);
}

#[test]
fn test_adaln_batch() {
    let vs = make_vs();
    let adaln = AdaLayerNormSingle::new(&vs.root(), 128);
    let t = Tensor::arange(4, (Kind::Int64, Device::Cpu));
    let (output, emb) = adaln.forward(&t, Kind::Float);
    assert_eq!(output.size(), vec![4, 768]);
    assert_eq!(emb.size(), vec![4, 128]);
}
