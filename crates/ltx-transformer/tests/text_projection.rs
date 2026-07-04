use ltx_transformer::text_projection::PixArtAlphaTextProjection;
use tch::{Device, Kind, Tensor};

fn make_vs() -> tch::nn::VarStore {
    tch::nn::VarStore::new(Device::Cpu)
}

#[test]
fn test_text_projection_output_shape() {
    let vs = make_vs();
    let proj = PixArtAlphaTextProjection::new(&vs.root(), 256, 64);
    let x = Tensor::randn([1, 10, 256], (Kind::Float, Device::Cpu));
    let out = proj.forward(&x);
    assert_eq!(out.size(), vec![1, 10, 64]);
}

#[test]
fn test_text_projection_matches_hidden_dim() {
    let vs = make_vs();
    let hidden_dim = 128;
    let proj = PixArtAlphaTextProjection::new(&vs.root(), 64, hidden_dim);
    let x = Tensor::randn([2, 8, 64], (Kind::Float, Device::Cpu));
    let out = proj.forward(&x);
    assert_eq!(out.size()[2], hidden_dim);
}

#[test]
fn test_text_projection_not_all_zero() {
    let vs = make_vs();
    let proj = PixArtAlphaTextProjection::new(&vs.root(), 32, 16);
    let x = Tensor::randn([1, 4, 32], (Kind::Float, Device::Cpu));
    let out = proj.forward(&x);
    // Output should have non-zero values (projection is not identity)
    assert!(out.abs().sum(tch::Kind::Float).double_value(&[]) > 0.0);
}
