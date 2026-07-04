use ltx_transformer::feed_forward::FeedForward;
use tch::{Device, Kind, Tensor};

fn make_vs() -> tch::nn::VarStore {
    tch::nn::VarStore::new(Device::Cpu)
}

#[test]
fn test_feed_forward_output_shape() {
    let vs = make_vs();
    let ff = FeedForward::new(&vs.root(), 64);
    let x = Tensor::randn([2, 10, 64], (Kind::Float, Device::Cpu));
    let out = ff.forward(&x);
    assert_eq!(out.size(), vec![2, 10, 64]);
}

#[test]
fn test_feed_forward_different_dims() {
    let vs = make_vs();
    let ff = FeedForward::new(&vs.root(), 128);
    let x = Tensor::randn([1, 5, 128], (Kind::Float, Device::Cpu));
    let out = ff.forward(&x);
    assert_eq!(out.size(), vec![1, 5, 128]);
}

#[test]
fn test_feed_forward_preserves_batch() {
    let vs = make_vs();
    let ff = FeedForward::new(&vs.root(), 32);
    for batch in [1, 4, 8] {
        let x = Tensor::randn([batch, 3, 32], (Kind::Float, Device::Cpu));
        let out = ff.forward(&x);
        assert_eq!(out.size()[0], batch);
    }
}
