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

// ── Numerical sanity tests ──────────────────────────────────────────

/// FeedForward output should be finite (no NaN/Inf from GELU).
#[test]
fn test_feed_forward_output_finite() {
    let vs = make_vs();
    let ff = FeedForward::new(&vs.root(), 64);
    let x = Tensor::randn([2, 8, 64], (Kind::Float, Device::Cpu));
    let out = ff.forward(&x);
    assert!(out.isfinite().all().double_value(&[]) > 0.0, "FeedForward produced NaN/Inf");
}

/// FeedForward with all-zero input should not produce NaN.
#[test]
fn test_feed_forward_zero_input() {
    let vs = make_vs();
    let ff = FeedForward::new(&vs.root(), 64);
    let x = Tensor::zeros([1, 4, 64], (Kind::Float, Device::Cpu));
    let out = ff.forward(&x);
    assert_eq!(out.isnan().any().double_value(&[]), 0.0, "FeedForward NaN on zero input");
}

/// FeedForward with extreme values should not overflow.
#[test]
fn test_feed_forward_extreme_values() {
    let vs = make_vs();
    let ff = FeedForward::new(&vs.root(), 64);
    let mut x = Tensor::randn([1, 4, 64], (Kind::Float, Device::Cpu));
    // Set a few elements to extreme values
    x = x.fill_(0.0);
    let _ = x.narrow(2, 0, 1).fill_(100.0);
    let _ = x.narrow(2, 1, 1).fill_(-100.0);
    let out = ff.forward(&x);
    assert!(out.isfinite().all().double_value(&[]) > 0.0, "FeedForward overflow on extreme values");
}
