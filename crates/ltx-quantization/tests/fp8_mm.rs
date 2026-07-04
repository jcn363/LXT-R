use ltx_quantization::FP8Linear;
use tch::nn::Module;
use tch::{Device, Kind, Tensor};

#[test]
fn test_fp8_linear_output_shape() {
    let weight = Tensor::randn([16, 32], (Kind::Float, Device::Cpu));
    let linear = FP8Linear::new(weight, None);
    let x = Tensor::randn([1, 32], (Kind::Float, Device::Cpu));
    let out = linear.forward(&x);
    assert_eq!(out.size(), vec![1, 16]);
}

#[test]
fn test_fp8_linear_with_bias() {
    let weight = Tensor::randn([8, 16], (Kind::Float, Device::Cpu));
    let bias = Tensor::zeros([8], (Kind::Float, Device::Cpu));
    let linear = FP8Linear::new(weight, Some(bias));
    let x = Tensor::randn([2, 16], (Kind::Float, Device::Cpu));
    let out = linear.forward(&x);
    assert_eq!(out.size(), vec![2, 8]);
}

#[test]
fn test_fp8_linear_in_out_features() {
    let weight = Tensor::randn([32, 64], (Kind::Float, Device::Cpu));
    let linear = FP8Linear::new(weight, None);
    assert_eq!(linear.in_features(), 64);
    assert_eq!(linear.out_features(), 32);
}

#[test]
fn test_fp8_linear_batch_preservation() {
    let weight = Tensor::randn([8, 16], (Kind::Float, Device::Cpu));
    let linear = FP8Linear::new(weight, None);
    for b in [1, 4, 8] {
        let x = Tensor::randn([b, 16], (Kind::Float, Device::Cpu));
        let out = linear.forward(&x);
        assert_eq!(out.size()[0], b);
    }
}

#[test]
fn test_fp8_linear_debug() {
    let weight = Tensor::randn([4, 8], (Kind::Float, Device::Cpu));
    let linear = FP8Linear::new(weight, None);
    let debug_str = format!("{:?}", linear);
    assert!(debug_str.contains("FP8Linear"));
}

// ── Bug verification: numerical correctness tests ──────────────────────

/// Bug verification: FP8Linear with non-square weight must transpose correctly.
/// Before the fix, `x.matmul(W)` with W=[out,in] would fail for non-square weights.
#[test]
fn test_fp8_linear_non_square_weight() {
    let weight = Tensor::randn([8, 32], (Kind::Float, Device::Cpu));
    let linear = FP8Linear::new(weight, None);
    let x = Tensor::randn([1, 32], (Kind::Float, Device::Cpu));
    let out = linear.forward(&x);
    assert_eq!(out.size(), vec![1, 8]);
    assert!(out.abs().sum(Kind::Float).double_value(&[]) > 0.0);
}

/// Bug verification: FP8Linear output should match reference `x @ W.t()` within
/// quantization tolerance. The FP8 quantization introduces ~0.5-1.0 max error.
#[test]
fn test_fp8_linear_matches_reference() {
    let out_features = 8;
    let in_features = 16;
    let weight = Tensor::randn([out_features, in_features], (Kind::Float, Device::Cpu));
    let linear = FP8Linear::new(weight.shallow_clone(), None);

    let x = Tensor::randn([2, in_features], (Kind::Float, Device::Cpu));

    // Reference: manually compute x @ W.t() (what nn.Linear does)
    let expected = x.matmul(&weight.transpose(0, 1));

    let actual = linear.forward(&x);

    // FP8 quantization introduces error — allow up to 2.0 absolute tolerance
    let max_diff = (&actual - &expected).abs().max().double_value(&[]);
    assert!(
        max_diff < 2.0,
        "FP8Linear numerical mismatch: max diff = {max_diff}, expected < 2.0"
    );
}

/// Bug verification: identity-like weight should preserve input (within FP8 tolerance).
#[test]
fn test_fp8_linear_identity_weight() {
    let dim = 8;
    // Identity matrix: weight[i,j] = 1.0 if i==j else 0.0
    let weight = Tensor::eye(dim, (Kind::Float, Device::Cpu));
    let linear = FP8Linear::new(weight, None);

    let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).reshape([1, dim]);
    let out = linear.forward(&x);

    // x @ I^T should ≈ x (within FP8 tolerance)
    let max_diff = (&out - &x).abs().max().double_value(&[]);
    assert!(
        max_diff < 1.0,
        "FP8Linear identity test: max diff = {max_diff}, expected < 1.0"
    );
}
