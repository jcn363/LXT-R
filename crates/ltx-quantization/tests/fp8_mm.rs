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
