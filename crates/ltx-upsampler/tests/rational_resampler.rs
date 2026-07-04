use ltx_upsampler::SpatialRationalResampler;
use tch::nn::ModuleT;
use tch::{Device, Kind, Tensor};

fn make_vs() -> tch::nn::VarStore {
    tch::nn::VarStore::new(Device::Cpu)
}

#[test]
fn test_rational_resampler_upsample_3_2() {
    let vs = make_vs();
    let rs = SpatialRationalResampler::new(vs.root(), 8, 3, 2, 8);
    let x = Tensor::randn([1, 8, 2, 8, 8], (Kind::Float, Device::Cpu));
    let out = rs.forward(&x);
    // 8 * 3 / 2 = 12
    assert_eq!(out.size(), vec![1, 8, 2, 12, 12]);
}

#[test]
fn test_rational_resampler_downsample_2_3() {
    let vs = make_vs();
    let rs = SpatialRationalResampler::new(vs.root(), 8, 2, 3, 8);
    let x = Tensor::randn([1, 8, 2, 12, 12], (Kind::Float, Device::Cpu));
    let out = rs.forward(&x);
    // 12 * 2 / 3 = 8
    assert_eq!(out.size(), vec![1, 8, 2, 8, 8]);
}

#[test]
fn test_rational_resampler_preserves_time() {
    let vs = make_vs();
    let rs = SpatialRationalResampler::new(vs.root(), 4, 3, 2, 4);
    let t = 7;
    let x = Tensor::randn([1, 4, t, 8, 8], (Kind::Float, Device::Cpu));
    let out = rs.forward(&x);
    assert_eq!(out.size()[2], t);
}

#[test]
fn test_rational_resampler_preserves_batch() {
    let vs = make_vs();
    let rs = SpatialRationalResampler::new(vs.root(), 4, 3, 2, 4);
    let b = 3;
    let x = Tensor::randn([b, 4, 2, 8, 8], (Kind::Float, Device::Cpu));
    let out = rs.forward(&x);
    assert_eq!(out.size()[0], b);
}

#[test]
fn test_rational_resampler_module_t_trait() {
    let vs = make_vs();
    let rs = SpatialRationalResampler::new(vs.root(), 8, 3, 2, 8);
    let x = Tensor::randn([1, 8, 2, 8, 8], (Kind::Float, Device::Cpu));
    let out = rs.forward_t(&x, false);
    assert_eq!(out.size(), vec![1, 8, 2, 12, 12]);
}

#[test]
#[should_panic(expected = "num and den must be positive")]
fn test_rational_resampler_zero_num_panics() {
    let vs = make_vs();
    let _rs = SpatialRationalResampler::new(vs.root(), 8, 0, 2, 8);
}

#[test]
#[should_panic(expected = "num == den means no resampling needed")]
fn test_rational_resampler_equal_panics() {
    let vs = make_vs();
    let _rs = SpatialRationalResampler::new(vs.root(), 8, 2, 2, 8);
}

#[test]
fn test_rational_resampler_large_factor() {
    let vs = make_vs();
    let rs = SpatialRationalResampler::new(vs.root(), 4, 4, 1, 4);
    let x = Tensor::randn([1, 4, 2, 4, 4], (Kind::Float, Device::Cpu));
    let out = rs.forward(&x);
    assert_eq!(out.size(), vec![1, 4, 2, 16, 16]);
}
