use ltx_upsampler::BlurDownsample;
use tch::nn::ModuleT;
use tch::{Device, Kind, Tensor};

fn make_vs() -> tch::nn::VarStore {
    tch::nn::VarStore::new(Device::Cpu)
}

#[test]
fn test_blur_downsample_4d_stride_2() {
    let vs = make_vs();
    let bd = BlurDownsample::new(vs.root(), 4, 8, 2, None);
    let x = Tensor::randn([1, 4, 16, 16], (Kind::Float, Device::Cpu));
    let out = bd.forward(&x);
    assert_eq!(out.size(), vec![1, 8, 8, 8]);
}

#[test]
fn test_blur_downsample_4d_stride_4() {
    let vs = make_vs();
    let bd = BlurDownsample::new(vs.root(), 4, 8, 4, None);
    let x = Tensor::randn([1, 4, 16, 16], (Kind::Float, Device::Cpu));
    let out = bd.forward(&x);
    assert_eq!(out.size(), vec![1, 8, 4, 4]);
}

#[test]
fn test_blur_downsample_5d_stride_2() {
    let vs = make_vs();
    let bd = BlurDownsample::new(vs.root(), 4, 8, 2, None);
    let x = Tensor::randn([1, 4, 3, 16, 16], (Kind::Float, Device::Cpu));
    let out = bd.forward(&x);
    assert_eq!(out.size(), vec![1, 8, 3, 8, 8]);
}

#[test]
fn test_blur_downsample_5d_preserves_time() {
    let vs = make_vs();
    let bd = BlurDownsample::new(vs.root(), 4, 8, 2, None);
    let t = 5;
    let x = Tensor::randn([1, 4, t, 16, 16], (Kind::Float, Device::Cpu));
    let out = bd.forward(&x);
    assert_eq!(out.size()[2], t);
}

#[test]
fn test_blur_downsample_custom_kernel() {
    let vs = make_vs();
    let bd = BlurDownsample::new(vs.root(), 4, 8, 2, Some(5));
    let x = Tensor::randn([1, 4, 16, 16], (Kind::Float, Device::Cpu));
    let out = bd.forward(&x);
    // kernel=5 with stride=2: output spatial = (16 - 5 + 2*2) / 2 + 1 = 7
    assert_eq!(out.size(), vec![1, 8, 7, 7]);
}

#[test]
fn test_blur_downsample_module_t_trait() {
    let vs = make_vs();
    let bd = BlurDownsample::new(vs.root(), 4, 8, 2, None);
    let x = Tensor::randn([1, 4, 16, 16], (Kind::Float, Device::Cpu));
    let out = bd.forward_t(&x, false);
    assert_eq!(out.size(), vec![1, 8, 8, 8]);
}

#[test]
#[should_panic(expected = "stride must be 2 or 4")]
fn test_blur_downsample_invalid_stride() {
    let vs = make_vs();
    let _bd = BlurDownsample::new(vs.root(), 4, 8, 3, None);
}

#[test]
#[should_panic(expected = "BlurDownsample expects 4D or 5D input")]
fn test_blur_downsample_3d_input_panics() {
    let vs = make_vs();
    let bd = BlurDownsample::new(vs.root(), 4, 8, 2, None);
    let x = Tensor::randn([4, 16, 16], (Kind::Float, Device::Cpu));
    let _ = bd.forward(&x);
}

// ── Bug verification: Gaussian kernel value tests ──────────────────────

/// Bug verification: the 2D Gaussian kernel must be [K, K], not a scalar.
#[test]
fn test_gaussian_kernel_shape() {
    for k in [3, 5, 7] {
        let kernel = BlurDownsample::make_gaussian_kernel(k);
        assert_eq!(kernel.size(), vec![k, k], "kernel size for k={k}");
    }
}

/// Bug verification: the 2D Gaussian kernel must sum to 1.0 (valid probability distribution).
#[test]
fn test_gaussian_kernel_sums_to_one() {
    for k in [3, 5, 7] {
        let kernel = BlurDownsample::make_gaussian_kernel(k);
        let sum = kernel.sum(Kind::Float).double_value(&[]);
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "kernel k={k} sums to {sum}, expected 1.0"
        );
    }
}

/// Bug verification: the 2D Gaussian kernel must be symmetric.
#[test]
fn test_gaussian_kernel_symmetry() {
    let kernel = BlurDownsample::make_gaussian_kernel(5);
    // K[i,j] == K[K-1-i, K-1-j] (centrally symmetric)
    for i in 0..5 {
        for j in 0..5 {
            let a = kernel.get(i).get(j).double_value(&[]);
            let b = kernel.get(4 - i).get(4 - j).double_value(&[]);
            assert!(
                (a - b).abs() < 1e-7,
                "kernel not symmetric: K[{i},{j}]={a} != K[{},{j}]={b}",
                4 - i,
            );
        }
    }
}

/// Bug verification: the center element must be the maximum.
#[test]
fn test_gaussian_kernel_center_is_max() {
    let kernel = BlurDownsample::make_gaussian_kernel(5);
    let center = kernel.get(2).get(2).double_value(&[]);
    let max_val = kernel.max().double_value(&[]);
    assert!(
        (center - max_val).abs() < 1e-7,
        "center={center} is not the max={max_val}"
    );
}
