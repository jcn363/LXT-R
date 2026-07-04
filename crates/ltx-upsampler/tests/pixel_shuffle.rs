use ltx_upsampler::PixelShuffleND;
use tch::nn::ModuleT;
use tch::{Device, Kind, Tensor};

#[test]
fn test_pixel_shuffle_3d_upscale_2() {
    let ps = PixelShuffleND::new(2, 3);
    // (B=1, C*4=8, T=2, H=4, W=4) → (B=1, C=2, T=2, H=8, W=8)
    let x = Tensor::randn([1, 8, 2, 4, 4], (Kind::Float, Device::Cpu));
    let out = ps.forward(&x);
    assert_eq!(out.size(), vec![1, 2, 2, 8, 8]);
}

#[test]
fn test_pixel_shuffle_3d_upscale_4() {
    let ps = PixelShuffleND::new(4, 3);
    // (B=1, C*16=16, T=1, H=2, W=2) → (B=1, C=1, T=1, H=8, W=8)
    let x = Tensor::randn([1, 16, 1, 2, 2], (Kind::Float, Device::Cpu));
    let out = ps.forward(&x);
    assert_eq!(out.size(), vec![1, 1, 1, 8, 8]);
}

#[test]
fn test_pixel_shuffle_3d_preserves_time() {
    let ps = PixelShuffleND::new(2, 3);
    let t = 5;
    let x = Tensor::randn([1, 8, t, 4, 4], (Kind::Float, Device::Cpu));
    let out = ps.forward(&x);
    assert_eq!(out.size()[2], t);
}

#[test]
fn test_pixel_shuffle_2d_upscale_2() {
    let ps = PixelShuffleND::new(2, 2);
    // (B=2, C*4=12, H=8, W=8) → (B=2, C=3, H=16, W=16)
    let x = Tensor::randn([2, 12, 8, 8], (Kind::Float, Device::Cpu));
    let out = ps.forward(&x);
    assert_eq!(out.size(), vec![2, 3, 16, 16]);
}

#[test]
fn test_pixel_shuffle_2d_upscale_4() {
    let ps = PixelShuffleND::new(4, 2);
    // (B=1, C*16=32, H=4, W=4) → (B=1, C=2, H=16, W=16)
    let x = Tensor::randn([1, 32, 4, 4], (Kind::Float, Device::Cpu));
    let out = ps.forward(&x);
    assert_eq!(out.size(), vec![1, 2, 16, 16]);
}

#[test]
fn test_pixel_shuffle_upscale_factor_accessor() {
    let ps = PixelShuffleND::new(4, 3);
    assert_eq!(ps.upscale_factor(), 4);
}

#[test]
fn test_pixel_shuffle_module_t_trait() {
    let ps = PixelShuffleND::new(2, 3);
    let x = Tensor::randn([1, 8, 2, 4, 4], (Kind::Float, Device::Cpu));
    let out = ps.forward_t(&x, false);
    assert_eq!(out.size(), vec![1, 2, 2, 8, 8]);
}

#[test]
#[should_panic(expected = "upscale_factor must be positive")]
fn test_pixel_shuffle_zero_upscale_panics() {
    let _ps = PixelShuffleND::new(0, 3);
}

#[test]
#[should_panic(expected = "num_spatial_dims must be 2 or 3")]
fn test_pixel_shuffle_1d_panics() {
    let _ps = PixelShuffleND::new(2, 1);
}

#[test]
fn test_pixel_shuffle_batch_preservation() {
    let ps = PixelShuffleND::new(2, 3);
    for b in [1, 2, 4] {
        let x = Tensor::randn([b, 8, 2, 4, 4], (Kind::Float, Device::Cpu));
        let out = ps.forward(&x);
        assert_eq!(out.size()[0], b);
    }
}
