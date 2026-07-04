use ltx_upsampler::LatentUpsampler;
use tch::nn::ModuleT;
use tch::{Device, Kind, Tensor};

#[test]
fn test_latent_upsampler_forward_shape() {
    let up = LatentUpsampler::new(Device::Cpu, 4, 4, 1, 8, 2, 8, 1);
    let x = Tensor::randn([1, 4, 2, 8, 8], (Kind::Float, Device::Cpu));
    let out = up.forward(&x);
    // After upscale_factor=2: spatial dims double (8→16, 8→16)
    assert_eq!(out.size(), vec![1, 4, 2, 16, 16]);
}

#[test]
fn test_latent_upsampler_different_configs() {
    let up = LatentUpsampler::new(Device::Cpu, 4, 8, 2, 16, 2, 16, 1);
    let x = Tensor::randn([1, 4, 2, 8, 8], (Kind::Float, Device::Cpu));
    let out = up.forward(&x);
    assert_eq!(out.size(), vec![1, 8, 2, 16, 16]);
}

#[test]
fn test_latent_upsampler_module_t_trait() {
    let up = LatentUpsampler::new(Device::Cpu, 4, 4, 1, 8, 2, 8, 1);
    let x = Tensor::randn([1, 4, 2, 8, 8], (Kind::Float, Device::Cpu));
    let out = up.forward_t(&x, false);
    assert_eq!(out.size(), vec![1, 4, 2, 16, 16]);
}

#[test]
fn test_latent_upsampler_preserves_batch() {
    let up = LatentUpsampler::new(Device::Cpu, 4, 4, 1, 8, 2, 8, 1);
    for b in [1, 2, 4] {
        let x = Tensor::randn([b, 4, 2, 8, 8], (Kind::Float, Device::Cpu));
        let out = up.forward(&x);
        assert_eq!(out.size()[0], b);
    }
}

#[test]
fn test_latent_upsampler_preserves_time() {
    let up = LatentUpsampler::new(Device::Cpu, 4, 4, 1, 8, 2, 8, 1);
    let t = 5;
    let x = Tensor::randn([1, 4, t, 8, 8], (Kind::Float, Device::Cpu));
    let out = up.forward(&x);
    assert_eq!(out.size()[2], t);
}

#[test]
fn test_latent_upsampler_debug() {
    let up = LatentUpsampler::new(Device::Cpu, 4, 4, 1, 8, 2, 8, 1);
    let debug_str = format!("{:?}", up);
    assert!(debug_str.contains("LatentUpsampler"));
}
