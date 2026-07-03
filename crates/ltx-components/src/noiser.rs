use tch::Tensor;

/// Gaussian noise provider for diffusion process.
pub struct GaussianNoiser;

impl GaussianNoiser {
    pub fn new() -> Self {
        Self
    }

    /// Add Gaussian noise to a latent tensor scaled by sigma.
    pub fn add_noise(&self, latent: &Tensor, noise: &Tensor, sigma: f64) -> Tensor {
        let sigma_t = Tensor::from_slice(&[sigma as f32])
            .to_kind(latent.kind())
            .to_device(latent.device());
        latent + noise * sigma_t
    }

    /// Compute the denoised prediction from a noisy sample and velocity.
    pub fn denoise(
        &self,
        noisy: &Tensor,
        noise: &Tensor,
        sigma: f64,
    ) -> Tensor {
        let sigma_t = Tensor::from_slice(&[sigma as f32])
            .to_kind(noisy.kind())
            .to_device(noisy.device());
        (noisy - noise * sigma_t).clamp(-1.0, 1.0)
    }
}

impl Default for GaussianNoiser {
    fn default() -> Self {
        Self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn test_add_noise() {
        let noiser = GaussianNoiser::new();
        let latent = Tensor::zeros([1, 4, 8, 8], (tch::Kind::Float, Device::Cpu));
        let noise = Tensor::ones([1, 4, 8, 8], (tch::Kind::Float, Device::Cpu));
        let result = noiser.add_noise(&latent, &noise, 0.5);
        // latent(0) + noise(1) * 0.5 = 0.5
        assert!((result.mean(tch::Kind::Float).double_value(&[]) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_denoise_roundtrip() {
        let noiser = GaussianNoiser::new();
        let original = Tensor::rand([1, 4, 8, 8], (tch::Kind::Float, Device::Cpu));
        let noise = Tensor::randn([1, 4, 8, 8], (tch::Kind::Float, Device::Cpu));
        let sigma = 0.3;
        let noisy = noiser.add_noise(&original, &noise, sigma);
        let recovered = noiser.denoise(&noisy, &noise, sigma);
        assert!(original.allclose(&recovered, 1e-5, 1e-5, false));
    }
}
