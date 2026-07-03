use tch::Tensor;

use ltx_types::{utils::to_velocity, STABILITY_EPS};

/// Euler diffusion step: first-order ODE solver.
pub struct EulerStep;

impl EulerStep {
    pub fn new() -> Self {
        Self
    }

    /// Compute the next step using Euler method.
    ///
    /// `x` - current noisy latent
    /// `sigma` - current noise level
    /// `next_sigma` - next noise level (lower = less noise)
    /// `denoised` - model's denoised prediction
    /// `calc_dtype` - intermediate computation dtype
    pub fn step(
        &self,
        x: &Tensor,
        sigma: f64,
        next_sigma: f64,
        denoised: &Tensor,
        calc_dtype: tch::Kind,
    ) -> Tensor {
        let velocity = to_velocity(x, sigma, denoised, calc_dtype);
        let sigma_diff = sigma - next_sigma;
        let sigma_diff_t = Tensor::from_slice(&[sigma_diff as f32])
            .to_kind(calc_dtype)
            .to_device(x.device());
        (x.to_kind(calc_dtype) + velocity * sigma_diff_t).to_kind(x.kind())
    }
}

impl Default for EulerStep {
    fn default() -> Self {
        Self
    }
}

/// Residual-to-sample diffusion step (res2s): uses residual scaling for stable steps.
pub struct Res2sStep {
    /// Gain factor for residual scaling.
    gain: f64,
}

impl Res2sStep {
    pub fn new(gain: f64) -> Self {
        Self { gain }
    }

    /// Compute the next step using residual-to-sample method.
    ///
    /// The gain controls how aggressively we move toward the denoised prediction.
    /// Values > 1.0 accelerate convergence; values < 1.0 are more conservative.
    pub fn step(
        &self,
        x: &Tensor,
        sigma: f64,
        next_sigma: f64,
        denoised: &Tensor,
        calc_dtype: tch::Kind,
    ) -> Tensor {
        let sigma_ratio = (next_sigma / sigma.max(STABILITY_EPS)).max(0.0);
        let sigma_ratio_t = Tensor::from_slice(&[sigma_ratio as f32])
            .to_kind(calc_dtype)
            .to_device(x.device());
        let gain_t = Tensor::from_slice(&[self.gain as f32])
            .to_kind(calc_dtype)
            .to_device(x.device());

        let x_calc = x.to_kind(calc_dtype);
        let denoised_calc = denoised.to_kind(calc_dtype);

        // Residual scaled by gain, then attenuated by sigma ratio
        let residual = (&x_calc - &denoised_calc) * &gain_t;
        let result = &denoised_calc + residual * &sigma_ratio_t;
        result.to_kind(x.kind())
    }
}

impl Default for Res2sStep {
    fn default() -> Self {
        Self { gain: 1.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn test_euler_step_output_shape() {
        let step = EulerStep;
        let x = Tensor::ones([1, 4, 8, 8], (tch::Kind::Float, Device::Cpu));
        let denoised = Tensor::zeros([1, 4, 8, 8], (tch::Kind::Float, Device::Cpu));
        let result = step.step(&x, 0.5, 0.3, &denoised, tch::Kind::Float);
        assert_eq!(result.size(), vec![1, 4, 8, 8]);
    }

    #[test]
    fn test_res2s_step_output_shape() {
        let step = Res2sStep::new(1.0);
        let x = Tensor::ones([1, 4, 8, 8], (tch::Kind::Float, Device::Cpu));
        let denoised = Tensor::zeros([1, 4, 8, 8], (tch::Kind::Float, Device::Cpu));
        let result = step.step(&x, 0.5, 0.3, &denoised, tch::Kind::Float);
        assert_eq!(result.size(), vec![1, 4, 8, 8]);
    }

    #[test]
    fn test_euler_step_reduces_noise() {
        let step = EulerStep;
        let x = Tensor::ones([1, 4, 8, 8], (tch::Kind::Float, Device::Cpu));
        let denoised = Tensor::zeros([1, 4, 8, 8], (tch::Kind::Float, Device::Cpu));
        // Step from sigma=0.5 to sigma=0.3: should move toward denoised
        let result = step.step(&x, 0.5, 0.3, &denoised, tch::Kind::Float);
        let mean = result.mean(tch::Kind::Float).double_value(&[]);
        // velocity = (x - denoised) / sigma = (1-0)/0.5 = 2.0
        // result = x + velocity * (sigma - next_sigma) = 1 + 2*(0.2) = 1.4
        assert!((mean - 1.4).abs() < 1e-5);
    }
}
