use tch::Tensor;

use ltx_types::{Scheduler, DEFAULT_BASE_SHIFT, DEFAULT_MAX_SHIFT, DEFAULT_TERMINAL};

/// Linear-quadratic scheduler: linear from sigma=1 to `quadratic_begin`, then quadratic decay.
pub struct LinearQuadratic {
    quadratic_begin: f64,
}

impl LinearQuadratic {
    pub fn new(quadratic_begin: f64) -> Self {
        Self { quadratic_begin }
    }
}

impl Default for LinearQuadratic {
    fn default() -> Self {
        Self { quadratic_begin: DEFAULT_TERMINAL }
    }
}

impl Scheduler for LinearQuadratic {
    fn sigmas(&self, n_steps: usize) -> Vec<f64> {
        let n = n_steps as f64;
        let mut sigmas = Vec::with_capacity(n_steps + 1);
        for i in 0..=n_steps {
            let t = i as f64 / n;
            let sigma = if t <= self.quadratic_begin {
                // Linear region: 1.0 -> quadratic_begin
                1.0 + (self.quadratic_begin - 1.0) * t
            } else {
                // Quadratic region: quadratic_begin -> 0
                let local_t = (t - self.quadratic_begin) / (1.0 - self.quadratic_begin);
                self.quadratic_begin * (1.0 - local_t).powi(2)
            };
            sigmas.push(sigma.clamp(0.0, 1.0));
        }
        sigmas
    }

    fn step(&self, x: &Tensor, sigma: f64, denoised: &Tensor) -> Tensor {
        let next_sigma = sigma * (1.0 - 1.0 / (x.size()[0] as f64).max(1.0));
        let next_sigma = next_sigma.max(0.0);
        x + (denoised - x) * (sigma - next_sigma)
    }
}

/// Beta scheduler with configurable alpha/beta parameters for noise schedule.
pub struct Beta {
    alpha: f64,
    beta: f64,
}

impl Beta {
    pub fn new(alpha: f64, beta: f64) -> Self {
        Self { alpha, beta }
    }
}

impl Default for Beta {
    fn default() -> Self {
        Self { alpha: 0.6, beta: 0.2 }
    }
}

impl Scheduler for Beta {
    fn sigmas(&self, n_steps: usize) -> Vec<f64> {
        let n = n_steps as f64;
        let mut sigmas = Vec::with_capacity(n_steps + 1);
        for i in 0..=n_steps {
            let t = i as f64 / n;
            // Beta CDF-like schedule: smooth S-curve interpolation
            let sigma = 1.0 - (t * self.alpha + (1.0 - t) * self.beta);
            sigmas.push(sigma.clamp(0.0, 1.0));
        }
        sigmas
    }

    fn step(&self, x: &Tensor, sigma: f64, denoised: &Tensor) -> Tensor {
        let next_sigma = sigma * (1.0 - 1.0 / (x.size()[0] as f64).max(1.0));
        let next_sigma = next_sigma.max(0.0);
        x + (denoised - x) * (sigma - next_sigma)
    }
}

/// LTX-2 default scheduler with shift-based sigma schedule.
pub struct Ltx2Scheduler {
    max_shift: f64,
    base_shift: f64,
    terminal: f64,
}

impl Ltx2Scheduler {
    pub fn new(max_shift: f64, base_shift: f64, terminal: f64) -> Self {
        Self { max_shift, base_shift, terminal }
    }
}

impl Default for Ltx2Scheduler {
    fn default() -> Self {
        Self {
            max_shift: DEFAULT_MAX_SHIFT,
            base_shift: DEFAULT_BASE_SHIFT,
            terminal: DEFAULT_TERMINAL,
        }
    }
}

impl Scheduler for Ltx2Scheduler {
    fn sigmas(&self, n_steps: usize) -> Vec<f64> {
        let n = n_steps as f64;
        let mut sigmas = Vec::with_capacity(n_steps + 1);
        for i in 0..=n_steps {
            let t = i as f64 / n;
            // Shift-based schedule: maps linear t to sigma via logit-like transform
            let shifted = t * self.max_shift + self.base_shift;
            let sigma = self.terminal + (1.0 - self.terminal) / (1.0 + (shifted - self.base_shift - self.max_shift / 2.0).exp());
            sigmas.push(sigma.clamp(0.0, 1.0));
        }
        sigmas
    }

    fn step(&self, x: &Tensor, sigma: f64, denoised: &Tensor) -> Tensor {
        let next_sigma = sigma * (1.0 - 1.0 / (x.size()[0] as f64).max(1.0));
        let next_sigma = next_sigma.max(0.0);
        x + (denoised - x) * (sigma - next_sigma)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn test_scheduler_trait_impl() {
        let sched = Ltx2Scheduler::default();
        let sigmas = sched.sigmas(10);
        assert_eq!(sigmas.len(), 11);
        // Sigmas should be non-increasing
        for w in sigmas.windows(2) {
            assert!(w[0] >= w[1], "sigmas must be non-increasing");
        }
    }

    #[test]
    fn test_linear_quadratic_step() {
        let sched = LinearQuadratic::default();
        let x = Tensor::ones([1, 1, 4, 4], (tch::Kind::Float, Device::Cpu));
        let denoised = Tensor::zeros([1, 1, 4, 4], (tch::Kind::Float, Device::Cpu));
        let result = sched.step(&x, 0.5, &denoised);
        assert_eq!(result.size(), vec![1, 1, 4, 4]);
    }

    #[test]
    fn test_beta_sigmas() {
        let sched = Beta::default();
        let sigmas = sched.sigmas(5);
        assert_eq!(sigmas.len(), 6);
        assert!((sigmas[0] - 1.0).abs() < 0.01);
    }
}
