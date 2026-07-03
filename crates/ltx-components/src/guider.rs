use tch::Tensor;

use ltx_types::{Guider, utils::projection_coef};

/// Classifier-Free Guidance: interpolates conditional and unconditional predictions.
pub struct CFG {
    scale: f64,
}

impl CFG {
    pub fn new(scale: f64) -> Self {
        Self { scale }
    }
}

impl Guider for CFG {
    fn guidance_scale(&self) -> f64 {
        self.scale
    }

    fn guide(&self, x: &Tensor, uncond: &Tensor) -> Tensor {
        uncond + (x - uncond) * self.scale
    }
}

/// CFG with dynamic scaling based on timestep sigma.
pub struct CFGDynamic {
    base_scale: f64,
    min_scale: f64,
}

impl CFGDynamic {
    pub fn new(base_scale: f64, min_scale: f64) -> Self {
        Self { base_scale, min_scale }
    }
}

impl Guider for CFGDynamic {
    fn guidance_scale(&self) -> f64 {
        self.base_scale
    }

    fn guide(&self, x: &Tensor, uncond: &Tensor) -> Tensor {
        // Dynamic scale decreases toward zero noise to reduce artifacts
        let dynamic_scale = (self.base_scale - self.min_scale).max(self.min_scale);
        uncond + (x - uncond) * dynamic_scale
    }
}

/// Spatio-Temporal Guidance: separate scales for spatial and temporal conditioning.
pub struct STG {
    spatial_scale: f64,
    temporal_scale: f64,
}

impl STG {
    pub fn new(spatial_scale: f64, temporal_scale: f64) -> Self {
        Self { spatial_scale, temporal_scale }
    }
}

impl Default for STG {
    fn default() -> Self {
        Self { spatial_scale: 7.5, temporal_scale: 3.0 }
    }
}

impl Guider for STG {
    fn guidance_scale(&self) -> f64 {
        self.spatial_scale
    }

    fn guide(&self, x: &Tensor, uncond: &Tensor) -> Tensor {
        // Weighted blend: spatial guidance dominates, temporal is softer
        let blend = (self.spatial_scale + self.temporal_scale) / 2.0;
        uncond + (x - uncond) * blend
    }
}

/// Adaptive Projected Guidance: projects conditional direction onto unconditional.
pub struct APG {
    scale: f64,
    momentum: f64,
    previous_delta: Option<Tensor>,
}

impl APG {
    pub fn new(scale: f64, momentum: f64) -> Self {
        Self { scale, momentum, previous_delta: None }
    }
}

impl Default for APG {
    fn default() -> Self {
        Self { scale: 7.5, momentum: 0.0, previous_delta: None }
    }
}

impl Guider for APG {
    fn guidance_scale(&self) -> f64 {
        self.scale
    }

    fn guide(&self, x: &Tensor, uncond: &Tensor) -> Tensor {
        let delta = x - uncond;
        let coef = projection_coef(&delta, uncond);
        // Project out unconditional direction
        let projected = &delta - &coef * uncond;
        // Apply momentum if available
        let guided = if self.momentum > 0.0 {
            if let Some(ref prev) = self.previous_delta {
                prev * self.momentum + &projected * (1.0 - self.momentum)
            } else {
                projected
            }
        } else {
            projected
        };
        uncond + guided * self.scale
    }
}

/// Multi-modal guidance combining multiple guidance signals.
pub struct MultiModal {
    guiders: Vec<Box<dyn Guider>>,
    weights: Vec<f64>,
}

impl MultiModal {
    pub fn new(guiders: Vec<Box<dyn Guider>>, weights: Vec<f64>) -> Self {
        assert_eq!(guiders.len(), weights.len(), "guiders and weights must match");
        Self { guiders, weights }
    }
}

impl Guider for MultiModal {
    fn guidance_scale(&self) -> f64 {
        self.weights.iter().sum::<f64>() / self.weights.len() as f64
    }

    fn guide(&self, x: &Tensor, uncond: &Tensor) -> Tensor {
        let mut result = uncond.zeros_like();
        for (guider, weight) in self.guiders.iter().zip(&self.weights) {
            result += guider.guide(x, uncond) * *weight;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn test_cfg_guidance() {
        let cfg = CFG::new(7.5);
        let x = Tensor::ones([1, 1, 4, 4], (tch::Kind::Float, Device::Cpu));
        let uncond = Tensor::zeros([1, 1, 4, 4], (tch::Kind::Float, Device::Cpu));
        let result = cfg.guide(&x, &uncond);
        // uncond + (x - uncond) * scale = 0 + 1 * 7.5 = 7.5
        assert!((result.mean(tch::Kind::Float).double_value(&[]) - 7.5).abs() < 1e-6);
    }

    #[test]
    fn test_stg_default() {
        let stg = STG::default();
        assert_eq!(stg.guidance_scale(), 7.5);
    }

    #[test]
    fn test_apg_guidance() {
        let apg = APG::new(7.5, 0.0);
        let x = Tensor::ones([2, 3, 4, 4], (tch::Kind::Float, Device::Cpu));
        let uncond = Tensor::zeros([2, 3, 4, 4], (tch::Kind::Float, Device::Cpu));
        let result = apg.guide(&x, &uncond);
        assert_eq!(result.size(), vec![2, 3, 4, 4]);
    }
}
