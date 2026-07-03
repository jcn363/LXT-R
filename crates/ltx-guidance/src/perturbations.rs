use serde::Deserialize;

/// Configuration for latent perturbations during conditioning.
#[derive(Debug, Clone, Deserialize)]
pub struct PerturbationConfig {
    /// Scale of perturbation noise (0.0 = no perturbation).
    pub scale: f64,
    /// Perturbation type.
    pub kind: PerturbationKind,
    /// Number of perturbation steps (for iterative perturbation).
    pub n_steps: usize,
}

impl Default for PerturbationConfig {
    fn default() -> Self {
        Self {
            scale: 0.0,
            kind: PerturbationKind::Gaussian,
            n_steps: 1,
        }
    }
}

/// Type of perturbation applied to latents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum PerturbationKind {
    /// Additive Gaussian noise.
    Gaussian,
    /// Uniform random noise in [-scale, scale].
    Uniform,
    /// Dropout (random zeroing of elements).
    Dropout,
}

/// Configuration for STG perturbation routing.
#[derive(Debug, Clone, Deserialize)]
pub struct StgPerturbationConfig {
    /// Whether to perturb spatial attention layers.
    pub spatial: bool,
    /// Whether to perturb temporal attention layers.
    pub temporal: bool,
    /// Spatial perturbation scale.
    pub spatial_scale: f64,
    /// Temporal perturbation scale.
    pub temporal_scale: f64,
}

impl Default for StgPerturbationConfig {
    fn default() -> Self {
        Self {
            spatial: true,
            temporal: true,
            spatial_scale: 0.0,
            temporal_scale: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perturbation_config_default() {
        let cfg = PerturbationConfig::default();
        assert_eq!(cfg.scale, 0.0);
        assert_eq!(cfg.kind, PerturbationKind::Gaussian);
    }

    #[test]
    fn test_stg_config_default() {
        let cfg = StgPerturbationConfig::default();
        assert!(cfg.spatial);
        assert!(cfg.temporal);
    }
}
