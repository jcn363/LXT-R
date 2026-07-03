use tch::Tensor;

use ltx_types::constants::ROPE_THETA;

/// Create sinusoidal timestep embedding. THE ONLY implementation.
pub fn get_timestep_embedding(timesteps: &Tensor, dim: i64, max_period: i64) -> Tensor {
    let half = dim / 2;
    let freqs = Tensor::arange_start(0i64, half, (tch::Kind::Float, timesteps.device()));
    let freqs = (-max_period as f64).ln() * freqs / (half as f64 - 1.0);
    let freqs = freqs.exp();
    let args = timesteps.unsqueeze(1).to_kind(tch::Kind::Float) * freqs.unsqueeze(0);
    Tensor::cat(&[args.sin(), args.cos()], 1)
}

/// Wrapper that applies sinusoidal projection to timesteps.
pub struct SinusoidalTimesteps {
    dim: i64,
    max_period: i64,
}

impl SinusoidalTimesteps {
    pub fn new(dim: i64) -> Self {
        Self {
            dim,
            max_period: ROPE_THETA as i64,
        }
    }

    pub fn with_max_period(dim: i64, max_period: i64) -> Self {
        Self { dim, max_period }
    }

    pub fn forward(&self, timesteps: &Tensor) -> Tensor {
        get_timestep_embedding(timesteps, self.dim, self.max_period)
    }
}
