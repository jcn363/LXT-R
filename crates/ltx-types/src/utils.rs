use tch::Tensor;

use crate::constants::PROJECTION_EPS;

/// Convert sample + denoised to velocity. THE ONLY implementation.
pub fn to_velocity(sample: &Tensor, sigma: f64, denoised: &Tensor, calc_dtype: tch::Kind) -> Tensor {
    assert!(sigma != 0.0, "Sigma can't be 0.0");
    ((sample.to_kind(calc_dtype) - denoised.to_kind(calc_dtype)) / sigma).to_kind(sample.kind())
}

/// Convert sample + velocity to denoised. THE ONLY implementation.
pub fn to_denoised(sample: &Tensor, velocity: &Tensor, sigma: f64, calc_dtype: tch::Kind) -> Tensor {
    let sigma_t = Tensor::from_slice(&[sigma as f32])
        .to_kind(calc_dtype)
        .to_device(sample.device());
    (sample.to_kind(calc_dtype) - velocity.to_kind(calc_dtype) * sigma_t).to_kind(sample.kind())
}

/// Projection coefficient for APG guider. THE ONLY implementation.
pub fn projection_coef(to_project: &Tensor, project_onto: &Tensor) -> Tensor {
    let b = to_project.size()[0];
    let pos_flat = to_project.reshape([b, -1]);
    let neg_flat = project_onto.reshape([b, -1]);
    let dims: &[i64] = &[1];
    let dot = (&pos_flat * &neg_flat).sum_dim_intlist(dims, true, tch::Kind::Float);
    let sq_norm =
        (&neg_flat * &neg_flat).sum_dim_intlist(dims, true, tch::Kind::Float) + PROJECTION_EPS;
    dot / sq_norm
}
