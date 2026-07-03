use tch::Tensor;

pub trait Patchifier {
    fn patchify(&self, x: &Tensor) -> Tensor;
    fn unpatchify(&self, x: &Tensor, shape: &[i64]) -> Tensor;
}

pub trait Scheduler {
    fn sigmas(&self, n_steps: usize) -> Vec<f64>;
    fn step(&self, x: &Tensor, sigma: f64, denoised: &Tensor) -> Tensor;
}

pub trait Guider {
    fn guidance_scale(&self) -> f64;
    fn guide(&self, x: &Tensor, uncond: &Tensor) -> Tensor;
}

pub trait Denoiser {
    fn denoise(&self, x: &Tensor, sigma: f64, conditioning: &Tensor) -> Tensor;
}

pub trait TimestepEmbedder {
    fn embed(&self, timesteps: &Tensor) -> Tensor;
}
