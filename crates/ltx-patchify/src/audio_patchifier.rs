use tch::Tensor;

use ltx_types::{AudioLatentShape, Patchifier};

use crate::ops;

/// Patchifier for audio latent tensors.
///
/// Converts audio between `(B,C,T,F)` form and `(B,T,C*F)` patch
/// representation. Audio patchification has no spatial downsampling — each
/// time step is treated as a single token.
pub struct AudioPatchifier {
    latent_shape: AudioLatentShape,
}

impl AudioPatchifier {
    pub fn new(latent_shape: AudioLatentShape) -> Self {
        Self { latent_shape }
    }

    /// Latent shape after patchification.
    pub fn latent_shape(&self) -> &AudioLatentShape {
        &self.latent_shape
    }

    /// Number of time steps in the latent representation.
    pub fn num_time_steps(&self) -> i64 {
        self.latent_shape.time
    }

    /// Feature dimension (channels × frequency) per time step.
    pub fn feature_dim(&self) -> i64 {
        self.latent_shape.channels * self.latent_shape.features
    }
}

impl Patchifier for AudioPatchifier {
    fn patchify(&self, x: &Tensor) -> Tensor {
        ops::patchify_audio(x)
    }

    fn unpatchify(&self, x: &Tensor, shape: &[i64]) -> Tensor {
        assert_eq!(
            shape.len(),
            4,
            "unpatchify shape must be [B, C, T, F], got {} elements",
            shape.len()
        );
        ops::unpatchify_audio(x, shape[1], shape[3])
    }
}

/// Timing information for audio patchification.
///
/// Tracks how audio samples map to patch indices, enabling lossless
/// round-tripping through the patchify/unpatchify cycle with timing
/// awareness.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioTiming {
    pub start_sample: i64,
    pub end_sample: i64,
    pub start_patch: i64,
    pub end_patch: i64,
}

impl AudioTiming {
    pub fn new(start_sample: i64, end_sample: i64, start_patch: i64, end_patch: i64) -> Self {
        Self { start_sample, end_sample, start_patch, end_patch }
    }

    /// Number of patches in this timing range.
    pub fn num_patches(&self) -> i64 {
        self.end_patch - self.start_patch
    }

    /// Number of raw samples covered by this range.
    pub fn num_samples(&self) -> i64 {
        self.end_sample - self.start_sample
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn test_audio_patchifier_roundtrip() {
        let shape = AudioLatentShape::new(1, 64, 128, 128);
        let patchifier = AudioPatchifier::new(shape);
        let x = Tensor::randn([1, 64, 128, 128], (tch::Kind::Float, Device::Cpu));
        let patched = patchifier.patchify(&x);
        assert_eq!(patched.size(), vec![1, 128, 64 * 128]);
        let unp = patchifier.unpatchify(&patched, &[1, 64, 128, 128]);
        assert_eq!(unp.size(), vec![1, 64, 128, 128]);
        assert!(x.allclose(&unp, 1e-6, 1e-6, false));
    }

    #[test]
    fn test_audio_timing() {
        let timing = AudioTiming::new(0, 1024, 0, 128);
        assert_eq!(timing.num_patches(), 128);
        assert_eq!(timing.num_samples(), 1024);
    }

    #[test]
    fn test_audio_patchifier_feature_dim() {
        let shape = AudioLatentShape::new(1, 32, 64, 256);
        let patchifier = AudioPatchifier::new(shape);
        assert_eq!(patchifier.feature_dim(), 32 * 256);
        assert_eq!(patchifier.num_time_steps(), 64);
    }
}
