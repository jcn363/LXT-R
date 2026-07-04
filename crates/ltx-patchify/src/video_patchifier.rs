use tch::Tensor;

use ltx_types::{Patchifier, VideoLatentShape};

use crate::ops;

/// Patchifier for video latent tensors.
///
/// Converts video between pixel-space `(B,C,F,H,W)` and latent patch
/// representation `(B, T, D)` where `T = (F/p1)*(H/p2)*(W/p3)` and
/// `D = C*p1*p2*p3`.
///
/// The patch sizes `(p1, p2, p3)` control temporal, height, and width
/// downsampling respectively.
pub struct VideoLatentPatchifier {
    latent_shape: VideoLatentShape,
    patch_size: [i64; 3],
}

impl VideoLatentPatchifier {
    pub fn new(latent_shape: VideoLatentShape, patch_size: [i64; 3]) -> Self {
        Self {
            latent_shape,
            patch_size,
        }
    }

    /// Create from channel count and patch sizes, deriving latent shape from
    /// input pixel dimensions divided by the patch factors.
    pub fn from_pixel_shape(
        batch: i64,
        channels: i64,
        frames: i64,
        height: i64,
        width: i64,
        patch_size: [i64; 3],
    ) -> Self {
        let latent_shape = VideoLatentShape::new(
            batch,
            channels,
            frames / patch_size[0],
            height / patch_size[1],
            width / patch_size[2],
        );
        Self {
            latent_shape,
            patch_size,
        }
    }

    /// Patch size `[p1, p2, p3]` for temporal, height, width axes.
    pub fn patch_size(&self) -> [i64; 3] {
        self.patch_size
    }

    /// Latent shape after patchification.
    pub fn latent_shape(&self) -> &VideoLatentShape {
        &self.latent_shape
    }

    /// Convert pixel-space dimensions to latent-space dimensions.
    pub fn pixel_to_latent_dims(&self, frames: i64, height: i64, width: i64) -> (i64, i64, i64) {
        (
            frames / self.patch_size[0],
            height / self.patch_size[1],
            width / self.patch_size[2],
        )
    }

    /// Convert latent-space dimensions back to pixel-space.
    pub fn latent_to_pixel_dims(&self, frames: i64, height: i64, width: i64) -> (i64, i64, i64) {
        (
            frames * self.patch_size[0],
            height * self.patch_size[1],
            width * self.patch_size[2],
        )
    }
}

impl Patchifier for VideoLatentPatchifier {
    fn patchify(&self, x: &Tensor) -> Tensor {
        ops::patchify_5d(
            x,
            self.patch_size[0],
            self.patch_size[1],
            self.patch_size[2],
        )
    }

    fn unpatchify(&self, x: &Tensor, shape: &[i64]) -> Tensor {
        assert_eq!(
            shape.len(),
            5,
            "unpatchify shape must be [B, C, F, H, W], got {} elements",
            shape.len()
        );
        ops::unpatchify_5d(
            x,
            shape[0],
            shape[1],
            shape[2],
            shape[3],
            shape[4],
            self.patch_size[0],
            self.patch_size[1],
            self.patch_size[2],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn test_video_patchifier_roundtrip() {
        let shape = VideoLatentShape::new(1, 128, 4, 16, 16);
        let patchifier = VideoLatentPatchifier::new(shape, [2, 4, 4]);
        let x = Tensor::randn([1, 128, 8, 64, 64], (tch::Kind::Float, Device::Cpu));
        let patched = patchifier.patchify(&x);
        // T = 4*16*16 = 1024, D = 128*2*4*4 = 4096
        assert_eq!(patched.size(), vec![1, 1024, 4096]);
        let unp = patchifier.unpatchify(&patched, &[1, 128, 8, 64, 64]);
        assert_eq!(unp.size(), vec![1, 128, 8, 64, 64]);
        assert!(x.allclose(&unp, 1e-6, 1e-6, false));
    }

    #[test]
    fn test_pixel_to_latent_dims() {
        let shape = VideoLatentShape::new(1, 128, 4, 16, 16);
        let patchifier = VideoLatentPatchifier::new(shape, [2, 4, 4]);
        let (f, h, w) = patchifier.pixel_to_latent_dims(16, 128, 128);
        assert_eq!((f, h, w), (8, 32, 32));
    }

    #[test]
    fn test_latent_to_pixel_dims() {
        let shape = VideoLatentShape::new(1, 128, 4, 16, 16);
        let patchifier = VideoLatentPatchifier::new(shape, [2, 4, 4]);
        let (f, h, w) = patchifier.latent_to_pixel_dims(4, 16, 16);
        assert_eq!((f, h, w), (8, 64, 64));
    }
}
