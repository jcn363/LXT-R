use tch::Tensor;

/// Latent-space conditioning: provides initial latent state for inpainting or editing.
pub struct LatentCond {
    pub latent: Tensor,
    pub mask: Option<Tensor>,
}

impl LatentCond {
    pub fn new(latent: Tensor, mask: Option<Tensor>) -> Self {
        Self { latent, mask }
    }

    /// Create an unconditional latent (no conditioning).
    pub fn empty(
        batch: i64,
        channels: i64,
        frames: i64,
        height: i64,
        width: i64,
        device: tch::Device,
    ) -> Self {
        let latent = Tensor::zeros(
            [batch, channels, frames, height, width],
            (tch::Kind::Float, device),
        );
        Self { latent, mask: None }
    }
}

/// Reference video for video-to-video or video editing tasks.
pub struct ReferenceVideo {
    pub frames: Tensor,
    pub start_frame: i64,
    pub end_frame: i64,
}

impl ReferenceVideo {
    pub fn new(frames: Tensor, start_frame: i64, end_frame: i64) -> Self {
        Self {
            frames,
            start_frame,
            end_frame,
        }
    }

    /// Number of reference frames.
    pub fn num_frames(&self) -> i64 {
        self.end_frame - self.start_frame
    }

    /// Extract frames in the reference range as a slice.
    pub fn slice_frames(&self) -> Tensor {
        self.frames.narrow(2, self.start_frame, self.num_frames())
    }
}

/// Keyframe for temporal consistency in video generation.
pub struct Keyframe {
    pub frame_index: i64,
    pub latent: Tensor,
    pub strength: f64,
}

impl Keyframe {
    pub fn new(frame_index: i64, latent: Tensor, strength: f64) -> Self {
        Self {
            frame_index,
            latent,
            strength,
        }
    }

    /// Create a hard keyframe (full strength).
    pub fn hard(frame_index: i64, latent: Tensor) -> Self {
        Self::new(frame_index, latent, 1.0)
    }

    /// Create a soft keyframe (partial influence).
    pub fn soft(frame_index: i64, latent: Tensor, strength: f64) -> Self {
        Self::new(frame_index, latent, strength)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latent_cond_empty() {
        let cond = LatentCond::empty(1, 4, 8, 16, 16, tch::Device::Cpu);
        assert_eq!(cond.latent.size(), vec![1, 4, 8, 16, 16]);
        assert!(cond.mask.is_none());
    }

    #[test]
    fn test_reference_video_num_frames() {
        let frames = Tensor::zeros([1, 4, 10, 8, 8], (tch::Kind::Float, tch::Device::Cpu));
        let ref_video = ReferenceVideo::new(frames, 2, 7);
        assert_eq!(ref_video.num_frames(), 5);
    }

    #[test]
    fn test_keyframe_hard() {
        let latent = Tensor::zeros([1, 4, 1, 8, 8], (tch::Kind::Float, tch::Device::Cpu));
        let kf = Keyframe::hard(0, latent);
        assert_eq!(kf.strength, 1.0);
    }
}
