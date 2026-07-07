use ltx_norm::RMSNorm;
use tch::nn::{Linear, Module};
use tch::Tensor;

use crate::block::BasicAVTransformerBlock;

/// DiT (Diffusion Transformer) for video and optional audio generation.
///
/// The model processes video tokens through N transformer blocks with AdaLN
/// modulation, cross-attention to text context, and optional audio processing.
///
/// For video-only inference, use `forward()`. For audio-enabled inference,
/// use `forward_av()` which processes both video and audio tokens through
/// each block, with bidirectional cross-attention between modalities.
pub struct LTXModel {
    blocks: Vec<BasicAVTransformerBlock>,
    norm_out: RMSNorm,
    proj_out: Linear,
}

impl LTXModel {
    pub fn new(blocks: Vec<BasicAVTransformerBlock>, norm_out: RMSNorm, proj_out: Linear) -> Self {
        Self {
            blocks,
            norm_out,
            proj_out,
        }
    }

    /// Video-only forward pass.
    ///
    /// `latent`: patchified video tokens `[B, T, dim]`
    /// `timestep`: scalar diffusion timestep `[1]`
    /// `context`: text encoder output `[B, seq, context_dim]`
    pub fn forward(
        &self,
        latent: &Tensor,
        timestep: &Tensor,
        context: &Tensor,
        mask: Option<&Tensor>,
        pe: Option<(&Tensor, &Tensor)>,
    ) -> Tensor {
        let mut x = latent.shallow_clone();
        for block in &self.blocks {
            x = block.forward(&x, timestep, context, mask, pe);
        }
        let x = self.norm_out.forward(&x);
        self.proj_out.forward(&x)
    }

    /// Audio-video joint forward pass.
    ///
    /// Processes both video and audio tokens through each block with
    /// bidirectional A2V/V2A cross-attention.
    ///
    /// `video`: patchified video tokens `[B, T_v, dim]`
    /// `audio`: patchified audio tokens `[B, T_a, dim]`
    /// Returns: `(video_velocity, audio_velocity)`
    pub fn forward_av(
        &self,
        video: &Tensor,
        audio: &Tensor,
        timestep: &Tensor,
        context: &Tensor,
        mask: Option<&Tensor>,
        pe: Option<(&Tensor, &Tensor)>,
    ) -> (Tensor, Tensor) {
        let mut v = video.shallow_clone();
        let mut a = audio.shallow_clone();
        for block in &self.blocks {
            let (v_out, a_out) = block.forward_av(&v, &a, timestep, context, mask, pe);
            v = v_out;
            a = a_out;
        }
        let v = self.norm_out.forward(&v);
        let a = self.norm_out.forward(&a);
        (self.proj_out.forward(&v), self.proj_out.forward(&a))
    }
}

impl std::fmt::Debug for LTXModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LTXModel")
            .field("num_blocks", &self.blocks.len())
            .field("has_audio", &self.blocks.iter().any(|b| b.audio.is_some()))
            .finish_non_exhaustive()
    }
}
