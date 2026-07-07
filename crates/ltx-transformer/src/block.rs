use ltx_attention::{RopeType, TransformerAttention};
use ltx_norm::RMSNorm;
use ltx_timestep::AdaLayerNormSingle;
use ltx_types::DEFAULT_SINUSOIDAL_DIM;
use tch::Tensor;

use crate::feed_forward::FeedForward;

/// Optional audio modality components for a transformer block.
///
/// When present, the block processes both video and audio tokens with
/// separate self-attention, cross-attention, and feedforward paths for
/// audio, plus bidirectional A2V/V2A cross-attention between modalities.
pub struct AudioModality {
    pub adaln: AdaLayerNormSingle,
    pub self_attn: TransformerAttention,
    pub cross_attn: TransformerAttention,
    pub norm1: RMSNorm,
    pub norm_cross: RMSNorm,
    pub norm2: RMSNorm,
    pub ff: FeedForward,
    /// A2V cross-attention: Q=video, KV=audio
    pub a2v_attn: TransformerAttention,
    pub norm_a2v: RMSNorm,
    /// V2A cross-attention: Q=audio, KV=video
    pub v2a_attn: TransformerAttention,
    pub norm_v2a: RMSNorm,
    /// Scale-shift table for A2V/V2A: [5, dim] → shift_a2v, scale_a2v, gate_a2v, shift_v2a, gate_v2a
    pub scale_shift_table: Tensor,
}

impl AudioModality {
    pub fn new(
        vs: &tch::nn::Path,
        dim: i64,
        num_heads: i64,
        head_dim: i64,
        context_dim: Option<i64>,
        rope_type: RopeType,
    ) -> Self {
        let adaln = AdaLayerNormSingle::new_with_input_dim(
            &(vs / "adaln"), dim, DEFAULT_SINUSOIDAL_DIM,
        );
        let self_attn = TransformerAttention::new(
            &(vs / "audio_attn1"),
            dim,
            num_heads,
            head_dim,
            None,
            rope_type,
        );
        let cross_attn = TransformerAttention::new(
            &(vs / "audio_attn2"),
            dim,
            num_heads,
            head_dim,
            context_dim,
            rope_type,
        );
        let norm1 = RMSNorm::default_eps_with_path(vs / "audio_norm1", dim);
        let norm_cross = RMSNorm::default_eps_with_path(vs / "audio_norm_cross", dim);
        let norm2 = RMSNorm::default_eps_with_path(vs / "audio_norm2", dim);
        let ff = FeedForward::new(&(vs / "audio_ff"), dim);

        let a2v_attn = TransformerAttention::new(
            &(vs / "audio_to_video_attn"),
            dim,
            num_heads,
            head_dim,
            None,
            rope_type,
        );
        let norm_a2v = RMSNorm::default_eps_with_path(vs / "audio_norm_a2v", dim);

        let v2a_attn = TransformerAttention::new(
            &(vs / "video_to_audio_attn"),
            dim,
            num_heads,
            head_dim,
            None,
            rope_type,
        );
        let norm_v2a = RMSNorm::default_eps_with_path(vs / "audio_norm_v2a", dim);

        let scale_shift_table =
            vs.var("audio_scale_shift_table", &[5, dim], tch::nn::init::Init::Const(0.0));

        Self {
            adaln,
            self_attn,
            cross_attn,
            norm1,
            norm_cross,
            norm2,
            ff,
            a2v_attn,
            norm_a2v,
            v2a_attn,
            norm_v2a,
            scale_shift_table,
        }
    }

    /// Process audio tokens through self-attn, cross-attn (text), FFN,
    /// and V2A cross-attention with video tokens.
    pub fn forward(
        &self,
        audio: &Tensor,
        video: &Tensor,
        timestep: &Tensor,
        context: &Tensor,
        mask: Option<&Tensor>,
        pe: Option<(&Tensor, &Tensor)>,
    ) -> (Tensor, Tensor) {
        let (modulation, _) = self.adaln_forward(timestep, audio.kind());
        let chunks: Vec<Tensor> = modulation.chunk(6, -1);
        let (shift_msa, scale_msa, gate_msa) = (
            chunks[0].unsqueeze(1),
            chunks[1].unsqueeze(1),
            chunks[2].unsqueeze(1),
        );
        let (shift_mlp, scale_mlp, gate_mlp) = (
            chunks[3].unsqueeze(1),
            chunks[4].unsqueeze(1),
            chunks[5].unsqueeze(1),
        );

        // Audio self-attention
        let h = self.norm1.forward(audio) * (Tensor::ones_like(&scale_msa) + &scale_msa) + &shift_msa;
        let h = self.self_attn.forward(&h, None, mask, pe);
        let audio = audio + &gate_msa * h;

        // Audio cross-attention with text context
        let h = self.norm_cross.forward(&audio);
        let h = self.cross_attn.forward(&h, Some(context), mask, None);
        let audio = audio + h;

        // Audio FFN
        let h = self.norm2.forward(&audio) * (Tensor::ones_like(&scale_mlp) + &scale_mlp) + &shift_mlp;
        let h = self.ff.forward(&h);
        let audio = audio + &gate_mlp * h;

        // V2A: audio attends to video
        let ss = &self.scale_shift_table;

        // A2V: video attends to audio
        let h_a2v = self.norm_a2v.forward(video);
        let h_a2v = self.a2v_attn.forward(&h_a2v, Some(&audio), mask, None);
        let shift_a2v = ss.narrow(0, 0, 1).unsqueeze(0).unsqueeze(0); // [1, 1, dim]
        let scale_a2v = ss.narrow(0, 1, 1).unsqueeze(0).unsqueeze(0);
        let gate_a2v = ss.narrow(0, 2, 1).unsqueeze(0).unsqueeze(0);
        let video = video + gate_a2v * (h_a2v * (Tensor::ones_like(&scale_a2v) + &scale_a2v) + &shift_a2v);

        // V2A: audio attends to video
        let h_v2a = self.norm_v2a.forward(&audio);
        let h_v2a = self.v2a_attn.forward(&h_v2a, Some(&video), mask, None);
        let shift_v2a = ss.narrow(0, 3, 1).unsqueeze(0).unsqueeze(0);
        let gate_v2a = ss.narrow(0, 4, 1).unsqueeze(0).unsqueeze(0);
        let audio = audio + gate_v2a * (h_v2a * (Tensor::ones_like(&shift_v2a) + &shift_v2a) + shift_v2a);

        (video, audio)
    }

    fn adaln_forward(&self, timestep: &Tensor, kind: tch::Kind) -> (Tensor, Tensor) {
        self.adaln.forward(timestep, kind)
    }
}

pub struct BasicAVTransformerBlock {
    adaln: AdaLayerNormSingle,
    self_attn: TransformerAttention,
    cross_attn: TransformerAttention,
    norm1: RMSNorm,
    norm_cross: RMSNorm,
    norm2: RMSNorm,
    ff: FeedForward,
    /// Optional audio modality components.
    pub audio: Option<AudioModality>,
}

impl BasicAVTransformerBlock {
    pub fn new(
        vs: &tch::nn::Path,
        dim: i64,
        num_heads: i64,
        head_dim: i64,
        context_dim: Option<i64>,
        rope_type: RopeType,
    ) -> Self {
        let adaln =
            AdaLayerNormSingle::new_with_input_dim(&(vs / "adaln"), dim, DEFAULT_SINUSOIDAL_DIM);
        let self_attn = TransformerAttention::new(
            &(vs / "self_attn"),
            dim,
            num_heads,
            head_dim,
            None,
            rope_type,
        );
        let cross_attn = TransformerAttention::new(
            &(vs / "cross_attn"),
            dim,
            num_heads,
            head_dim,
            context_dim,
            rope_type,
        );
        let norm1 = RMSNorm::default_eps_with_path(vs / "norm1", dim);
        let norm_cross = RMSNorm::default_eps_with_path(vs / "norm_cross", dim);
        let norm2 = RMSNorm::default_eps_with_path(vs / "norm2", dim);
        let ff = FeedForward::new(&(vs / "ff"), dim);

        // Audio modality is created via new_with_audio() only.
        // Plain new() is video-only for backward compatibility.
        let audio = None;

        Self { adaln, self_attn, cross_attn, norm1, norm_cross, norm2, ff, audio }
    }

    /// Create a block with explicit audio modality control.
    pub fn new_with_audio(
        vs: &tch::nn::Path,
        dim: i64,
        num_heads: i64,
        head_dim: i64,
        context_dim: Option<i64>,
        rope_type: RopeType,
        enable_audio: bool,
    ) -> Self {
        let adaln =
            AdaLayerNormSingle::new_with_input_dim(&(vs / "adaln"), dim, DEFAULT_SINUSOIDAL_DIM);
        let self_attn = TransformerAttention::new(
            &(vs / "self_attn"), dim, num_heads, head_dim, None, rope_type,
        );
        let cross_attn = TransformerAttention::new(
            &(vs / "cross_attn"), dim, num_heads, head_dim, context_dim, rope_type,
        );
        let norm1 = RMSNorm::default_eps_with_path(vs / "norm1", dim);
        let norm_cross = RMSNorm::default_eps_with_path(vs / "norm_cross", dim);
        let norm2 = RMSNorm::default_eps_with_path(vs / "norm2", dim);
        let ff = FeedForward::new(&(vs / "ff"), dim);

        let audio = if enable_audio {
            Some(AudioModality::new(vs, dim, num_heads, head_dim, context_dim, rope_type))
        } else {
            None
        };

        Self { adaln, self_attn, cross_attn, norm1, norm_cross, norm2, ff, audio }
    }

    pub fn forward(
        &self,
        x: &Tensor,
        timestep: &Tensor,
        context: &Tensor,
        mask: Option<&Tensor>,
        pe: Option<(&Tensor, &Tensor)>,
    ) -> Tensor {
        let (modulation, _) = self.adaln.forward(timestep, x.kind());
        let chunks: Vec<Tensor> = modulation.chunk(6, -1);
        let (shift_msa, scale_msa, gate_msa) = (
            chunks[0].unsqueeze(1), chunks[1].unsqueeze(1), chunks[2].unsqueeze(1),
        );
        let (shift_mlp, scale_mlp, gate_mlp) = (
            chunks[3].unsqueeze(1), chunks[4].unsqueeze(1), chunks[5].unsqueeze(1),
        );

        let h = self.norm1.forward(x) * (Tensor::ones_like(&scale_msa) + &scale_msa) + &shift_msa;
        let h = self.self_attn.forward(&h, None, mask, pe);
        let x = x + &gate_msa * h;

        let h = self.norm_cross.forward(&x);
        let h = self.cross_attn.forward(&h, Some(context), mask, None);
        let x = x + h;

        let h = self.norm2.forward(&x) * (Tensor::ones_like(&scale_mlp) + &scale_mlp) + &shift_mlp;
        let h = self.ff.forward(&h);
        x + &gate_mlp * h
    }

    /// Forward with audio modality. Returns `(video_out, audio_out)`.
    pub fn forward_av(
        &self,
        video: &Tensor,
        audio: &Tensor,
        timestep: &Tensor,
        context: &Tensor,
        mask: Option<&Tensor>,
        pe: Option<(&Tensor, &Tensor)>,
    ) -> (Tensor, Tensor) {
        let video_out = self.forward(video, timestep, context, mask, pe);
        if let Some(ref audio_mod) = self.audio {
            audio_mod.forward(audio, &video_out, timestep, context, mask, pe)
        } else {
            (video_out, audio.shallow_clone())
        }
    }
}

impl std::fmt::Debug for BasicAVTransformerBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BasicAVTransformerBlock")
            .finish_non_exhaustive()
    }
}
