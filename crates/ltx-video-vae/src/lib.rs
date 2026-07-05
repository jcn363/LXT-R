//! Video VAE (Variational Autoencoder) for the LTX-2.3 Rust rewrite.
//!
//! Provides VideoEncoder and VideoDecoder for converting between
//! pixel-space and latent-space video representations.

pub mod configurator;
pub mod decoder_blocks;
pub mod encoder_blocks;
pub mod sampling;

use tch::nn::{ModuleT, Path};
use tch::Tensor;

use ltx_conv::make_conv_nd;
use ltx_timestep::get_timestep_embedding;
use ltx_types::NormLayerType;

use sampling::{depth_to_space, space_to_depth};

pub use configurator::default_encoder_block_descs;

use encoder_blocks::EncoderStage;

// ---------------------------------------------------------------------------
// VideoEncoder — matches Python LTX-Video checkpoint architecture
// ---------------------------------------------------------------------------

/// Video VAE encoder: pixel-space `(B,3,T,H,W)` → latent distribution.
///
/// Architecture (from `ltx-video-2b-v0.9.1.safetensors`):
/// - `space_to_depth(r=4)`: 3 → 48 channels
/// - `conv_in`: 48 → 128
/// - 10 heterogeneous `down_blocks`:
///   - ResBlock stages (0, 3, 6, 8, 9)
///   - Stride-2 convs (1, 4, 7)
///   - Channel-change downsamples (2, 5)
/// - `conv_out`: 512 → 129 (mean 64 + logvar 64 + scale 1)
/// - No mid block, no conv_norm_out, no timestep conditioning
pub struct VideoEncoder {
    conv_in: Box<dyn ModuleT>,
    blocks: Vec<EncoderStage>,
    conv_out: Box<dyn ModuleT>,
}

impl std::fmt::Debug for VideoEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VideoEncoder").finish()
    }
}

/// Descriptor for one encoder down-block, used by the config.
pub struct EncoderBlockDesc {
    pub kind: EncoderBlockKind,
    pub in_ch: i64,
    pub out_ch: i64,
}

pub enum EncoderBlockKind {
    ResBlocks(i64),           // num resblocks
    DownsampleConv,           // stride-2 conv
    ChannelChangeDownsample,  // stride-2 + channel doubling + shortcut + norm
}

impl VideoEncoder {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vs: &tch::nn::Path,
        conv_in_channels: i64,   // 48 (from space_to_depth r=4)
        base_channels: i64,      // 128
        block_descs: &[EncoderBlockDesc],
        conv_out_channels: i64,  // 129
        norm_type: NormLayerType,
        norm_groups: i64,
        causal: bool,
    ) -> Self {
        let conv_in = make_conv_nd(
            vs / "conv_in",
            3,
            conv_in_channels,
            base_channels,
            3, 1, 1, causal, "zeros",
        );

        let mut blocks = Vec::new();
        for (i, desc) in block_descs.iter().enumerate() {
            let block_vs = vs / format!("down_blocks.{i}");
            let stage = match &desc.kind {
                EncoderBlockKind::ResBlocks(n) => {
                    EncoderStage::ResBlocks(
                        encoder_blocks::make_resblock_stage(
                            &block_vs, desc.out_ch, *n,
                            norm_type, norm_groups, causal,
                        )
                    )
                }
                EncoderBlockKind::DownsampleConv => {
                    EncoderStage::DownsampleConv(
                        encoder_blocks::make_downsample_conv(
                            &block_vs, desc.in_ch, causal,
                        )
                    )
                }
                EncoderBlockKind::ChannelChangeDownsample => {
                    EncoderStage::ChannelChange(
                        encoder_blocks::ChannelChangeDownsample::new(
                            &block_vs, desc.in_ch, desc.out_ch,
                            norm_type, norm_groups, causal,
                        )
                    )
                }
            };
            blocks.push(stage);
        }

        // conv_out: last block's output channels → conv_out_channels
        let last_out = block_descs.last().map(|d| d.out_ch).unwrap_or(base_channels);
        let conv_out = make_conv_nd(
            vs / "conv_out",
            3,
            last_out,
            conv_out_channels,
            3, 1, 1, causal, "zeros",
        );

        Self { conv_in, blocks, conv_out }
    }

    /// Encode pixel-space video to distribution parameters.
    ///
    /// Returns raw conv_out output of shape `(B, 129, T', H', W')`.
    /// Callers split into mean/logvar/scale as needed.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let x = space_to_depth(x, 4);
        let mut h = self.conv_in.forward_t(&x, false);
        for block in &self.blocks {
            h = block.forward(&h);
        }
        self.conv_out.forward_t(&h, false)
    }

    /// Encode and sample: returns `(B, 128, T', H', W')` latent.
    ///
    /// Splits the 129-channel output into mean(64) + logvar(64) + scale(1),
    /// then reparameterizes: `latent = mean + exp(0.5 * logvar) * noise`.
    /// The scale channel is discarded.
    pub fn encode(&self, x: &Tensor) -> Tensor {
        let raw = self.forward(x);
        let mean = raw.narrow(1, 0, 64);
        let logvar = raw.narrow(1, 64, 64);
        let std = (logvar * 0.5).exp();
        let noise = Tensor::randn_like(&mean);
        mean + std * noise
    }

    /// Encode and return the mean only (deterministic, for img2img).
    pub fn encode_mean(&self, x: &Tensor) -> Tensor {
        let raw = self.forward(x);
        raw.narrow(1, 0, 64)
    }
}

impl ModuleT for VideoEncoder {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        self.forward(xs)
    }
}

// ---------------------------------------------------------------------------
// VideoDecoder — matches Python LTX-Video checkpoint architecture
// ---------------------------------------------------------------------------

use decoder_blocks::{DecoderResBlock, ConvUpsample, TimestepEmbedding};

/// Per-resblock data: the resblock itself + its scale_shift_table parameter.
struct ResBlockWithMod {
    block: DecoderResBlock,
    scale_shift_table: Tensor, // [4, C]
}

/// Per-stage data: time_embedder + resblocks + scale_shift_tables.
struct ResBlockStage {
    time_embedder: TimestepEmbedding,
    resblocks: Vec<ResBlockWithMod>,
}

/// Video VAE decoder with timestep conditioning.
///
/// Architecture (from `ltx-video-2b-v0.9.1.safetensors`):
/// - `conv_in`: 128 → 1024
/// - 7 up_blocks (alternating ResBlock stages + ConvUpsample):
///   - Block 0: 8 DecoderResBlocks (ch=1024) + time_embedder
///   - Block 1: ConvUpsample 1024 → 4096
///   - Block 2: 7 DecoderResBlocks (ch=512) + time_embedder
///   - Block 3: ConvUpsample 512 → 2048
///   - Block 4: 6 DecoderResBlocks (ch=256) + time_embedder
///   - Block 5: ConvUpsample 256 → 1024
///   - Block 6: 5 DecoderResBlocks (ch=128) + time_embedder
/// - `last_time_embedder` + `last_scale_shift_table`
/// - `conv_out`: 128 → 48 → depth_to_space(r=4) → 3 RGB
pub struct VideoDecoder {
    conv_in: Box<dyn ModuleT>,
    // Timestep
    timestep_scale_multiplier: Tensor,
    sinusoidal_dim: i64,
    // 4 ResBlock stages (blocks 0, 2, 4, 6)
    stages: Vec<ResBlockStage>,
    // 3 ConvUpsample blocks (blocks 1, 3, 5)
    conv_upsamples: Vec<ConvUpsample>,
    // Final timestep components
    last_time_embedder: TimestepEmbedding,
    last_scale_shift_table: Tensor, // [2, 128]
    conv_out: Box<dyn ModuleT>,
}

impl std::fmt::Debug for VideoDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VideoDecoder").finish()
    }
}

impl VideoDecoder {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vs: &Path,
        in_channels: i64,       // 128 (sampled latent)
        base_channels: i64,     // 1024
        norm_type: NormLayerType,
        norm_groups: i64,
        causal: bool,
    ) -> Self {
        let conv_in = make_conv_nd(vs / "conv_in", 3, in_channels, base_channels, 3, 1, 1, causal, "zeros");

        let timestep_scale_multiplier = vs.var("timestep_scale_multiplier", &[], tch::nn::init::Init::Const(1.0));

        let sinusoidal_dim = 256i64;

        // Build 4 ResBlock stages
        let stage_descs: &[(usize, i64)] = &[(0, 1024), (2, 512), (4, 256), (6, 128)];
        let resblock_counts: &[i64] = &[8, 7, 6, 5];
        let time_embed_out_dims: &[i64] = &[4096, 2048, 1024, 512]; // 4*C

        let mut stages = Vec::new();
        for (s, (&(block_idx, ch), &n_res)) in stage_descs.iter().zip(resblock_counts.iter()).enumerate() {
            let te = TimestepEmbedding::new(
                &(vs / format!("up_blocks.{block_idx}")),
                sinusoidal_dim,
                time_embed_out_dims[s],
            );
            let mut resblocks = Vec::new();
            for j in 0..n_res {
                let rb = DecoderResBlock::new(
                    &(vs / format!("up_blocks.{block_idx}.res_blocks.{j}")),
                    ch, norm_type, norm_groups, causal,
                );
                let ss = vs.var(
                    &format!("up_blocks.{block_idx}.res_blocks.{j}.scale_shift_table"),
                    &[4, ch],
                    tch::nn::init::Init::Const(0.0),
                );
                resblocks.push(ResBlockWithMod { block: rb, scale_shift_table: ss });
            }
            stages.push(ResBlockStage { time_embedder: te, resblocks });
        }

        // Build 3 ConvUpsample blocks
        let conv_descs: &[(usize, i64, i64)] = &[(1, 1024, 4096), (3, 512, 2048), (5, 256, 1024)];
        let conv_upsamples: Vec<ConvUpsample> = conv_descs.iter()
            .map(|&(block_idx, in_ch, out_ch)| {
                ConvUpsample::new(
                    &(vs / format!("up_blocks.{block_idx}")),
                    in_ch, out_ch, norm_type, norm_groups, causal,
                )
            })
            .collect();

        let last_time_embedder = TimestepEmbedding::new(vs, sinusoidal_dim, 128);
        let last_scale_shift_table = vs.var("last_scale_shift_table", &[2, 128], tch::nn::init::Init::Const(0.0));
        let conv_out = make_conv_nd(vs / "conv_out", 3, 128, 48, 3, 1, 1, causal, "zeros");

        Self {
            conv_in,
            timestep_scale_multiplier,
            sinusoidal_dim,
            stages,
            conv_upsamples,
            last_time_embedder,
            last_scale_shift_table,
            conv_out,
        }
    }

    /// Decode latent to pixel space with timestep conditioning.
    ///
    /// `x`: `[B, 128, T, H, W]` latent
    /// `timestep`: scalar diffusion timestep
    pub fn forward(&self, x: &Tensor, timestep: &Tensor) -> Tensor {
        let t = get_timestep_embedding(timestep, self.sinusoidal_dim, 10000) * &self.timestep_scale_multiplier;
        let mut h = self.conv_in.forward_t(x, false);

        let mut stage_idx = 0;
        for block_idx in 0..7 {
            if block_idx % 2 == 0 {
                // ResBlock stage with timestep conditioning
                let stage = &self.stages[stage_idx];
                let ch = stage.resblocks[0].scale_shift_table.size()[1];

                // Embed timestep: [B, 256] -> [B, 4*C] -> [B, 4, C]
                let t_emb = stage.time_embedder.forward(&t);
                let t_emb = t_emb.reshape([t_emb.size()[0], 4, ch]);

                for rb in &stage.resblocks {
                    // modulated = scale_shift_table [4, C] + t_emb [B, 4, C]
                    let modulated = &rb.scale_shift_table + &t_emb; // [B, 4, C]
                    h = rb.block.forward_modulated(&h, &modulated);
                }
                stage_idx += 1;
            } else {
                // ConvUpsample
                let cu_idx = (block_idx - 1) / 2;
                h = self.conv_upsamples[cu_idx].forward(&h);
                h = depth_to_space(&h, 2);
            }
        }

        // last_time_embedder + last_scale_shift_table modulation
        let t_last = self.last_time_embedder.forward(&t);
        let last_mod = &self.last_scale_shift_table + &t_last.unsqueeze(1);
        let chunks = last_mod.chunk(2, 1);
        let shift = &chunks[0];
        let scale = &chunks[1];
        // Apply to the final 128-ch output: we need GroupNorm here
        // But the checkpoint has no norm after the last resblock stage.
        // The modulated norm is applied: h * (1 + scale) + shift
        h = h * (1.0 + scale) + shift;

        let h = h.silu();
        let h = self.conv_out.forward_t(&h, false);
        depth_to_space(&h, 4)
    }
}

impl ModuleT for VideoDecoder {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        self.forward(xs, &Tensor::zeros([], (tch::Kind::Float, xs.device())))
    }
}

// ---------------------------------------------------------------------------
// VideoVAE
// ---------------------------------------------------------------------------

/// Complete Video VAE — encoder + decoder with a spatial downsample factor.
pub struct VideoVAE {
    pub(crate) encoder: VideoEncoder,
    pub(crate) decoder: VideoDecoder,
    pub(crate) spatial_downsample_factor: i64,
}

impl std::fmt::Debug for VideoVAE {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VideoVAE")
            .field("spatial_downsample_factor", &self.spatial_downsample_factor)
            .finish()
    }
}

impl VideoVAE {
    pub fn new_encoder_decoder(
        encoder: VideoEncoder,
        decoder: VideoDecoder,
        spatial_downsample_factor: i64,
    ) -> Self {
        Self { encoder, decoder, spatial_downsample_factor }
    }

    pub fn encode(&self, x: &Tensor) -> Tensor {
        self.encoder.forward(x)
    }

    pub fn decode(&self, x: &Tensor, timestep: &Tensor) -> Tensor {
        self.decoder.forward(x, timestep)
    }

    pub fn forward(&self, x: &Tensor, timestep: &Tensor) -> Tensor {
        let latent = self.encode(x);
        self.decode(&latent, timestep)
    }

    pub fn spatial_downsample_factor(&self) -> i64 {
        self.spatial_downsample_factor
    }
}

impl ModuleT for VideoVAE {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        // Default timestep = 0 for ModuleT trait compatibility
        self.forward(xs, &Tensor::zeros([], (tch::Kind::Float, xs.device())))
    }
}
