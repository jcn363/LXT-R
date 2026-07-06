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

/// Load VAE weights from a safetensors file into a VarStore.
///
/// Maps VarStore paths (`/` separators) to checkpoint paths (`.` separators),
/// handling the Python convention where conv modules store parameters as
/// `conv.weight`/`conv.bias` rather than bare `weight`/`bias`.
pub fn load_vae_weights(vs: &tch::nn::VarStore, path: &str, prefix: &str) -> u32 {
    let data = std::fs::read(path).expect("failed to read weight file");
    let st = safetensors::SafeTensors::deserialize(&data).expect("failed to deserialize");
    let mut loaded = 0u32;
    let _no_grad = tch::no_grad_guard();
    let mut vars = vs.variables();

    for (name, tensor) in vars.iter_mut() {
        let ckpt_name = format!("{prefix}{}", name.replace('/', "."));
        if load_one(&st, &ckpt_name, tensor) {
            loaded += 1;
            continue;
        }
        // Try .conv. wrapper fallback
        if let Some(pos) = ckpt_name.rfind('.') {
            let suffix = &ckpt_name[pos + 1..];
            if suffix == "weight" || suffix == "bias" {
                let base = &ckpt_name[..pos];
                let wrapped = format!("{base}.conv.{suffix}");
                if load_one(&st, &wrapped, tensor) {
                    loaded += 1;
                    continue;
                }
            }
        }
        // Fallback: per_channel_statistics path uses hyphens in checkpoint
        if name.contains("per_channel_statistics") {
            // VarStore: {encoder|decoder}.per_channel_statistics/mean_of_means
            // Checkpoint: vae.per_channel_statistics.mean-of-means
            let stripped = name.trim_start_matches("encoder.").trim_start_matches("decoder.")
                               .trim_start_matches("encoder/").trim_start_matches("decoder/");
            let alt = stripped.replace("mean_of_means", "mean-of-means")
                              .replace("std_of_means", "std-of-means")
                              .replace('/', ".");
            let alt_ckpt = format!("{prefix}{alt}");
            if load_one(&st, &alt_ckpt, tensor) {
                loaded += 1;
                continue;
            }
        }
        eprintln!("UNMATCHED: {name} (looked for {ckpt_name})");
    }
    loaded
}

fn load_one(st: &safetensors::SafeTensors, key: &str, tensor: &mut tch::Tensor) -> bool {
    if let Ok(view) = st.tensor(key) {
        let kind = match view.dtype() {
            safetensors::Dtype::F16 => tch::Kind::Half,
            safetensors::Dtype::BF16 => tch::Kind::BFloat16,
            _ => tch::Kind::Float,
        };
        let shape: Vec<i64> = view.shape().iter().map(|&s| s as i64).collect();
        let loaded = tch::Tensor::from_data_size(view.data(), &shape, kind);
        if tensor.size() == loaded.size() {
            tensor.copy_(&loaded);
            return true;
        }
    }
    false
}

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
    // Per-channel latent normalization
    pc_mean: Tensor, // [128]
    pc_std: Tensor,  // [128]
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
            let block_vs = vs / format!("down_blocks/{i}");
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
                        encoder_blocks::DownsampleConv::new_block(&block_vs, desc.in_ch)
                    )
                }
                EncoderBlockKind::ChannelChangeDownsample => {
                    EncoderStage::ChannelChange(
                        encoder_blocks::ChannelChangeDownsample::new(
                            &block_vs, desc.in_ch, desc.out_ch,
                            norm_type, norm_groups,
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

        // Per-channel latent normalization
        let pc_mean = vs.var("per_channel_statistics/mean_of_means", &[128], tch::nn::init::Init::Const(0.0));
        let pc_std = vs.var("per_channel_statistics/std_of_means", &[128], tch::nn::init::Init::Const(1.0));

        Self { conv_in, blocks, conv_out, pc_mean, pc_std }
    }

    /// Encode pixel-space video to distribution parameters.
    ///
    /// Returns raw conv_out output of shape `(B, 129, T', H', W')`.
    /// Callers split into mean/logvar/scale as needed.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let x = space_to_depth(x, 4);
        let mut h = self.conv_in.forward_t(&x, false);

        for stage in &self.blocks {
            h = stage.forward(&h);
        }
        self.conv_out.forward_t(&h, false)
    }

    /// Encode and sample: returns `(B, 128, T', H', W')` latent.
    ///
    /// Splits the 129-channel output into mean(64) + logvar(64) + scale(1),
    /// then reparameterizes: `latent = mean + exp(0.5 * logvar) * noise`.
    pub fn encode(&self, x: &Tensor) -> Tensor {
        let raw = self.forward(x);
        let mean = raw.narrow(1, 0, 64);
        let logvar = raw.narrow(1, 64, 64);
        let std = (logvar * 0.5).exp();
        let noise = Tensor::randn_like(&mean);
        mean + std * noise
    }

    /// Encode and return the full latent (deterministic, for img2img).
    ///
    /// Returns the first 128 of 129 conv_out channels, then applies
    /// per-channel normalization: `(x - mean) / std`.
    pub fn encode_mean(&self, x: &Tensor) -> Tensor {
        let raw = self.forward(x);
        let means = raw.narrow(1, 0, 128);
        // Normalize: (x - mean) / std
        (&means - &self.pc_mean.view([1, -1, 1, 1, 1])) / &self.pc_std.view([1, -1, 1, 1, 1])
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

use decoder_blocks::{DecoderResBlock, CompressAllUpsample, TimestepEmbedding};

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
/// Architecture (from checkpoint metadata config):
/// - `conv_in`: 128 → 1024
/// - 7 up_blocks (alternating ResBlock stages + CompressAllUpsample):
///   - Block 0: 8 ResBlocks (ch=1024) + time_embedder, no noise injection
///   - Block 1: CompressAll 1024 → 512 (conv 1024→4096, depth_to_space_3d r=2)
///   - Block 2: 7 ResBlocks (ch=512) + time_embedder, noise injection
///   - Block 3: CompressAll 512 → 256 (conv 512→2048, depth_to_space_3d r=2)
///   - Block 4: 6 ResBlocks (ch=256) + time_embedder, noise injection
///   - Block 5: CompressAll 256 → 128 (conv 256→1024, depth_to_space_3d r=2)
///   - Block 6: 5 ResBlocks (ch=128) + time_embedder, noise injection
/// - `last_time_embedder` + `last_scale_shift_table`
/// - `conv_out`: 128 → 48 → depth_to_space(r=4) → 3 RGB
pub struct VideoDecoder {
    conv_in: Box<dyn ModuleT>,
    timestep_scale_multiplier: Tensor,
    sinusoidal_dim: i64,
    stages: Vec<ResBlockStage>,
    compress_upsamples: Vec<CompressAllUpsample>,
    last_time_embedder: TimestepEmbedding,
    last_scale_shift_table: Tensor, // [2, 128]
    conv_out: Box<dyn ModuleT>,
    // Per-channel latent normalization
    pc_mean: Tensor, // [128]
    pc_std: Tensor,  // [128]
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
        in_channels: i64,       // 128
        base_channels: i64,     // 1024
        norm_type: NormLayerType,
        norm_groups: i64,
        causal: bool,
    ) -> Self {
        let conv_in = make_conv_nd(vs / "conv_in", 3, in_channels, base_channels, 3, 1, 1, causal, "zeros");
        let timestep_scale_multiplier = vs.var("timestep_scale_multiplier", &[], tch::nn::init::Init::Const(1.0));
        let sinusoidal_dim = 256i64;

        // ResBlock stages: block_idx, channels, num_resblocks, inject_noise
        let stage_params: &[(usize, i64, i64, bool)] = &[
            (0, 1024, 8, false),
            (2, 512, 7, true),
            (4, 256, 6, true),
            (6, 128, 5, true),
        ];

        let mut stages = Vec::new();
        for &(block_idx, ch, n_res, inject_noise) in stage_params {
            let te = TimestepEmbedding::new(
                &(vs / format!("up_blocks/{block_idx}/time_embedder")),
                sinusoidal_dim,
                4 * ch,
            );
            let mut resblocks = Vec::new();
            for j in 0..n_res {
                let rb = DecoderResBlock::new(
                    &(vs / format!("up_blocks/{block_idx}/res_blocks/{j}")),
                    ch, norm_type, norm_groups, causal, inject_noise,
                );
                let ss = vs.var(
                    &format!("up_blocks/{block_idx}/res_blocks/{j}/scale_shift_table"),
                    &[4, ch],
                    tch::nn::init::Init::Const(0.0),
                );
                resblocks.push(ResBlockWithMod { block: rb, scale_shift_table: ss });
            }
            stages.push(ResBlockStage { time_embedder: te, resblocks });
        }

        // CompressAllUpsample blocks: block_idx, in_channels, multiplier, residual
        let compress_params: &[(usize, i64, i64, bool)] = &[
            (1, 1024, 2, true),
            (3, 512, 2, true),
            (5, 256, 2, true),
        ];
        let compress_upsamples: Vec<CompressAllUpsample> = compress_params.iter()
            .map(|&(block_idx, in_ch, mult, res)| {
                CompressAllUpsample::new(
                    &(vs / format!("up_blocks/{block_idx}")),
                    in_ch, mult, causal, res,
                )
            })
            .collect();

        let last_time_embedder = TimestepEmbedding::new(&(vs / "last_time_embedder"), sinusoidal_dim, sinusoidal_dim);
        let last_scale_shift_table = vs.var("last_scale_shift_table", &[2, 128], tch::nn::init::Init::Const(0.0));
        let conv_out = make_conv_nd(vs / "conv_out", 3, 128, 48, 3, 1, 1, causal, "zeros");

        // Per-channel latent normalization (denormalize before decoding)
        let pc_mean = vs.var("per_channel_statistics/mean_of_means", &[128], tch::nn::init::Init::Const(0.0));
        let pc_std = vs.var("per_channel_statistics/std_of_means", &[128], tch::nn::init::Init::Const(1.0));

        Self {
            conv_in,
            timestep_scale_multiplier,
            sinusoidal_dim,
            stages,
            compress_upsamples,
            last_time_embedder,
            last_scale_shift_table,
            conv_out,
            pc_mean,
            pc_std,
        }
    }

    /// Decode latent to pixel space with timestep conditioning.
    pub fn forward(&self, x: &Tensor, timestep: &Tensor) -> Tensor {
        // Denormalize latent: (x * std) + mean
        let x = x * &self.pc_std.view([1, -1, 1, 1, 1]) + &self.pc_mean.view([1, -1, 1, 1, 1]);
        // Scale timestep BEFORE sinusoidal embedding (matches Python: scaled_timestep = timestep * multiplier)
        let scaled_t = timestep * &self.timestep_scale_multiplier;
        let t = get_timestep_embedding(&scaled_t, self.sinusoidal_dim, 10000);
        let mut h = self.conv_in.forward_t(&x, false);

        let mut stage_idx = 0;
        let mut compress_idx = 0;
        for block_idx in 0..7 {
            if block_idx % 2 == 0 {
                // ResBlock stage with timestep conditioning
                let stage = &self.stages[stage_idx];
                let ch = stage.resblocks[0].scale_shift_table.size()[1];

                let t_emb = stage.time_embedder.forward(&t);
                let t_emb = t_emb.reshape([t_emb.size()[0], 4, ch]);

                for rb in &stage.resblocks {
                    let modulated = &rb.scale_shift_table + &t_emb;
                    h = rb.block.forward_modulated(&h, &modulated);
                }
                stage_idx += 1;
            } else {
                h = self.compress_upsamples[compress_idx].forward(&h);
                compress_idx += 1;
            }
        }

        // Final timestep modulation
        let t_last = self.last_time_embedder.forward(&t); // [B, 256]
        let t_last = t_last.reshape([t_last.size()[0], 2, 128]); // [B, 2, 128]
        let last_mod = &self.last_scale_shift_table + &t_last; // [B, 2, 128]
        let bsz = last_mod.size()[0];
        let c = last_mod.size()[2];
        let flat = last_mod.reshape([bsz * 2 * c]);
        let shift = flat.narrow(0, 0, c).reshape([bsz, c, 1, 1, 1]);
        let scale = flat.narrow(0, c, c).reshape([bsz, c, 1, 1, 1]);
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
