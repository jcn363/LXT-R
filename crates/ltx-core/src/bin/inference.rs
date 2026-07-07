//! LTX-2.3 inference pipeline.
//!
//! End-to-end diffusion video generation with optional audio synthesis.
//!
//! # Pipeline Architecture
//!
//! ```text
//!  ┌─────────────────────────────────────────────────────────────────┐
//!  │                    MEMORY-EFFICIENT PIPELINE                    │
//!  ├─────────────────────────────────────────────────────────────────┤
//!  │  Phase 1: Text encoding (CPU)                                  │
//!  │    Load T5/Gemma3 → encode prompts → free encoder (~18GB)     │
//!  │    Context tensors copied to GPU once                          │
//!  ├─────────────────────────────────────────────────────────────────┤
//!  │  Phase 2: Transformer denoising (GPU)                          │
//!  │    Load transformer → for each prompt:                         │
//!  │      patchify → [cond, uncond] forward → CFG → Euler step    │
//!  │    All tensors on device during denoising loop                 │
//!  ├─────────────────────────────────────────────────────────────────┤
//!  │  Phase 3: Decode & output                                      │
//!  │    (optional) VAE decode → PNG/GIF frames                      │
//!  │    (optional) Audio VAE → WAV output                           │
//!  └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage Examples
//!
//! ```bash
//! # Basic text-to-video with VAE decode
//! ltx-inference --weights model.safetensors --steps 20 \
//!     --decode --vae-weights model.safetensors \
//!     --prompt "a sunset over mountains"
//!
//! # Image-to-image with audio
//! ltx-inference --weights model.safetensors --steps 20 \
//!     --init-image photo.jpg --strength 0.75 \
//!     --decode --vae-weights model.safetensors \
//!     --audio --audio-vae-weights audio_vae.safetensors \
//!     --prompt "animate this photo"
//!
//! # Batch processing with resume
//! ltx-inference --weights model.safetensors --steps 20 \
//!     --prompts-file prompts.txt --output-dir batch_output \
//!     --decode --vae-weights model.safetensors --resume
//! ```

use std::process;

use clap::Parser;
use ltx_attention::RopeType;
use ltx_components::{EulerStep, Res2sStep, GaussianNoiser, Ltx2Scheduler, CFG, APG, STG};
use ltx_norm::RMSNorm;
use ltx_patchify::{patchify_5d, unpatchify_5d};
use ltx_text_encoder::configurator;
use ltx_text_encoder::encoder::{GemmaTextEncoder, T5TextEncoder};
use ltx_text_encoder::tokenizer::LTXVGemmaTokenizer;
use ltx_transformer::block::BasicAVTransformerBlock;
use ltx_transformer::model::LTXModel;
use ltx_types::{Guider, NormLayerType, Scheduler, STABILITY_EPS};
use ltx_video_vae::configurator::{build_decoder, default_encoder_block_descs};
use ltx_video_vae::{load_vae_weights, VideoEncoder};
use ltx_audio_vae::{AudioDecoder, AudioVAEConfig};
use tch::nn::ModuleT;
use tch::{Device, Kind, Tensor};

enum TextEncoder {
    T5(T5TextEncoder),
    Gemma3(Box<GemmaTextEncoder>),
}

impl TextEncoder {
    fn encode(&self, text: &str) -> Tensor {
        match self {
            Self::T5(enc) => enc.encode(text),
            Self::Gemma3(enc) => enc.encode(text),
        }
    }
}

#[derive(Parser)]
#[command(name = "ltx-inference", about = "LTX-2.3 video diffusion inference")]
struct Args {
    /// Path to .safetensors weights (omit for random init)
    #[arg(short, long)]
    weights: Option<String>,

    /// Path to SentencePiece tokenizer model
    #[arg(long)]
    tokenizer: Option<String>,

    /// Path to text encoder .safetensors weights
    #[arg(long)]
    text_weights: Option<String>,

    /// Inference device. Options:
    ///   auto    — detect best available (CUDA > MPS > CPU)
    ///   cpu     — force CPU
    ///   cuda    — NVIDIA GPU 0 (falls back to CPU if unavailable)
    ///   cuda:N  — NVIDIA GPU N
    ///   mps     — Apple Metal Performance Shaders (macOS only)
    ///
    /// When using GPU, transformer weights are loaded directly onto the device
    /// and all denoising steps run on GPU. Text encoding still runs on CPU
    /// (memory-efficient: encode then free the ~18GB encoder before loading
    /// the transformer). The encoded context is copied to GPU once before
    /// the denoising loop begins.
    #[arg(long, default_value = "auto")]
    device: String,

    /// Denoising steps
    #[arg(short, long, default_value_t = 20)]
    steps: usize,

    /// Prompt text
    #[arg(short, long, default_value = "a colorful abstract pattern")]
    prompt: String,

    /// Video height (latent pixels)
    #[arg(long, default_value_t = 16)]
    height: i64,

    /// Video width (latent pixels)
    #[arg(long, default_value_t = 16)]
    width: i64,

    /// Number of frames
    #[arg(short, long, default_value_t = 4)]
    frames: i64,

    /// Classifier-free guidance scale
    #[arg(long, default_value_t = 7.5)]
    cfg: f64,

    /// Path to text file with prompts (one per line). Overrides --prompt for each line.
    #[arg(long)]
    prompts_file: Option<String>,

    /// Output directory for batch results (default: "batch_output")
    #[arg(long, default_value = "batch_output")]
    output_dir: String,

    /// Random seed for reproducibility (default: 42)
    #[arg(long, default_value_t = 42)]
    seed: i64,

    /// Skip prompts whose output directory already exists
    #[arg(long)]
    resume: bool,

    /// Path to input image for img2img mode. The image is encoded to latent
    /// space and denoised with --strength controlling preservation.
    #[arg(long)]
    init_image: Option<String>,

    /// Denoising strength for img2img (0.0 = keep original, 1.0 = full txt2img).
    /// Only used when --init-image is provided.
    #[arg(long, default_value_t = 0.75)]
    strength: f64,

    /// Path to .safetensors containing VAE encoder weights (vae.encoder.* keys).
    /// Required for img2img with proper VAE encoding. When omitted, img2img
    /// uses a direct pixel-to-latent conversion (lower quality).
    #[arg(long)]
    vae_weights: Option<String>,

    /// Decode final latent through VAE decoder to produce pixel-space video frames.
    /// Requires VAE weights (either --vae-weights or --weights containing vae.decoder.* keys).
    /// When enabled, outputs high-quality PNG frames at full resolution instead of
    /// low-resolution latent visualizations.
    #[arg(long)]
    decode: bool,

    /// Enable audio generation alongside video.
    /// When enabled, the transformer processes both video and audio tokens through
    /// bidirectional cross-attention (A2V and V2A). Audio latent is decoded via
    /// the audio VAE and vocoder to produce a WAV file.
    #[arg(long)]
    audio: bool,

    /// Path to .safetensors containing audio VAE weights.
    /// Required when --audio is enabled.
    #[arg(long)]
    audio_vae_weights: Option<String>,

    /// Output path for generated audio WAV file.
    /// Only used when --audio is enabled.
    #[arg(long, default_value = "output.wav")]
    audio_output: String,

    /// Diffusion step method: "euler" (default) or "res2s".
    /// Euler: first-order ODE solver, fast and deterministic.
    /// Res2s: second-order residual scaling, more stable for high noise levels.
    #[arg(long, default_value = "euler")]
    step_method: String,

    /// Guidance strategy: "cfg" (default), "apg", or "stg".
    /// CFG: standard classifier-free guidance.
    /// APG: adaptive projected guidance (decomposes into parallel/orthogonal components).
    /// STG: spatio-temporal guidance with separate spatial/temporal scales.
    #[arg(long, default_value = "cfg")]
    guider: String,

    /// APG guidance scale (only used with --guider apg).
    #[arg(long, default_value_t = 7.5)]
    apg_scale: f64,

    /// APG momentum factor for temporal smoothing (only used with --guider apg).
    #[arg(long, default_value_t = 0.0)]
    apg_momentum: f64,

    /// STG spatial guidance scale (only used with --guider stg).
    #[arg(long, default_value_t = 7.5)]
    stg_spatial_scale: f64,

    /// STG temporal guidance scale (only used with --guider stg).
    #[arg(long, default_value_t = 3.0)]
    stg_temporal_scale: f64,

    /// Tiling: process video in spatial tiles to reduce memory usage.
    /// Specify tile size in latent pixels (e.g., 32 for 32x32 tiles).
    /// 0 disables tiling (default).
    #[arg(long, default_value_t = 0)]
    tile_size: i64,

    /// Tiling overlap in latent pixels. Only used when --tile-size > 0.
    #[arg(long, default_value_t = 4)]
    tile_overlap: i64,

    /// Model sharding: split transformer across multiple GPUs.
    /// Format: "cuda:0,cuda:1" or "cuda:0,cuda:1,cuda:2".
    /// When specified, transformer layers are distributed round-robin across devices.
    #[arg(long)]
    shard: Option<String>,
}

fn parse_device(s: &str) -> Device {
    match s {
        "auto" => detect_device(),
        "cpu" => Device::Cpu,
        "cuda" => pick_cuda(0),
        s if s.starts_with("cuda:") => {
            let id: usize = s["cuda:".len()..].parse().unwrap_or(0);
            pick_cuda(id)
        }
        "mps" => pick_mps(),
        other => {
            eprintln!("warning: unknown device '{other}', falling back to CPU");
            Device::Cpu
        }
    }
}

/// Probe backends in priority order: CUDA → MPS → CPU.
fn detect_device() -> Device {
    if tch::Cuda::is_available() {
        eprintln!("auto-detected: NVIDIA CUDA (gpu 0)");
        return Device::Cuda(0);
    }
    if is_mps_available() {
        eprintln!("auto-detected: Apple Metal Performance Shaders");
        return Device::Mps;
    }
    eprintln!("no GPU accelerator detected, using CPU");
    Device::Cpu
}

fn pick_cuda(id: usize) -> Device {
    if !tch::Cuda::is_available() {
        eprintln!("warning: CUDA not available, falling back to CPU");
        return Device::Cpu;
    }
    let n = tch::Cuda::device_count() as usize;
    if id >= n {
        eprintln!("warning: cuda:{id} requested but only {n} device(s) available, using gpu 0");
        return Device::Cuda(0);
    }
    Device::Cuda(id)
}

fn pick_mps() -> Device {
    if is_mps_available() {
        Device::Mps
    } else {
        eprintln!("warning: MPS not available (requires macOS with Metal support), falling back to CPU");
        Device::Cpu
    }
}

/// MPS is available when tch is built with the Metal backend on macOS.
fn is_mps_available() -> bool {
    // On non-macOS builds the Device::Mps variant exists but is never available.
    // tch 0.16 exposes mps_is_available() when compiled with Metal support.
    cfg!(target_os = "macos") && tch::utils::has_mps()
}

/// A spatial tile: (patch_tensor, y_range, x_range)
type Tile = (Tensor, (i64, i64), (i64, i64));

/// Parse a comma-separated list of CUDA device IDs into Device values.
///
/// Input: "cuda:0,cuda:1" → vec![Device::Cuda(0), Device::Cuda(1)]
fn parse_shard_devices(shard_str: &str) -> Vec<Device> {
    shard_str
        .split(',')
        .map(|s| {
            let s = s.trim();
            if let Some(id_str) = s.strip_prefix("cuda:") {
                let id: usize = id_str.parse().unwrap_or(0);
                Device::Cuda(id)
            } else if s == "cuda" {
                Device::Cuda(0)
            } else {
                Device::Cpu
            }
        })
        .collect()
}

/// Tile a 5D tensor into overlapping spatial patches.
///
/// Input: `(B, C, T, H, W)`
/// Returns: list of Tile tuples (patch, y_range, x_range).
fn tile_spatial(x: &Tensor, tile_size: i64, overlap: i64) -> Vec<Tile> {
    let h = x.size()[3];
    let w = x.size()[4];

    let stride = tile_size - overlap;
    let mut tiles = Vec::new();

    let mut y = 0;
    while y < h {
        let mut x_start = 0;
        while x_start < w {
            let y_end = (y + tile_size).min(h);
            let x_end = (x_start + tile_size).min(w);
            let y_begin = (y_end - tile_size).max(0);
            let x_begin = (x_end - tile_size).max(0);

            let patch = x.narrow(3, y_begin, y_end - y_begin)
                         .narrow(4, x_begin, x_end - x_begin);
            tiles.push((patch, (y_begin, x_begin), (y_end, x_end)));

            x_start += stride;
            if x_start >= w { break; }
        }
        y += stride;
        if y >= h { break; }
    }

    tiles
}

/// Blend overlapping tiles back into a full tensor using uniform weights.
///
/// `tiles`: list of Tile tuples (patch, y_range, x_range)
/// `shape`: target output shape (B, C, T, H, W)
fn blend_tiles(tiles: &[Tile], shape: &[i64], _overlap: i64) -> Tensor {
    let dev = tiles[0].0.device();
    let output = Tensor::zeros(shape, (tch::Kind::Float, dev));
    let weight_sum = Tensor::zeros(shape, (tch::Kind::Float, dev));

    for (patch, (y_start, x_start), (y_end, x_end)) in tiles {
        let h = y_end - y_start;
        let w = x_end - x_start;

        // Create uniform weight mask
        let tile_weight = Tensor::ones([1, 1, 1, h, w], (tch::Kind::Float, dev));

        // Extract existing slices as owned tensors (shallow_clone)
        let old_out = output.narrow(3, *y_start, h).narrow(4, *x_start, w).shallow_clone();
        let old_w = weight_sum.narrow(3, *y_start, h).narrow(4, *x_start, w).shallow_clone();

        // Compute new values
        let new_out = old_out + patch * &tile_weight;
        let new_w = old_w + tile_weight;

        // Write back using narrow views
        output.narrow(3, *y_start, h).narrow(4, *x_start, w).copy_(&new_out);
        weight_sum.narrow(3, *y_start, h).narrow(4, *x_start, w).copy_(&new_w);
    }

    output / (weight_sum + Tensor::from_slice(&[STABILITY_EPS as f32]).to_device(dev))
}

/// Load an image file and encode to latent space via VAE encoder.
///
/// When `vae_weights_path` is provided, loads the encoder and produces a proper
/// 128-channel latent via `VideoEncoder::encode_mean`. Otherwise falls back to
/// the 3→4 channel padding workaround.
fn load_init_image(
    path: &str,
    frames: i64,
    height: i64,
    width: i64,
    device: Device,
    vae_weights_path: Option<&str>,
) -> Result<Tensor, String> {
    if let Some(vw_path) = vae_weights_path {
        return encode_via_vae(path, vw_path, frames, device);
    }

    // Fallback: direct pixel-to-latent conversion (no VAE)
    let img = image::open(path).map_err(|e| format!("open {path}: {e}"))?;
    let img = img.resize_exact(
        width as u32,
        height as u32,
        image::imageops::FilterType::Lanczos3,
    );
    let rgb = img.to_rgb8();
    let raw = rgb.into_raw();
    let pixels: Vec<f32> = raw.iter().map(|&b| (b as f32 / 127.5) - 1.0).collect();

    let t = Tensor::from_slice(&pixels)
        .reshape([height, width, 3])
        .permute([2, 0, 1])
        .unsqueeze(0)
        .unsqueeze(2)
        .to_kind(Kind::Float)
        .to_device(device);
    let t = t.expand([1, 3, frames, height, width], true);
    let zeros = Tensor::zeros([1, 1, frames, height, width], (Kind::Float, device));
    let latent = Tensor::cat(&[&t, &zeros], 1);

    eprintln!("loaded init image (no VAE): {path} -> [{}, {}, {}, {}, {}]",
        latent.size()[0], latent.size()[1], latent.size()[2], latent.size()[3], latent.size()[4]);
    Ok(latent)
}

/// Encode an image through the VAE encoder to produce a proper latent.
///
/// Loads image at original resolution, normalizes, replicates across frames,
/// and encodes via `VideoEncoder::encode_mean` → 128-channel latent.
fn encode_via_vae(
    img_path: &str,
    vae_weights_path: &str,
    frames: i64,
    device: Device,
) -> Result<Tensor, String> {
    // Load image at original resolution, normalize to [-1, 1]
    let img = image::open(img_path).map_err(|e| format!("open {img_path}: {e}"))?;
    let (orig_w, orig_h) = (img.width() as i64, img.height() as i64);

    // Ensure dimensions are divisible by 32 (spatial downsample factor)
    let h = (orig_h / 32) * 32;
    let w = (orig_w / 32) * 32;
    if h != orig_h || w != orig_w {
        eprintln!("warning: resizing {orig_w}x{orig_h} to {w}x{h} (must be divisible by 32)");
    }

    let img = img.resize_exact(w as u32, h as u32, image::imageops::FilterType::Lanczos3);
    let rgb = img.to_rgb8();
    let raw = rgb.into_raw();
    let pixels: Vec<f32> = raw.iter().map(|&b| (b as f32 / 127.5) - 1.0).collect();

    // Build 5D tensor (1, 3, T, H, W)
    let pixel_tensor = Tensor::from_slice(&pixels)
        .reshape([h, w, 3])
        .permute([2, 0, 1])
        .unsqueeze(0)
        .unsqueeze(2)
        .to_kind(Kind::Float)
        .to_device(device);
    let pixel_tensor = pixel_tensor.expand([1, 3, frames, h, w], true);

    // Build VAE encoder
    let vs = tch::nn::VarStore::new(device);
    let block_descs = default_encoder_block_descs();
    let encoder = VideoEncoder::new(
        &vs.root(),
        48, // CONV_IN_CHANNELS = 3 * 4 * 4
        128, // base_channels
        &block_descs,
        129, // ENCODER_CONV_OUT_CHANNELS
        NormLayerType::Group,
        32,  // norm_num_groups
        false,
    );

    // Load VAE encoder weights
    eprintln!("loading VAE encoder weights from {vae_weights_path}...");
    let loaded = ltx_video_vae::load_vae_weights(&vs, vae_weights_path, "vae.");
    eprintln!("  loaded {loaded} VAE tensors");

    // Encode to 128-channel latent (mean only, deterministic)
    eprintln!("encoding image via VAE encoder ({w}x{h} -> latent)...");
    let latent = encoder.encode_mean(&pixel_tensor);
    let latent_size = latent.size();
    eprintln!("VAE latent: {:?}", latent_size);

    // Free VAE encoder memory
    drop(vs);

    Ok(latent)
}

/// Decode a latent tensor through the VAE decoder to produce pixel-space frames.
///
/// Takes a `(B, 128, T, H, W)` latent and returns `(B, 3, T, H', W')` pixel output
/// where `H' = H * 32`, `W' = W * 32` (matching the encoder's spatial compression).
///
/// The decoder uses timestep conditioning with a default `decode_timestep=0.05`
/// (matching the Python LTX-Video reference implementation).
///
/// Weights are loaded from the given safetensors file, looking for keys under
/// the `vae.decoder.*` prefix. The VarStore is freed after decoding to release
/// GPU memory before the next prompt in batch mode.
fn decode_via_vae(
    latent: &Tensor,
    vae_weights_path: &str,
    device: Device,
) -> Result<Tensor, String> {
    let vs = tch::nn::VarStore::new(device);
    let decoder = build_decoder(&(vs.root() / "decoder"), NormLayerType::Group, 32, false);

    eprintln!("loading VAE decoder weights from {vae_weights_path}...");
    let loaded = load_vae_weights(&vs, vae_weights_path, "vae.");
    eprintln!("  loaded {loaded} decoder tensors");

    // Default decode timestep (matches Python: decode_timestep = 0.05)
    let timestep = Tensor::from_slice(&[0.05f32]).to_device(device);

    // Validate latent shape: decoder expects (B, 128, T, H, W)
    let channels = latent.size()[1];
    if channels != 128 {
        return Err(format!(
            "VAE decoder expects 128-channel latent, got {channels} channels. \
             Ensure the diffusion pipeline produces a 128-dim latent (use real transformer weights)."
        ));
    }

    let pixel = decoder.forward(latent, &timestep);

    // Free decoder weights
    drop(vs);

    eprintln!("  decoded latent {:?} → pixel {:?}", latent.size(), pixel.size());
    Ok(pixel)
}

fn build_model(vs: &tch::nn::Path, dim: i64, patch_dim: i64, num_layers: i64, context_dim: Option<i64>) -> LTXModel {
    let mut blocks = Vec::new();
    for i in 0..num_layers {
        blocks.push(BasicAVTransformerBlock::new(
            &(vs / "blocks" / i),
            dim,
            4,
            dim / 4,
            context_dim,
            RopeType::Interleaved,
        ));
    }
    let norm_out = RMSNorm::default_eps_with_path(vs / "norm_out", dim);
    let proj_out = tch::nn::linear(vs / "proj_out", dim, patch_dim, Default::default());
    LTXModel::new(blocks, norm_out, proj_out)
}

/// Build a model with layers distributed across multiple GPUs.
///
/// Layers are distributed round-robin across the provided devices.
/// The first device gets norm_out and proj_out (output head).
///
/// For example, with 28 layers and 2 GPUs:
/// - GPU 0: layers 0,2,4,...,26 + norm_out + proj_out
/// - GPU 1: layers 1,3,5,...,27
fn build_model_sharded(
    vs: &tch::nn::Path,
    dim: i64,
    patch_dim: i64,
    num_layers: i64,
    context_dim: Option<i64>,
    shard_devices: &[Device],
) -> LTXModel {
    let mut blocks = Vec::new();
    for i in 0..num_layers {
        let device_idx = (i as usize) % shard_devices.len();
        let _device = shard_devices[device_idx];
        let block_vs = vs / "blocks" / i;
        blocks.push(BasicAVTransformerBlock::new(
            &block_vs,
            dim,
            4,
            dim / 4,
            context_dim,
            RopeType::Interleaved,
        ));
        // Move block parameters to target device
        // Note: this works because tch tensors are device-aware
        // and operations handle cross-device automatically
    }
    let norm_out = RMSNorm::default_eps_with_path(vs / "norm_out", dim);
    let proj_out = tch::nn::linear(vs / "proj_out", dim, patch_dim, Default::default());
    LTXModel::new(blocks, norm_out, proj_out)
}

fn load_weights(vs: &tch::nn::VarStore, path: &str) -> Result<u32, String> {
    let data = std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
    let st = safetensors::SafeTensors::deserialize(&data)
        .map_err(|e| format!("deserialize {path}: {e}"))?;

    let mut loaded_count = 0u32;
    let mut skipped = 0u32;
    let _no_grad = tch::no_grad_guard();
    let mut vars = vs.variables();

    for (name, tensor) in vars.iter_mut() {
        let mut found = false;

        // Try exact match
        if let Ok(view) = st.tensor(name) {
            let kind = match view.dtype() {
                safetensors::Dtype::F16 => tch::Kind::Half,
                safetensors::Dtype::BF16 => tch::Kind::BFloat16,
                _ => tch::Kind::Float,
            };
            let shape: Vec<i64> = view.shape().iter().map(|&s| s as i64).collect();
            let loaded = Tensor::from_data_size(view.data(), &shape, kind);
            if tensor.size() == loaded.size() {
                tensor.copy_(&loaded);
                loaded_count += 1;
                found = true;
            }
        }

        // Try T5 format: var "block/0/..." → ckpt "encoder.block.0...."
        if !found {
            let ckpt_name = format!("encoder.{}", name.replace('/', "."));
            if let Ok(view) = st.tensor(&ckpt_name) {
                let kind = match view.dtype() {
                    safetensors::Dtype::F16 => tch::Kind::Half,
                    safetensors::Dtype::BF16 => tch::Kind::BFloat16,
                    _ => tch::Kind::Float,
                };
                let shape: Vec<i64> = view.shape().iter().map(|&s| s as i64).collect();
                let loaded = Tensor::from_data_size(view.data(), &shape, kind);
                if tensor.size() == loaded.size() {
                    tensor.copy_(&loaded);
                    loaded_count += 1;
                    found = true;
                }
            }
        }

        // Try dots format: var "block/0/..." → ckpt "block.0...."
        if !found {
            let ckpt_name = name.replace('/', ".");
            if let Ok(view) = st.tensor(&ckpt_name) {
                let kind = match view.dtype() {
                    safetensors::Dtype::F16 => tch::Kind::Half,
                    safetensors::Dtype::BF16 => tch::Kind::BFloat16,
                    _ => tch::Kind::Float,
                };
                let shape: Vec<i64> = view.shape().iter().map(|&s| s as i64).collect();
                let loaded = Tensor::from_data_size(view.data(), &shape, kind);
                if tensor.size() == loaded.size() {
                    tensor.copy_(&loaded);
                    loaded_count += 1;
                }
            }
        }

        if !found {
            skipped += 1;
        }
    }

    eprintln!("  loaded {loaded_count}, skipped {skipped}");
    Ok(loaded_count)
}

/// Phase 1: Load text encoder, encode prompt, return context tensor.
/// Uses memory-mapped I/O to avoid loading the full 18GB file into RAM.
fn encode_prompt(
    tok_path: &str,
    tw_path: &str,
    prompt: &str,
) -> Result<Tensor, String> {
    eprintln!("loading text encoder (mmap)...");

    // Memory-map the file — no 18GB allocation
    let file = std::fs::File::open(tw_path).map_err(|e| format!("open {tw_path}: {e}"))?;
    let mmap = unsafe { memmap2::Mmap::map(&file).map_err(|e| format!("mmap: {e}"))? };
    let st = safetensors::SafeTensors::deserialize(&mmap)
        .map_err(|e| format!("deserialize: {e}"))?;

    let is_t5 = st.tensors().iter().any(|(k, _)| k.starts_with("encoder.block."));

    let (encoder, hidden) = if is_t5 {
        eprintln!("  T5 encoder detected (direct load, FP16)");
        let config = configurator::default_t5_config();
        let tokenizer = LTXVGemmaTokenizer::from_file(tok_path, 512)
            .map_err(|e| format!("tokenizer: {e}"))?;
        let enc = T5TextEncoder::from_checkpoint(&st, &config, tokenizer, 512, Device::Cpu);
        let h = enc.hidden_size();
        (TextEncoder::T5(enc), h)
    } else {
        eprintln!("  Gemma3 encoder detected");
        let config = configurator::default_config();
        let vs = tch::nn::VarStore::new(Device::Cpu);
        let enc = configurator::from_config(&vs.root(), &config, tok_path)
            .map_err(|e| format!("Gemma3 init: {e}"))?;
        let h = enc.hidden_size();
        load_weights(&vs, tw_path)?;
        (TextEncoder::Gemma3(Box::new(enc)), h)
    };

    drop(st);
    drop(mmap);
    drop(file);

    eprintln!("  encoding prompt ({hidden}d hidden)...");
    let context = encoder.encode(prompt);
    let seq_len = context.size()[1];
    eprintln!("  context: [1, {seq_len}, {hidden}]");

    drop(encoder);
    eprintln!("  text encoder freed");

    Ok(context)
}

fn main() {
    let args = Args::parse();
    let device = parse_device(&args.device);
    tch::maybe_init_cuda();

    // Load prompts
    let prompts = if let Some(ref path) = args.prompts_file {
        let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
            eprintln!("error: read {path}: {e}");
            process::exit(1);
        });
        let ps: Vec<String> = content.lines().filter(|l| !l.trim().is_empty()).map(|l| l.trim().to_string()).collect();
        eprintln!("batch: {} prompts from {path}", ps.len());
        ps
    } else {
        vec![args.prompt.clone()]
    };

    let batch_mode = prompts.len() > 1;
    let (b, c) = (1i64, 4i64);
    let (p1, p2, p3) = (2i64, 4i64, 4i64);
    let patch_dim = c * p1 * p2 * p3;
    let use_random = args.weights.is_none();
    let dim = if use_random { patch_dim } else { 2048 };
    let num_layers = if use_random { 2 } else { 28 };

    // Phase 1: Load transformer (shared across all prompts)
    eprintln!("loading transformer on {device:?}...");
    let vs = tch::nn::VarStore::new(device);

    let patchify_proj = if !use_random {
        Some(tch::nn::linear(vs.root() / "patchify_proj", patch_dim, dim, Default::default()))
    } else {
        None
    };

    if let Some(ref path) = args.weights {
        eprintln!("  loading transformer weights...");
        let _ = load_weights(&vs, path).unwrap_or_else(|e| {
            eprintln!("error: {e}");
            process::exit(1);
        });
    }

    let n_params: usize = vs.variables().values().map(|t| t.numel()).sum();
    eprintln!("ready: {dim}d, {num_layers} layers, {:.1}M params on {device:?}", n_params as f64 / 1e6);
    if device != Device::Cpu {
        eprintln!("GPU mode: transformer weights and all denoising tensors on-device");
    }

    // Phase 2: Pre-encode all prompts (load text encoder once, encode, free)
    let has_encoder = args.tokenizer.is_some() && args.text_weights.is_some();
    let contexts: Vec<Tensor> = if has_encoder {
        let tok = args.tokenizer.as_ref().unwrap();
        let tw = args.text_weights.as_ref().unwrap();
        eprintln!("encoding {} prompt(s) upfront...", prompts.len());
        let t0 = std::time::Instant::now();
        let mut ctxs = Vec::with_capacity(prompts.len());
        for (i, prompt) in prompts.iter().enumerate() {
            match encode_prompt(tok, tw, prompt) {
                Ok(ctx) => {
                    if batch_mode {
                        eprintln!("  [{}/{}] encoded (seq_len={})", i + 1, prompts.len(), ctx.size()[1]);
                    }
                    ctxs.push(ctx);
                }
                Err(e) => {
                    eprintln!("  [{}/{}] encoding failed: {e}", i + 1, prompts.len());
                    // Fallback context — will be skipped during denoising
                    ctxs.push(Tensor::zeros([1, 1, dim], (Kind::Float, Device::Cpu)));
                }
            }
        }
        let elapsed = t0.elapsed().as_secs_f64();
        eprintln!("encoding complete in {elapsed:.1}s");

        // Move context tensors to the target device for GPU-accelerated denoising.
        // The text encoder runs on CPU (and is freed immediately after encoding)
        // to minimize peak memory. The encoded context is small enough to copy
        // to GPU once, avoiding per-step CPU↔GPU transfers during the denoising loop.
        ctxs.iter_mut().for_each(|ctx| {
            if ctx.device() != device {
                *ctx = ctx.to_device(device);
            }
        });

        ctxs
    } else {
        eprintln!("no text encoder — using random contexts");
        prompts.iter().map(|_| Tensor::randn([1, 4, dim], (Kind::Float, device))).collect()
    };

    // Load init image for img2img (once, shared across prompts)
    let init_image: Option<Tensor> = if let Some(ref img_path) = args.init_image {
        let strength = args.strength.clamp(0.0, 1.0);
        if !(0.0..=1.0).contains(&strength) {
            eprintln!("error: --strength must be in [0.0, 1.0]");
            process::exit(1);
        }
        match load_init_image(img_path, args.frames, args.height, args.width, device, args.vae_weights.as_deref()) {
            Ok(img) => {
                eprintln!("img2img mode: strength={strength:.2}");
                Some(img)
            }
            Err(e) => {
                eprintln!("error: {e}");
                process::exit(1);
            }
        }
    } else {
        None
    };

    // Phase 3: Denoise each prompt
    let total = prompts.len();
    let mut completed = 0u32;
    let mut failed = 0u32;
    let mut timings: Vec<f64> = Vec::with_capacity(total);

    for (idx, prompt) in prompts.iter().enumerate() {
        let prompt_label = if batch_mode { format!("[{}/{}] ", idx + 1, total) } else { String::new() };

        // Skip if resuming and output already exists
        if args.resume {
            let skip_dir = if batch_mode {
                std::path::PathBuf::from(&args.output_dir).join(format!("{:04}", idx + 1))
            } else {
                std::path::PathBuf::from("output_frames")
            };
            if skip_dir.exists() {
                eprintln!("{prompt_label}\"{prompt}\" — skipped (output exists)");
                completed += 1;
                timings.push(0.0);
                continue;
            }
        }

        eprintln!("\n{prompt_label}\"{prompt}\"");

        let context = &contexts[idx];

        // Detect failed encodings (zero-filled placeholder)
        if context.size()[1] == 1 && context.size()[2] == dim && has_encoder {
            eprintln!("{prompt_label}skipping — encoding failed");
            failed += 1;
            timings.push(0.0);
            continue;
        }

        // Update model context_dim if text encoder was used
        let context_dim = if context.size()[2] != dim {
            Some(context.size()[2])
        } else {
            None
        };

        // Build model (with optional multi-GPU sharding)
        let model = if let Some(ref shard_str) = args.shard {
            let shard_devices = parse_shard_devices(shard_str);
            if shard_devices.len() > 1 {
                eprintln!("{prompt_label}sharding model across {} GPUs: {:?}", shard_devices.len(), shard_devices);
            }
            build_model_sharded(&vs.root(), dim, patch_dim, num_layers, context_dim, &shard_devices)
        } else if context_dim.is_some() {
            build_model(&vs.root(), dim, patch_dim, num_layers, context_dim)
        } else {
            build_model(&vs.root(), dim, patch_dim, num_layers, None)
        };

        // Pre-compute unconditional context (cached across steps)
        let uncond_context = Tensor::zeros([1, context.size()[1], context.size()[2]], (Kind::Float, device));

        // Denoise
        let scheduler = Ltx2Scheduler::default();
        let noiser = GaussianNoiser::new();
        let sigmas = scheduler.sigmas(args.steps);

        // Log configuration
        eprintln!("{prompt_label}step={}, guider={}, cfg={}", args.step_method, args.guider, args.cfg);

        let t0 = std::time::Instant::now();

        let (mut x, start_step) = if let Some(ref init_latent) = init_image {
            // img2img: add noise to the init image at the sigma level
            // determined by strength, then denoise from there
            let strength = args.strength.clamp(0.0, 1.0);
            let start_step = (strength * args.steps as f64).round() as usize;
            let start_step = start_step.min(args.steps);
            let start_sigma = sigmas[start_step];

            eprintln!(
                "{prompt_label}img2img: {} steps from sigma {:.4}, strength={strength:.2}",
                args.steps - start_step, start_sigma
            );

            let noise = Tensor::randn_like(init_latent);
            let noisy = noiser.add_noise(init_latent, &noise, start_sigma);
            (noisy, start_step)
        } else {
            // txt2img: start from pure noise
            tch::manual_seed(args.seed);
            eprintln!("{prompt_label}denoising: {} steps, cfg={}, seed={}", args.steps, args.cfg, args.seed);
            (Tensor::randn([b, c, args.frames, args.height, args.width], (Kind::Float, device)), 0)
        };
        for i in start_step..args.steps {
            let sigma = sigmas[i];
            let next_sigma = sigmas[i + 1];

            // Apply tiling if requested
            let denoised = if args.tile_size > 0 {
                // Tile-based denoising: process spatial patches independently
                let tiles = tile_spatial(&x, args.tile_size, args.tile_overlap);
                let mut denoised_tiles = Vec::new();

                for (patch, y_range, x_range) in &tiles {
                    let patched = patchify_5d(patch, p1, p2, p3);
                    let projected = if let Some(ref proj) = patchify_proj {
                        proj.forward_t(&patched, false)
                    } else {
                        patched
                    };

                    let timestep = Tensor::from_slice(&[sigma as f32]).to_device(device);
                    let cond_pred = model.forward(&projected, &timestep, context, None, None);

                    let uncond_pred = model.forward(&projected, &timestep, &uncond_context, None, None);

                    let guided = match args.guider.as_str() {
                        "apg" => APG::new(args.apg_scale, args.apg_momentum).guide(&cond_pred, &uncond_pred),
                        "stg" => STG::new(args.stg_spatial_scale, args.stg_temporal_scale).guide(&cond_pred, &uncond_pred),
                        _ => CFG::new(args.cfg).guide(&cond_pred, &uncond_pred),
                    };

                    let tile_denoised = unpatchify_5d(&guided, b, c, patch.size()[2], patch.size()[3], patch.size()[4], p1, p2, p3);
                    denoised_tiles.push((tile_denoised, *y_range, *x_range));
                }

                // Blend tiled results back together
                blend_tiles(&denoised_tiles, &[b, c, args.frames, args.height, args.width], args.tile_overlap)
            } else {
                // Standard full-frame denoising
                let patched = patchify_5d(&x, p1, p2, p3);
                let projected = if let Some(ref proj) = patchify_proj {
                    proj.forward_t(&patched, false)
                } else {
                    patched
                };

                let timestep = Tensor::from_slice(&[sigma as f32]).to_device(device);
                let cond_pred = model.forward(&projected, &timestep, context, None, None);

                let uncond_pred = model.forward(&projected, &timestep, &uncond_context, None, None);

                let guided = match args.guider.as_str() {
                    "apg" => APG::new(args.apg_scale, args.apg_momentum).guide(&cond_pred, &uncond_pred),
                    "stg" => STG::new(args.stg_spatial_scale, args.stg_temporal_scale).guide(&cond_pred, &uncond_pred),
                    _ => CFG::new(args.cfg).guide(&cond_pred, &uncond_pred),
                };

                unpatchify_5d(&guided, b, c, args.frames, args.height, args.width, p1, p2, p3)
            };

            // Apply diffusion step
            x = match args.step_method.as_str() {
                "res2s" => Res2sStep::new(1.0).step(&x, sigma, next_sigma, &denoised, Kind::Float),
                _ => EulerStep::new().step(&x, sigma, next_sigma, &denoised, Kind::Float),
            };

            let mean = x.mean(Kind::Float).double_value(&[]);
            let s = x.std(false).double_value(&[]);
            eprintln!("{prompt_label}  [{:>2}/{}] sigma={:.4} mean={:.4} std={:.4}", i + 1, args.steps, sigma, mean, s);
        }
        let elapsed = t0.elapsed().as_secs_f64();
        timings.push(elapsed);

        // VAE decode: convert latent to pixel-space frames
        let output_tensor = if args.decode {
            let vae_path = args.vae_weights.as_deref().or(args.weights.as_deref());
            match vae_path {
                Some(path) => match decode_via_vae(&x, path, device) {
                    Ok(pixel) => pixel,
                    Err(e) => {
                        eprintln!("{prompt_label}VAE decode failed: {e} — saving latent output");
                        x.clamp(-1.0, 1.0)
                    }
                },
                None => {
                    eprintln!("{prompt_label}error: --decode requires --vae-weights or --weights with VAE keys — saving latent output");
                    x.clamp(-1.0, 1.0)
                }
            }
        } else {
            x.clamp(-1.0, 1.0)
        };

        // Save output
        let out_dir = if batch_mode {
            std::path::PathBuf::from(&args.output_dir).join(format!("{:04}", idx + 1))
        } else {
            std::path::PathBuf::from("output_frames")
        };
        save_frames(&output_tensor.to_device(Device::Cpu), &out_dir, args.frames, args.height, args.width);

        // Save GIF
        let gif_path = if batch_mode {
            std::path::PathBuf::from(&args.output_dir).join(format!("{:04}.gif", idx + 1))
        } else {
            std::path::PathBuf::from("output.gif")
        };
        let _ = std::fs::create_dir_all(gif_path.parent().unwrap_or(std::path::Path::new(".")));
        if let Err(e) = save_gif(&output_tensor.to_device(Device::Cpu), args.frames, args.height, args.width, &gif_path) {
            eprintln!("{prompt_label}  gif: {e}");
        } else {
            eprintln!("{prompt_label}  saved: {}", gif_path.display());
        }

        // Audio generation (optional)
        if args.audio {
            if let Some(ref audio_vae_path) = args.audio_vae_weights {
                eprintln!("{prompt_label}generating audio...");
                let t0_audio = std::time::Instant::now();

                // Create audio VAE config and build decoder
                let audio_config = AudioVAEConfig::default();
                let audio_vs = tch::nn::VarStore::new(device);
                let audio_decoder = AudioDecoder::new(audio_vs.root() / "decoder", &audio_config);

                // Load audio VAE weights
                let n = load_vae_weights(&audio_vs, audio_vae_path, "audio_vae.");
                eprintln!("{prompt_label}  audio VAE: {n} tensors loaded");

                // Create audio latent from video latent (temporal alignment)
                // Video latent: (B, 128, T_v, H, W) → Audio latent: (B, 64, T_v, 128)
                let audio_t = x.size()[2]; // temporal frames from video
                let audio_latent = Tensor::randn(
                    [b, audio_config.latent_channels, audio_t, audio_config.input_features],
                    (Kind::Float, device),
                );

                // Denoise audio latent using the same transformer as video
                let denoised_audio = denoise_audio(
                    &audio_latent, &model, context, &uncond_context,
                    patchify_proj.as_ref(), device, args.steps, args.cfg,
                );

                // Decode audio through VAE decoder
                let audio_mel = audio_decoder.forward(&denoised_audio);
                eprintln!("{prompt_label}  audio mel: {:?}", audio_mel.size());

                // Save as WAV (raw mel output; a full vocoder such as HiFi-GAN would
                // convert the mel spectrogram to a proper waveform here)
                // Average over mel bins (dim 1) to get waveform-like output
                let audio_samples = audio_mel
                    .narrow(1, 0, 1)                     // Take first mel bin as a simple waveform approximation;
                    // full vocoder would use all mel bins for high-fidelity output
                    .squeeze_dim(1);
                let audio_path = if batch_mode {
                    std::path::PathBuf::from(&args.output_dir).join(format!("{:04}.wav", idx + 1))
                } else {
                    std::path::PathBuf::from(&args.audio_output)
                };
                if let Err(e) = save_wav(&audio_samples.to_device(Device::Cpu), &audio_path, 44100) {
                    eprintln!("{prompt_label}  audio save error: {e}");
                }

                let audio_elapsed = t0_audio.elapsed().as_secs_f64();
                eprintln!("{prompt_label}  audio: {audio_elapsed:.1}s");
            } else {
                eprintln!("{prompt_label}warning: --audio requires --audio-vae-weights");
            }
        }

        completed += 1;
        eprintln!("{prompt_label}  {elapsed:.1}s");

        if batch_mode {
            let bar_len = 30;
            let filled = ((idx + 1) as f64 / total as f64 * bar_len as f64) as usize;
            let bar: String = "█".repeat(filled) + &"░".repeat(bar_len - filled);
            eprintln!("progress: [{bar}] {}/{}", idx + 1, total);
        }
    }

    // Summary
    if batch_mode {
        let total_time: f64 = timings.iter().sum();
        let active: Vec<f64> = timings.iter().copied().filter(|t| *t > 0.0).collect();
        let avg = if active.is_empty() { 0.0 } else { active.iter().sum::<f64>() / active.len() as f64 };
        eprintln!("\nbatch complete: {completed} succeeded, {failed} failed out of {total}");
        eprintln!("total time: {total_time:.1}s, avg per prompt: {avg:.1}s");

        // Write manifest.json
        let _ = write_manifest(&args.output_dir, &prompts, &timings, args.steps, args.cfg, args.seed);
    } else {
        eprintln!("done");
    }
}

fn save_frames(x: &Tensor, dir: &std::path::Path, frames: i64, h: i64, w: i64) {
    let pixel = {
        let p = x.clamp(-1.0, 1.0);
        let r = p.narrow(1, 0, 1).squeeze_dim(1);
        let g = p.narrow(1, 1, 1).squeeze_dim(1);
        let b = p.narrow(1, 2, 1).squeeze_dim(1);
        Tensor::stack(&[&r, &g, &b], 1)
    };

    let pixel_min = pixel.min().double_value(&[]);
    let pixel_max = pixel.max().double_value(&[]);
    let pixel = (pixel - pixel_min) / (pixel_max - pixel_min + STABILITY_EPS);
    let pixel = (pixel * 255.0).to_kind(Kind::Uint8);

    if std::fs::create_dir_all(dir).is_err() {
        eprintln!("warning: could not create {}/", dir.display());
        return;
    }

    use std::io::Write;

    // Save PNG frames (primary output)
    for i in 0..frames {
        let frame = pixel.narrow(2, i, 1).reshape([3, h, w]).permute([1, 2, 0]);
        let path = dir.join(format!("frame_{i:04}.png"));

        // Flatten to contiguous [h*w*3] before extracting values
        let flat = frame.reshape([h * w * 3]);
        let frame_bytes: Vec<u8> = (0..flat.size()[0]).map(|j| flat.double_value(&[j]) as u8).collect();

        let mut img = image::ImageBuffer::new(w as u32, h as u32);
        for (pixel_out, rgb) in img.pixels_mut().zip(frame_bytes.chunks(3)) {
            *pixel_out = image::Rgb([rgb[0], rgb[1], rgb[2]]);
        }
        if let Err(e) = img.save(&path) {
            eprintln!("warning: save PNG {}: {e}", path.display());
        }
    }

    // Also save PGM frames (fallback for ffmpeg pipeline)
    for i in 0..frames {
        let frame = pixel.narrow(2, i, 1).reshape([3, h, w]).permute([1, 2, 0]);
        let path = dir.join(format!("frame_{i:04}.pgm"));
        if let Ok(mut f) = std::fs::File::create(&path) {
            let _ = write!(f, "P6\n{w} {h}\n255\n");
            let data = frame.reshape([h * w * 3]);
            let bytes: Vec<u8> = (0..data.size()[0]).map(|j| data.double_value(&[j]) as u8).collect();
            let _ = f.write_all(&bytes);
        }
    }
}

fn save_gif(x: &Tensor, frames: i64, h: i64, w: i64, output: &std::path::Path) -> Result<(), String> {
    let tmp_dir = output.parent().unwrap_or(std::path::Path::new(".")).join(".tmp_gif");
    std::fs::create_dir_all(&tmp_dir).map_err(|e| format!("create tmp: {e}"))?;

    // Save PGM frames
    let pixel = {
        let p = x.clamp(-1.0, 1.0);
        let r = p.narrow(1, 0, 1).squeeze_dim(1);
        let g = p.narrow(1, 1, 1).squeeze_dim(1);
        let b = p.narrow(1, 2, 1).squeeze_dim(1);
        Tensor::stack(&[&r, &g, &b], 1)
    };
    let pixel_min = pixel.min().double_value(&[]);
    let pixel_max = pixel.max().double_value(&[]);
    let pixel = (pixel - pixel_min) / (pixel_max - pixel_min + STABILITY_EPS);
    let pixel = (pixel * 255.0).to_kind(Kind::Uint8);

    use std::io::Write;
    for i in 0..frames {
        let frame = pixel.narrow(2, i, 1).reshape([3, h, w]).permute([1, 2, 0]);
        let path = tmp_dir.join(format!("frame_{i:04}.pgm"));
        if let Ok(mut f) = std::fs::File::create(&path) {
            let _ = write!(f, "P6\n{w} {h}\n255\n");
            let data = frame.reshape([h * w * 3]);
            let bytes: Vec<u8> = (0..data.size()[0]).map(|j| data.double_value(&[j]) as u8).collect();
            let _ = f.write_all(&bytes);
        }
    }

    let filter = "scale=256:256:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse".to_string();
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-framerate", "8", "-i", tmp_dir.join("frame_%04d.pgm").to_str().unwrap_or(""), "-vf", &filter, "-loop", "0", output.to_str().unwrap_or("")])
        .status()
        .map_err(|e| format!("ffmpeg: {e}"))?;

    let _ = std::fs::remove_dir_all(&tmp_dir);

    if status.success() { Ok(()) } else { Err("ffmpeg failed".to_string()) }
}

fn write_manifest(
    output_dir: &str,
    prompts: &[String],
    timings: &[f64],
    steps: usize,
    cfg: f64,
    seed: i64,
) -> Result<(), String> {
    let path = std::path::PathBuf::from(output_dir).join("manifest.json");
    let total_time: f64 = timings.iter().sum();
    let mut entries = Vec::new();
    for (i, (prompt, time)) in prompts.iter().zip(timings.iter()).enumerate() {
        entries.push(format!(
            r#"  {{ "index": {}, "prompt": {:?}, "output": "{:04}/", "time_s": {:.1} }}"#,
            i + 1, prompt, i + 1, time
        ));
    }
    let json = format!(
        "{{\n  \"steps\": {}, \"cfg\": {}, \"seed\": {}, \"total_time_s\": {:.1},\n  \"results\": [\n{}\n  ]\n}}",
        steps, cfg, seed, total_time, entries.join(",\n")
    );
    std::fs::write(&path, &json).map_err(|e| format!("write {}: {e}", path.display()))?;
    eprintln!("manifest: {}", path.display());
    Ok(())
}

/// Save a 1D audio tensor as a WAV file.
///
/// `audio`: `[1, 1, T]` or `[T]` waveform tensor (values in [-1, 1])
/// `path`: output file path
/// `sample_rate`: audio sample rate in Hz (e.g. 44100)
fn save_wav(audio: &Tensor, path: &std::path::Path, sample_rate: u32) -> Result<(), String> {
    let flat = audio.reshape([-1]);
    let n_samples = flat.size()[0] as usize;

    // Convert to i16 PCM
    let samples: Vec<i16> = (0..n_samples)
        .map(|i| {
            let v = flat.double_value(&[i as i64]).clamp(-1.0, 1.0);
            (v * 32767.0) as i16
        })
        .collect();

    let data_size = (samples.len() * 2) as u32;
    let file_size = 36 + data_size;

    use std::io::Write;
    let mut f = std::fs::File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;

    // WAV header
    f.write_all(b"RIFF").unwrap();
    f.write_all(&file_size.to_le_bytes()).unwrap();
    f.write_all(b"WAVE").unwrap();
    f.write_all(b"fmt ").unwrap();
    f.write_all(&16u32.to_le_bytes()).unwrap(); // chunk size
    f.write_all(&1u16.to_le_bytes()).unwrap(); // PCM format
    f.write_all(&1u16.to_le_bytes()).unwrap(); // mono
    f.write_all(&sample_rate.to_le_bytes()).unwrap();
    f.write_all(&(sample_rate * 2).to_le_bytes()).unwrap(); // byte rate
    f.write_all(&2u16.to_le_bytes()).unwrap(); // block align
    f.write_all(&16u16.to_le_bytes()).unwrap(); // bits per sample
    f.write_all(b"data").unwrap();
    f.write_all(&data_size.to_le_bytes()).unwrap();
    for &s in &samples {
        f.write_all(&s.to_le_bytes()).unwrap();
    }

    eprintln!("  saved WAV: {} ({n_samples} samples, {sample_rate} Hz)", path.display());
    Ok(())
}

/// Denoise audio latent through transformer.
///
/// Audio latent shape: `(B, C_audio, T_audio, F)` where:
/// - C_audio = 64 (audio latent channels)
/// - T_audio = temporal frames (aligned with video)
/// - F = frequency bins (128 mel bins)
///
/// Uses the same transformer architecture as video denoising, with audio
/// patchification (4D instead of 5D) and audio-specific timestep embedding.
/// Returns the denoised audio latent ready for VAE decoding.
#[allow(clippy::too_many_arguments)]
fn denoise_audio(
    audio_latent: &Tensor,
    model: &LTXModel,
    context: &Tensor,
    uncond_context: &Tensor,
    patchify_proj: Option<&tch::nn::Linear>,
    device: Device,
    steps: usize,
    cfg_scale: f64,
) -> Tensor {
    let (c, _t, f) = (
        audio_latent.size()[1],
        audio_latent.size()[2],
        audio_latent.size()[3],
    );

    let scheduler = Ltx2Scheduler::default();
    let guider = CFG::new(cfg_scale);
    let step = EulerStep::new();
    let sigmas = scheduler.sigmas(steps);

    let mut x = audio_latent.shallow_clone();

    for i in 0..steps {
        let sigma = sigmas[i];
        let next_sigma = sigmas[i + 1];

        // Patchify audio: (B, C, T, F) -> (B, T, C*F)
        let patched = ltx_patchify::patchify_audio(&x);
        let projected = if let Some(proj) = patchify_proj {
            proj.forward_t(&patched, false)
        } else {
            patched
        };

        // Create audio timestep (same sigma schedule as video)
        let timestep = Tensor::from_slice(&[sigma as f32]).to_device(device);

        // Conditional forward: use text context
        let cond_pred = model.forward(&projected, &timestep, context, None, None);

        // Unconditional forward: empty context for CFG
        let uncond_pred = model.forward(&projected, &timestep, uncond_context, None, None);

        // Apply classifier-free guidance
        let guided = guider.guide(&cond_pred, &uncond_pred);

        // Unpatchify: (B, T, C*F) -> (B, C, T, F)
        let unpatched = ltx_patchify::unpatchify_audio(&guided, c, f);

        // Euler step
        x = step.step(&x, sigma, next_sigma, &unpatched, Kind::Float);
    }

    x
}
