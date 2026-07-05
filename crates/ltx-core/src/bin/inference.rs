//! LTX-2.3 inference demo.
//!
//! Usage:
//!   ltx-inference --steps 4
//!   ltx-inference --weights model.safetensors --steps 20
//!   ltx-inference --weights model.safetensors --tokenizer tok.model --text-weights text.safetensors --prompt "a sunset"
//!
//! Memory-efficient: encodes prompt, frees text encoder, then loads transformer.

use std::process;

use clap::Parser;
use ltx_attention::RopeType;
use ltx_components::{EulerStep, GaussianNoiser, Ltx2Scheduler, CFG};
use ltx_norm::RMSNorm;
use ltx_patchify::{patchify_5d, unpatchify_5d};
use ltx_text_encoder::configurator;
use ltx_text_encoder::encoder::{GemmaTextEncoder, T5TextEncoder};
use ltx_text_encoder::tokenizer::LTXVGemmaTokenizer;
use ltx_transformer::block::BasicAVTransformerBlock;
use ltx_transformer::model::LTXModel;
use ltx_types::{Guider, Scheduler, STABILITY_EPS};
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

/// Load an image file and convert to a 5D latent tensor: (1, 4, T, H, W).
///
/// The image is resized to (width, height), normalized to [-1, 1], and its 3
/// RGB channels are projected to 4 latent channels by padding the 4th with
/// zeros.  The single frame is replicated across all T time steps.
fn load_init_image(
    path: &str,
    frames: i64,
    height: i64,
    width: i64,
    device: Device,
) -> Result<Tensor, String> {
    let img = image::open(path).map_err(|e| format!("open {path}: {e}"))?;
    let img = img.resize_exact(
        width as u32,
        height as u32,
        image::imageops::FilterType::Lanczos3,
    );
    let rgb = img.to_rgb8();
    let raw = rgb.into_raw();
    let pixels: Vec<f32> = raw.iter().map(|&b| (b as f32 / 127.5) - 1.0).collect();

    // (H, W, 3) → (1, 3, 1, H, W)
    let t = Tensor::from_slice(&pixels)
        .reshape([height, width, 3])
        .permute([2, 0, 1]) // (3, H, W)
        .unsqueeze(0) // (3, H, W) -> (1, 3, H, W)
        .unsqueeze(2) // (1, 3, H, W) -> (1, 3, 1, H, W)
        .to_kind(Kind::Float)
        .to_device(device);

    // Replicate single frame across T time steps
    let t = t.expand([1, 3, frames, height, width], true);

    // Pad 3 → 4 latent channels (4th channel = zeros)
    let zeros = Tensor::zeros([1, 1, frames, height, width], (Kind::Float, device));
    let latent = Tensor::cat(&[&t, &zeros], 1);

    eprintln!("loaded init image: {path} -> [{}, {}, {}, {}, {}]",
        latent.size()[0], latent.size()[1], latent.size()[2], latent.size()[3], latent.size()[4]);
    Ok(latent)
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
    eprintln!("ready: {dim}d, {num_layers} layers, {:.1}M params", n_params as f64 / 1e6);

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
                    // Placeholder context — will be skipped during denoising
                    ctxs.push(Tensor::zeros([1, 1, dim], (Kind::Float, Device::Cpu)));
                }
            }
        }
        let elapsed = t0.elapsed().as_secs_f64();
        eprintln!("encoding complete in {elapsed:.1}s");
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
        match load_init_image(img_path, args.frames, args.height, args.width, device) {
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
        let model = if context_dim.is_some() {
            build_model(&vs.root(), dim, patch_dim, num_layers, context_dim)
        } else {
            build_model(&vs.root(), dim, patch_dim, num_layers, None)
        };

        // Denoise
        let scheduler = Ltx2Scheduler::default();
        let guider = CFG::new(args.cfg);
        let step = EulerStep::new();
        let noiser = GaussianNoiser::new();
        let sigmas = scheduler.sigmas(args.steps);

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

            let patched = patchify_5d(&x, p1, p2, p3);
            let projected = if let Some(ref proj) = patchify_proj {
                proj.forward_t(&patched, false)
            } else {
                patched
            };

            let timestep = Tensor::from_slice(&[sigma as f32]);
            let cond_pred = model.forward(&projected, &timestep, context, None, None);

            let uncond_context = Tensor::zeros([1, context.size()[1], context.size()[2]], (Kind::Float, device));
            let uncond_pred = model.forward(&projected, &timestep, &uncond_context, None, None);

            let guided = guider.guide(&cond_pred, &uncond_pred);
            let denoised = unpatchify_5d(&guided, b, c, args.frames, args.height, args.width, p1, p2, p3);
            x = step.step(&x, sigma, next_sigma, &denoised, Kind::Float);

            let mean = x.mean(Kind::Float).double_value(&[]);
            let s = x.std(false).double_value(&[]);
            eprintln!("{prompt_label}  [{:>2}/{}] sigma={:.4} mean={:.4} std={:.4}", i + 1, args.steps, sigma, mean, s);
        }
        let elapsed = t0.elapsed().as_secs_f64();
        timings.push(elapsed);

        // Save output
        let out_dir = if batch_mode {
            std::path::PathBuf::from(&args.output_dir).join(format!("{:04}", idx + 1))
        } else {
            std::path::PathBuf::from("output_frames")
        };
        save_frames(&x.to_device(Device::Cpu), &out_dir, args.frames, args.height, args.width);

        // Save GIF
        let gif_path = if batch_mode {
            std::path::PathBuf::from(&args.output_dir).join(format!("{:04}.gif", idx + 1))
        } else {
            std::path::PathBuf::from("output.gif")
        };
        let _ = std::fs::create_dir_all(gif_path.parent().unwrap_or(std::path::Path::new(".")));
        if let Err(e) = save_gif(&x.to_device(Device::Cpu), args.frames, args.height, args.width, &gif_path) {
            eprintln!("{prompt_label}  gif: {e}");
        } else {
            eprintln!("{prompt_label}  saved: {}", gif_path.display());
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

    let filter = format!("scale=256:256:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse");
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
