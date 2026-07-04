//! LTX-2.3 inference demo.
//!
//! Usage:
//!   ltx-inference --steps 4
//!   ltx-inference --weights model.safetensors --steps 20
//!   ltx-inference --weights model.safetensors --height 32 --width 32 --frames 4

use std::process;

use clap::Parser;
use ltx_attention::RopeType;
use ltx_components::{EulerStep, Ltx2Scheduler, CFG};
use ltx_norm::RMSNorm;
use ltx_patchify::{patchify_5d, unpatchify_5d};
use ltx_transformer::block::BasicAVTransformerBlock;
use ltx_transformer::model::LTXModel;
use ltx_types::{Guider, Scheduler, STABILITY_EPS};
use tch::nn::ModuleT;
use tch::{Device, Kind, Tensor};

#[derive(Parser)]
#[command(name = "ltx-inference", about = "LTX-2.3 video diffusion inference")]
struct Args {
    /// Path to .safetensors weights (omit for random init)
    #[arg(short, long)]
    weights: Option<String>,

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
}

fn build_model(vs: &tch::nn::Path, dim: i64, patch_dim: i64, num_layers: i64) -> LTXModel {
    let mut blocks = Vec::new();
    for i in 0..num_layers {
        blocks.push(BasicAVTransformerBlock::new(
            &(vs / "blocks" / i),
            dim,
            4,
            dim / 4,
            None,
            RopeType::Interleaved,
        ));
    }
    let norm_out = RMSNorm::default_eps_with_path(vs / "norm_out", dim);
    let proj_out = tch::nn::linear(vs / "proj_out", dim, patch_dim, Default::default());
    LTXModel::new(blocks, norm_out, proj_out)
}

fn main() {
    let args = Args::parse();

    let (b, c) = (1i64, 4i64);
    let (p1, p2, p3) = (2i64, 4i64, 4i64);
    let patch_dim = c * p1 * p2 * p3;
    let use_random = args.weights.is_none();
    let dim = if use_random { patch_dim } else { 2048 };
    let num_layers = if use_random { 2 } else { 28 };

    tch::maybe_init_cuda();
    let vs = tch::nn::VarStore::new(Device::Cpu);
    let model = build_model(&vs.root(), dim, patch_dim, num_layers);

    let patchify_proj = if !use_random {
        Some(tch::nn::linear(
            vs.root() / "patchify_proj",
            patch_dim,
            dim,
            Default::default(),
        ))
    } else {
        None
    };

    if let Some(ref path) = args.weights {
        if let Err(e) = load_weights(&vs, path) {
            eprintln!("error: {e}");
            process::exit(1);
        }
    }

    let n_params: usize = vs.variables().values().map(|t| t.numel()).sum();
    eprintln!(
        "init: {dim}d, {num_layers} layers, {:.1}M params",
        n_params as f64 / 1e6
    );

    tch::manual_seed(42);
    let mut x = Tensor::randn([b, c, args.frames, args.height, args.width], (Kind::Float, Device::Cpu));
    let context = Tensor::randn([1, 4, dim], (Kind::Float, Device::Cpu));

    let scheduler = Ltx2Scheduler::default();
    let guider = CFG::new(args.cfg);
    let step = EulerStep::new();
    let sigmas = scheduler.sigmas(args.steps);

    eprintln!("denoising: {} steps, cfg={}", args.steps, args.cfg);
    for i in 0..args.steps {
        let sigma = sigmas[i];
        let next_sigma = sigmas[i + 1];

        let patched = patchify_5d(&x, p1, p2, p3);
        let projected = if let Some(ref proj) = patchify_proj {
            proj.forward_t(&patched, false)
        } else {
            patched
        };

        let timestep = Tensor::from_slice(&[sigma as f32]);
        let cond_pred = model.forward(&projected, &timestep, &context, None, None);

        let uncond_context = Tensor::zeros(
            [1, context.size()[1], context.size()[2]],
            (Kind::Float, Device::Cpu),
        );
        let uncond_pred = model.forward(&projected, &timestep, &uncond_context, None, None);

        let guided = guider.guide(&cond_pred, &uncond_pred);
        let denoised = unpatchify_5d(&guided, b, c, args.frames, args.height, args.width, p1, p2, p3);
        x = step.step(&x, sigma, next_sigma, &denoised, Kind::Float);

        let mean = x.mean(Kind::Float).double_value(&[]);
        let std = x.std(false).double_value(&[]);
        eprintln!("  [{:>2}/{}] σ={:.4} μ={:.4} σ={:.4}", i + 1, args.steps, sigma, mean, std);
    }

    save_frames(&x, args.frames, args.height, args.width);
    eprintln!("done");
}

fn load_weights(vs: &tch::nn::VarStore, path: &str) -> Result<(), String> {
    let data = std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
    let st = safetensors::SafeTensors::deserialize(&data)
        .map_err(|e| format!("deserialize {path}: {e}"))?;

    let mut loaded_count = 0u32;
    let mut skipped = 0u32;
    let _no_grad = tch::no_grad_guard();
    let mut vars = vs.variables();

    for (name, tensor) in vars.iter_mut() {
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
            } else {
                skipped += 1;
            }
        } else {
            skipped += 1;
        }
    }

    eprintln!("weights: {loaded_count} loaded, {skipped} skipped");
    Ok(())
}

fn save_frames(x: &Tensor, frames: i64, h: i64, w: i64) {
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

    let dir = "output_frames";
    if std::fs::create_dir_all(dir).is_err() {
        eprintln!("warning: could not create {dir}/");
        return;
    }

    use std::io::Write;
    for i in 0..frames {
        let frame = pixel.narrow(2, i, 1).reshape([3, h, w]).permute([1, 2, 0]);
        let path = format!("{dir}/frame_{i:04}.pgm");
        if let Ok(mut f) = std::fs::File::create(&path) {
            let _ = write!(f, "P6\n{w} {h}\n255\n");
            let data = frame.reshape([h * w * 3]);
            let bytes: Vec<u8> = (0..data.size()[0])
                .map(|i| data.double_value(&[i]) as u8)
                .collect();
            let _ = f.write_all(&bytes);
        }
    }

    let video = "output.mp4";
    let ok = std::process::Command::new("ffmpeg")
        .args(["-y", "-framerate", "8", "-i", &format!("{dir}/frame_%04d.pgm"), "-c:v", "libx264", "-pix_fmt", "yuv420p", video])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if ok {
        eprintln!("output: {video}");
    } else {
        eprintln!("output: {dir}/ (frames only, ffmpeg unavailable)");
    }
}
