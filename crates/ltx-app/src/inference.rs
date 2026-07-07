use std::sync::mpsc;

use ltx_attention::RopeType;
use ltx_components::{Beta, EulerStep, LinearQuadratic, Ltx2Scheduler, CFG};
use ltx_norm::RMSNorm;
use ltx_patchify::{patchify_5d, unpatchify_5d};
use ltx_transformer::block::BasicAVTransformerBlock;
use ltx_transformer::model::LTXModel;
use ltx_types::{Guider, Scheduler};
use tch::nn::ModuleT;
use tch::{Device, Kind, Tensor};

use crate::state::{DeviceKind, GuiCommand, GuiEvent, InferenceParams, SchedulerKind};

pub fn spawn_inference_thread(
    params: InferenceParams,
    event_tx: mpsc::Sender<GuiEvent>,
    cmd_rx: mpsc::Receiver<GuiCommand>,
) {
    std::thread::spawn(move || {
        if let Err(e) = run_inference(params, event_tx.clone(), cmd_rx) {
            let _ = event_tx.send(GuiEvent::Error(e));
        }
    });
}

fn run_inference(
    params: InferenceParams,
    event_tx: mpsc::Sender<GuiEvent>,
    cmd_rx: mpsc::Receiver<GuiCommand>,
) -> Result<(), String> {
    let device = match params.device {
        DeviceKind::Cpu => Device::Cpu,
        DeviceKind::Cuda | DeviceKind::Rocm => Device::Cuda(0),
    };

    let _ = event_tx.send(GuiEvent::Progress {
        step: 0,
        total: params.steps,
        sigma: 0.0,
    });

    // Build model
    let vs = tch::nn::VarStore::new(device);
    let (b, c) = (1i64, 4i64);
    let (p1, p2, p3) = (2i64, 4i64, 4i64);
    let patch_dim = c * p1 * p2 * p3;
    let use_random = params.weights_path.is_none();
    let dim = if use_random { patch_dim } else { 2048 };
    let num_layers = if use_random { 2 } else { 28 };

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

    // Load weights
    if let Some(ref path) = params.weights_path {
        let _ = event_tx.send(GuiEvent::Progress {
            step: 0,
            total: params.steps,
            sigma: 0.0,
        });
        load_weights(&vs, path)?;
    }

    // Create latent
    tch::manual_seed(42);
    let mut x = Tensor::randn(
        [b, c, params.frames, params.height, params.width],
        (Kind::Float, device),
    );

    // Text context
    let context = if let (Some(tok_path), Some(tw_path)) =
        (&params.tokenizer_path, &params.text_weights_path)
    {
        let config = ltx_text_encoder::configurator::default_config();
        let encoder_vs = tch::nn::VarStore::new(Device::Cpu);
        let encoder = ltx_text_encoder::configurator::from_config(
            &encoder_vs.root(),
            &config,
            tok_path.to_str().unwrap_or(""),
        )
        .map_err(|e| format!("text encoder init: {e}"))?;
        load_weights(&encoder_vs, tw_path)?;
        let encoded = encoder.encode(&params.prompt);
        let seq_len = encoded.size()[1];
        let _ = event_tx.send(GuiEvent::Progress {
            step: 0,
            total: params.steps,
            sigma: 0.0,
        });
        eprintln!("encoded prompt: [1, {seq_len}, {}]", encoded.size()[2]);
        encoded
    } else {
        Tensor::randn([1, 4, dim], (Kind::Float, device))
    };

    // Setup scheduler + guider + step
    let scheduler: Box<dyn Scheduler> = match params.scheduler {
        SchedulerKind::Ltx2 => Box::new(Ltx2Scheduler::default()),
        SchedulerKind::LinearQuadratic => Box::new(LinearQuadratic::default()),
        SchedulerKind::Beta => Box::new(Beta::default()),
    };
    let guider = CFG::new(params.cfg_scale);
    let step = EulerStep::new();
    let sigmas = scheduler.sigmas(params.steps);

    // Denoising loop
    for i in 0..params.steps {
        // Check for cancellation
        if let Ok(GuiCommand::Cancel) = cmd_rx.try_recv() {
            return Err("Cancelled".to_string());
        }

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
            (Kind::Float, device),
        );
        let uncond_pred = model.forward(&projected, &timestep, &uncond_context, None, None);

        let guided = guider.guide(&cond_pred, &uncond_pred);
        let denoised = unpatchify_5d(
            &guided,
            b,
            c,
            params.frames,
            params.height,
            params.width,
            p1,
            p2,
            p3,
        );
        x = step.step(&x, sigma, next_sigma, &denoised, Kind::Float);

        let _ = event_tx.send(GuiEvent::Progress {
            step: i + 1,
            total: params.steps,
            sigma,
        });
    }

    // Decode to frames
    let _ = event_tx.send(GuiEvent::Decoding);

    let frames = latent_to_frames(&x, params.frames, params.height, params.width);
    let _ = event_tx.send(GuiEvent::FramesReady(frames));

    Ok(())
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

fn load_weights(vs: &tch::nn::VarStore, path: &std::path::Path) -> Result<(), String> {
    let data = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let st = safetensors::SafeTensors::deserialize(&data)
        .map_err(|e| format!("deserialize {}: {e}", path.display()))?;

    let _no_grad = tch::no_grad_guard();
    let mut vars = vs.variables();
    let mut loaded = 0u32;

    for (name, tensor) in vars.iter_mut() {
        if let Ok(view) = st.tensor(name) {
            let kind = match view.dtype() {
                safetensors::Dtype::F16 => tch::Kind::Half,
                safetensors::Dtype::BF16 => tch::Kind::BFloat16,
                _ => tch::Kind::Float,
            };
            let shape: Vec<i64> = view.shape().iter().map(|&s| s as i64).collect();
            let loaded_tensor = Tensor::from_data_size(view.data(), &shape, kind);
            if tensor.size() == loaded_tensor.size() {
                tensor.copy_(&loaded_tensor);
                loaded += 1;
            }
        }
    }

    eprintln!("weights: {loaded} loaded");
    Ok(())
}

fn latent_to_frames(x: &Tensor, frames: i64, h: i64, w: i64) -> Vec<Vec<u8>> {
    let pixel = {
        let p = x.clamp(-1.0, 1.0);
        let r = p.narrow(1, 0, 1).squeeze_dim(1);
        let g = p.narrow(1, 1, 1).squeeze_dim(1);
        let b = p.narrow(1, 2, 1).squeeze_dim(1);
        Tensor::stack(&[&r, &g, &b], 1)
    };

    let pixel_min = pixel.min().double_value(&[]);
    let pixel_max = pixel.max().double_value(&[]);
    let eps = 1e-8;
    let pixel = (pixel - pixel_min) / (pixel_max - pixel_min + eps);
    let pixel = (pixel * 255.0).to_kind(Kind::Uint8);

    let mut result = Vec::new();
    for i in 0..frames {
        let frame = pixel.narrow(2, i, 1).reshape([3, h, w]).permute([1, 2, 0]);
        let data = frame.reshape([h * w * 3]);
        let bytes: Vec<u8> = (0..data.size()[0])
            .map(|j| data.double_value(&[j]) as u8)
            .collect();
        result.push(bytes);
    }
    result
}
