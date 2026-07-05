use eframe::egui;

use crate::state::{AppState, DeviceKind, GuiCommand, InferenceParams, SchedulerKind};

pub fn show(ui: &mut egui::Ui, state: &mut AppState) {
    ui.heading("Controls");
    ui.separator();

    // Prompt
    ui.label("Prompt").on_hover_text("Text description of the video to generate");
    ui.text_edit_singleline(&mut state.prompt)
        .on_hover_text("Enter a text prompt to guide video generation. Longer prompts produce more detailed output.");
    ui.add_space(4.0);

    // Weights
    ui.label("Model Weights").on_hover_text("Transformer .safetensors checkpoint (omit for random init)");
    ui.horizontal(|ui| {
        let label = match &state.weights_path {
            Some(p) => p.file_name().unwrap_or_default().to_string_lossy().to_string(),
            None => "None (random init)".to_string(),
        };
        ui.label(&label).on_hover_text("Currently loaded weight file");
        if ui.button("Browse…").on_hover_text("Select transformer weights file").clicked() {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("SafeTensors", &["safetensors"])
                .pick_file()
            {
                state.weights_path = Some(path);
            }
        }
        if state.weights_path.is_some() && ui.button("×").on_hover_text("Clear weights (revert to random init)").clicked() {
            state.weights_path = None;
        }
    });
    ui.add_space(4.0);

    // Text Encoder
    ui.label("Text Encoder").on_hover_text("SentencePiece tokenizer + text encoder weights (T5 or Gemma3) for prompt conditioning");
    ui.horizontal(|ui| {
        let label = match &state.tokenizer_path {
            Some(p) => p.file_name().unwrap_or_default().to_string_lossy().to_string(),
            None => "No tokenizer".to_string(),
        };
        ui.label(&label).on_hover_text("SentencePiece tokenizer model (.model file)");
        if ui.button("Browse…").on_hover_text("Select tokenizer model file").clicked() {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("SentencePiece", &["model"])
                .pick_file()
            {
                state.tokenizer_path = Some(path);
            }
        }
    });
    ui.horizontal(|ui| {
        let label = match &state.text_weights_path {
            Some(p) => p.file_name().unwrap_or_default().to_string_lossy().to_string(),
            None => "No text weights".to_string(),
        };
        ui.label(&label).on_hover_text("Text encoder weights (.safetensors file)");
        if ui.button("Browse…").on_hover_text("Select text encoder weights").clicked() {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("SafeTensors", &["safetensors"])
                .pick_file()
            {
                state.text_weights_path = Some(path);
            }
        }
    });
    ui.add_space(4.0);

    // Resolution
    ui.label("Resolution").on_hover_text("Latent-space dimensions (pixel = latent × scale factor, e.g. 16×32 = 512×1024)");
    ui.horizontal(|ui| {
        ui.label("H").on_hover_text("Latent height in pixels");
        ui.add(egui::DragValue::new(&mut state.height).range(4..=256).suffix("px"))
            .on_hover_text("Height of the generated video in latent space");
        ui.label("W").on_hover_text("Latent width in pixels");
        ui.add(egui::DragValue::new(&mut state.width).range(4..=256).suffix("px"))
            .on_hover_text("Width of the generated video in latent space");
    });
    ui.horizontal(|ui| {
        ui.label("Frames").on_hover_text("Number of frames to generate");
        ui.add(egui::DragValue::new(&mut state.frames).range(1..=128))
            .on_hover_text("Number of video frames (e.g. 8 frames at 8fps = 1 second)");
    });
    ui.add_space(4.0);

    // Inference params
    ui.label("Steps").on_hover_text("Number of denoising steps (more = better quality, slower)");
    ui.add(egui::Slider::new(&mut state.steps, 1..=100))
        .on_hover_text("Denoising steps: 5-10 for quick preview, 20-50 for quality");
    ui.label("CFG Scale").on_hover_text("Classifier-free guidance strength (higher = follows prompt more)");
    ui.add(egui::Slider::new(&mut state.cfg_scale, 1.0..=20.0).step_by(0.5))
        .on_hover_text("CFG scale: 1.0=no guidance, 7.5=default, 15.0+=strong adherence to prompt");
    ui.add_space(4.0);

    // Scheduler
    ui.label("Scheduler").on_hover_text("Noise schedule for diffusion denoising");
    ui.horizontal(|ui| {
        egui::ComboBox::from_id_salt("scheduler")
            .selected_text(state.scheduler.label())
            .show_ui(ui, |ui| {
                for &kind in SchedulerKind::ALL {
                    ui.selectable_value(&mut state.scheduler, kind, kind.label());
                }
            });
        ui.label("?").on_hover_text("LTX-2: default schedule. Linear-Quadratic: linear then quadratic decay. Beta: smooth S-curve.");
    });
    ui.add_space(4.0);

    // Device
    ui.label("Device").on_hover_text("Compute device for transformer inference");
    ui.horizontal(|ui| {
        egui::ComboBox::from_id_salt("device")
            .selected_text(state.device.label())
            .show_ui(ui, |ui| {
                for &kind in DeviceKind::all_available() {
                    ui.selectable_value(&mut state.device, kind, kind.label());
                }
            });
        ui.label("?").on_hover_text("CPU: always works. CUDA: requires NVIDIA GPU with CUDA libtorch installed. See README for setup.");
    });
    ui.add_space(8.0);

    // Generate button
    let can_generate = matches!(state.inference, crate::state::InferenceState::Idle | crate::state::InferenceState::Done | crate::state::InferenceState::Error(_));
    if can_generate {
        if ui
            .add_sized([ui.available_width(), 36.0], egui::Button::new("Generate"))
            .on_hover_text("Start video generation. Requires weights + tokenizer + text weights for best results.")
            .clicked()
        {
            let (cmd_tx, cmd_rx) = std::sync::mpsc::channel();
            let (event_tx, event_rx) = std::sync::mpsc::channel();

            let params = InferenceParams {
                prompt: state.prompt.clone(),
                weights_path: state.weights_path.clone(),
                tokenizer_path: state.tokenizer_path.clone(),
                text_weights_path: state.text_weights_path.clone(),
                height: state.height,
                width: state.width,
                frames: state.frames,
                steps: state.steps,
                cfg_scale: state.cfg_scale,
                scheduler: state.scheduler,
                device: state.device,
            };

            state.inference = crate::state::InferenceState::Loading;
            state.frames_display.clear();
            state.current_frame = 0;
            state.cmd_tx = Some(cmd_tx);
            state.event_rx = Some(event_rx);

            crate::inference::spawn_inference_thread(params, event_tx, cmd_rx);
        }
    } else {
        let can_cancel = matches!(
            state.inference,
            crate::state::InferenceState::Denoising { .. }
                | crate::state::InferenceState::Loading
                | crate::state::InferenceState::Decoding
        );
        if can_cancel && ui
            .add_sized([ui.available_width(), 36.0], egui::Button::new("Cancel"))
            .on_hover_text("Stop generation and discard current progress")
            .clicked()
        {
            if let Some(tx) = &state.cmd_tx {
                let _ = tx.send(GuiCommand::Cancel);
            }
        }
    }

    ui.add_space(8.0);

    // Export
    if !state.frames_display.is_empty() {
        ui.separator();
        ui.label("Export").on_hover_text("Save generated frames to disk");
        ui.horizontal(|ui| {
            if ui.button("Save PNGs").on_hover_text("Save each frame as a separate PNG file").clicked() {
                let dir = rfd::FileDialog::new()
                    .set_title("Save frames")
                    .pick_folder();
                if let Some(dir) = dir {
                    if let Err(e) = crate::export::save_frames_png(
                        &state.frames_display,
                        state.height,
                        state.width,
                        &dir,
                    ) {
                        state.inference = crate::state::InferenceState::Error(e);
                    }
                }
            }
            if ui.button("Save Video").on_hover_text("Save as MP4 video (H.264 encoded, 8fps)").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("MP4", &["mp4"])
                    .save_file()
                {
                    let dir = path.parent().unwrap_or(std::path::Path::new("."));
                    match crate::export::save_video_mp4(
                        &state.frames_display,
                        state.height,
                        state.width,
                        dir,
                        &path,
                    ) {
                        Ok(_) => state.output_dir = Some(path),
                        Err(e) => state.inference = crate::state::InferenceState::Error(e),
                    }
                }
            }
            if ui.button("Save GIF").on_hover_text("Save as animated GIF (256×256, lanczos scaling)").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("GIF", &["gif"])
                    .save_file()
                {
                    let dir = path.parent().unwrap_or(std::path::Path::new("."));
                    match crate::export::save_gif(
                        &state.frames_display,
                        state.height,
                        state.width,
                        dir,
                        &path,
                        8,
                        256,
                    ) {
                        Ok(_) => state.output_dir = Some(path),
                        Err(e) => state.inference = crate::state::InferenceState::Error(e),
                    }
                }
            }
        });
    }
}
