use eframe::egui;

use crate::state::{AppState, DeviceKind, GuiCommand, InferenceParams, SchedulerKind};

pub fn show(ui: &mut egui::Ui, state: &mut AppState) {
    ui.heading("Controls");
    ui.separator();

    // Prompt
    ui.label("Prompt");
    ui.text_edit_singleline(&mut state.prompt);
    ui.add_space(4.0);

    // Weights
    ui.label("Model Weights");
    ui.horizontal(|ui| {
        let label = match &state.weights_path {
            Some(p) => p.file_name().unwrap_or_default().to_string_lossy().to_string(),
            None => "None (random init)".to_string(),
        };
        ui.label(&label);
        if ui.button("Browse…").clicked() {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("SafeTensors", &["safetensors"])
                .pick_file()
            {
                state.weights_path = Some(path);
            }
        }
        if state.weights_path.is_some() && ui.button("×").clicked() {
            state.weights_path = None;
        }
    });
    ui.add_space(4.0);

    // Text Encoder
    ui.label("Text Encoder (optional)");
    ui.horizontal(|ui| {
        let label = match &state.tokenizer_path {
            Some(p) => p.file_name().unwrap_or_default().to_string_lossy().to_string(),
            None => "No tokenizer".to_string(),
        };
        ui.label(&label);
        if ui.button("Browse…").clicked() {
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
        ui.label(&label);
        if ui.button("Browse…").clicked() {
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
    ui.label("Resolution");
    ui.horizontal(|ui| {
        ui.label("H");
        ui.add(egui::DragValue::new(&mut state.height).range(4..=256).suffix("px"));
        ui.label("W");
        ui.add(egui::DragValue::new(&mut state.width).range(4..=256).suffix("px"));
    });
    ui.horizontal(|ui| {
        ui.label("Frames");
        ui.add(egui::DragValue::new(&mut state.frames).range(1..=128));
    });
    ui.add_space(4.0);

    // Inference params
    ui.label("Steps");
    ui.add(egui::Slider::new(&mut state.steps, 1..=100));
    ui.label("CFG Scale");
    ui.add(egui::Slider::new(&mut state.cfg_scale, 1.0..=20.0).step_by(0.5));
    ui.add_space(4.0);

    // Scheduler
    ui.label("Scheduler");
    egui::ComboBox::from_id_salt("scheduler")
        .selected_text(state.scheduler.label())
        .show_ui(ui, |ui| {
            for &kind in SchedulerKind::ALL {
                ui.selectable_value(&mut state.scheduler, kind, kind.label());
            }
        });
    ui.add_space(4.0);

    // Device
    ui.label("Device");
    egui::ComboBox::from_id_salt("device")
        .selected_text(state.device.label())
        .show_ui(ui, |ui| {
            for &kind in DeviceKind::ALL {
                ui.selectable_value(&mut state.device, kind, kind.label());
            }
        });
    ui.add_space(8.0);

    // Generate button
    let can_generate = matches!(state.inference, crate::state::InferenceState::Idle | crate::state::InferenceState::Done | crate::state::InferenceState::Error(_));
    if can_generate {
        if ui
            .add_sized([ui.available_width(), 36.0], egui::Button::new("Generate"))
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
        ui.label("Export");
        ui.horizontal(|ui| {
            if ui.button("Save PNGs").clicked() {
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
            if ui.button("Save Video").clicked() {
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
            if ui.button("Save GIF").clicked() {
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
