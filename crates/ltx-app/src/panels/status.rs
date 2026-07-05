use eframe::egui;

use crate::state::{AppState, InferenceState};

pub fn show(ui: &mut egui::Ui, state: &AppState) {
    ui.horizontal(|ui| {
        let (label, tooltip, color) = match &state.inference {
            InferenceState::Idle => ("Ready".to_string(), "No generation in progress".to_string(), egui::Color32::GRAY),
            InferenceState::Loading => ("Loading…".to_string(), "Loading model weights".to_string(), egui::Color32::YELLOW),
            InferenceState::Denoising { step, total, sigma } => {
                let pct = (*step as f64 / *total as f64 * 100.0) as u32;
                (format!("Step {step}/{total} ({pct}%)  σ={sigma:.4}"),
                 format!("Denoising step {step} of {total} ({pct}% complete)"),
                 egui::Color32::from_rgb(100, 180, 255))
            }
            InferenceState::Decoding => ("Decoding…".to_string(), "Converting latent to pixel space".to_string(), egui::Color32::YELLOW),
            InferenceState::Done => ("Done".to_string(), "Generation complete — frames ready for preview and export".to_string(), egui::Color32::from_rgb(100, 220, 100)),
            InferenceState::Error(e) => (format!("Error: {e}"), e.clone(), egui::Color32::RED),
        };

        ui.label(egui::RichText::new(label).color(color)).on_hover_text(tooltip);
        ui.separator();

        // Dimensions
        if !state.frames_display.is_empty() {
            ui.label(format!("{}×{}×{}", state.width, state.height, state.frames_display.len()))
                .on_hover_text(format!("Width × Height × Frames ({} total pixels)", state.width * state.height * state.frames_display.len() as i64));
        } else {
            ui.label(format!("{}×{}×{}", state.width, state.height, state.frames))
                .on_hover_text("Width × Height × Frames".to_string());
        }
    });
}
