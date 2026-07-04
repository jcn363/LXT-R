use eframe::egui;

use crate::state::{AppState, InferenceState};

pub fn show(ui: &mut egui::Ui, state: &AppState) {
    ui.horizontal(|ui| {
        let (label, color) = match &state.inference {
            InferenceState::Idle => ("Ready".to_string(), egui::Color32::GRAY),
            InferenceState::Loading => ("Loading…".to_string(), egui::Color32::YELLOW),
            InferenceState::Denoising { step, total, sigma } => {
                let pct = (*step as f64 / *total as f64 * 100.0) as u32;
                (format!("Step {step}/{total} ({pct}%)  σ={sigma:.4}"), egui::Color32::from_rgb(100, 180, 255))
            }
            InferenceState::Decoding => ("Decoding…".to_string(), egui::Color32::YELLOW),
            InferenceState::Done => ("Done".to_string(), egui::Color32::from_rgb(100, 220, 100)),
            InferenceState::Error(e) => (format!("Error: {e}"), egui::Color32::RED),
        };

        ui.label(egui::RichText::new(label).color(color));
        ui.separator();

        // Dimensions
        if !state.frames_display.is_empty() {
            ui.label(format!(
                "{}×{}×{}",
                state.width, state.height, state.frames_display.len()
            ));
        } else {
            ui.label(format!("{}×{}×{}", state.width, state.height, state.frames));
        }
    });
}
