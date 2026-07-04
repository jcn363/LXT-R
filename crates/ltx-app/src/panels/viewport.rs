use eframe::egui;

use crate::state::AppState;

pub fn show(ui: &mut egui::Ui, state: &mut AppState, ctx: &egui::Context) {
    if state.frames_display.is_empty() {
        ui.centered_and_justified(|ui| {
            ui.label(
                egui::RichText::new("No frames yet.\nConfigure and click Generate.")
                    .size(16.0)
                    .color(ui.visuals().weak_text_color()),
            );
        });
        return;
    }

    let total = state.frames_display.len();
    if state.current_frame >= total {
        state.current_frame = 0;
    }

    // --- Video playback tick ---
    if state.playing {
        ctx.request_repaint();
        let dt = ctx.input(|i| i.predicted_dt);
        let frame_duration = 1.0 / state.fps as f32;
        let advance = (dt / frame_duration) as usize;
        if advance > 0 {
            state.current_frame += advance;
            if state.current_frame >= total {
                state.current_frame = 0;
            }
        }
    }

    let frame_data = &state.frames_display[state.current_frame];
    let h = state.height as u32;
    let w = state.width as u32;

    // Create texture from frame data
    let image = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &{
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for c in frame_data.chunks(3) {
            rgba.extend_from_slice(&[c[0], c[1], c[2], 255]);
        }
        rgba
    });

    let texture_handle = ctx.load_texture("frame", image, egui::TextureOptions::NEAREST);

    // Scale up for display
    let available = ui.available_size();
    let scale = ((available.x / w as f32).min((available.y - 80.0) / h as f32)).clamp(1.0, 16.0);
    let display_size = egui::vec2(w as f32 * scale, h as f32 * scale);

    // Center the frame
    ui.vertical_centered(|ui| {
        ui.add_space(8.0);
        ui.image((texture_handle.id(), display_size));
    });

    // --- Toolbar ---
    ui.add_space(4.0);
    ui.separator();

    ui.horizontal(|ui| {
        // Play / Pause
        let play_icon = if state.playing { "⏸" } else { "▶" };
        if ui.button(play_icon).clicked() {
            state.playing = !state.playing;
        }

        // Frame counter
        ui.label(format!("{:>3} / {}", state.current_frame + 1, total));

        // Scrubber
        let scrubber = ui.add(
            egui::Slider::new(&mut state.current_frame, 0..=total.saturating_sub(1))
                .show_value(false),
        );
        if scrubber.changed() {
            state.playing = false;
        }

        ui.separator();

        // FPS control
        ui.label("FPS");
        ui.add(
            egui::DragValue::new(&mut state.fps)
                .range(1.0..=60.0)
                .prefix(""),
        );

        // Speed indicator
        ui.label(format!("{:.1}×", state.fps / 8.0));
    });
}
