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
        ui.image((texture_handle.id(), display_size))
            .on_hover_text(format!(
                "Frame {} of {} ({}×{} pixels)",
                state.current_frame + 1,
                total,
                w,
                h
            ));
    });

    // --- Toolbar ---
    ui.add_space(4.0);
    ui.separator();

    ui.horizontal(|ui| {
        // Play / Pause
        let play_icon = if state.playing { "⏸" } else { "▶" };
        if ui
            .button(play_icon)
            .on_hover_text(if state.playing {
                "Pause playback"
            } else {
                "Play animation"
            })
            .clicked()
        {
            state.playing = !state.playing;
        }

        // Frame counter
        ui.label(format!("{:>3} / {}", state.current_frame + 1, total))
            .on_hover_text("Current frame / total frames");

        // Scrubber
        let scrubber = ui
            .add(
                egui::Slider::new(&mut state.current_frame, 0..=total.saturating_sub(1))
                    .show_value(false),
            )
            .on_hover_text("Drag to scrub through frames (pauses playback)");
        if scrubber.changed() {
            state.playing = false;
        }

        ui.separator();

        // FPS control
        ui.label("FPS").on_hover_text("Playback frames per second");
        ui.add(
            egui::DragValue::new(&mut state.fps)
                .range(1.0..=60.0)
                .prefix(""),
        )
        .on_hover_text("Adjust animation speed (1-60 fps)");

        // Speed indicator
        ui.label(format!("{:.1}×", state.fps / 8.0))
            .on_hover_text("Speed relative to 8fps default");

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui
                .button("⬇ GIF")
                .on_hover_text("Export as animated GIF (256×256)")
                .clicked()
            {
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
                        Ok(_) => {}
                        Err(e) => state.inference = crate::state::InferenceState::Error(e),
                    }
                }
            }
            if ui
                .button("⬇ Video")
                .on_hover_text("Export as MP4 video (H.264)")
                .clicked()
            {
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
                        Ok(_) => {}
                        Err(e) => state.inference = crate::state::InferenceState::Error(e),
                    }
                }
            }
            if ui
                .button("⬇ PNGs")
                .on_hover_text("Export as individual PNG files")
                .clicked()
            {
                if let Some(dir) = rfd::FileDialog::new()
                    .set_title("Save frames")
                    .pick_folder()
                {
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
        });
    });
}
