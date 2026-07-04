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

    let frame_data = &state.frames_display[state.current_frame];
    let h = state.height as u32;
    let w = state.width as u32;

    // Create texture from frame data
    let pixels: Vec<egui::Color32> = frame_data
        .chunks(3)
        .map(|c| egui::Color32::from_rgb(c[0], c[1], c[2]))
        .collect();

    let image = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &{
        let mut rgba = Vec::with_capacity(pixels.len() * 4);
        for p in &pixels {
            rgba.push(p.r());
            rgba.push(p.g());
            rgba.push(p.b());
            rgba.push(255);
        }
        rgba
    });

    let texture_handle = ctx.load_texture(
        "frame",
        image,
        egui::TextureOptions::NEAREST,
    );

    // Scale up for display (nearest-neighbor for pixel art feel)
    let available = ui.available_size();
    let scale = ((available.x / w as f32).min(available.y / h as f32)).clamp(1.0, 16.0);
    let display_size = egui::vec2(w as f32 * scale, h as f32 * scale);

    ui.vertical_centered(|ui| {
        ui.image((texture_handle.id(), display_size));

        ui.add_space(8.0);

        // Frame navigation
        ui.horizontal(|ui| {
            if ui.button("◀").clicked() && state.current_frame > 0 {
                state.current_frame -= 1;
            }
            ui.label(format!("{} / {}", state.current_frame + 1, total));
            if ui.button("▶").clicked() && state.current_frame + 1 < total {
                state.current_frame += 1;
            }
        });
    });
}
