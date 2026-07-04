mod app;
mod export;
mod inference;
mod panels;
mod state;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([1024.0, 768.0])
            .with_min_inner_size([640.0, 480.0])
            .with_title("LTX-R"),
        ..Default::default()
    };

    eframe::run_native(
        "LTX-R",
        options,
        Box::new(|cc| {
            // Minimalist styling
            let mut style = (*cc.egui_ctx.style()).clone();
            style.spacing.item_spacing = eframe::egui::vec2(8.0, 6.0);
            cc.egui_ctx.set_style(style);
            Ok(Box::new(app::LtxApp::default()))
        }),
    )
}
