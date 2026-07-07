use eframe::egui;

use crate::panels;
use crate::state::{AppState, GuiEvent, InferenceState};

#[derive(Default)]
pub struct LtxApp {
    state: AppState,
}

impl eframe::App for LtxApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_events();

        // Status bar (top)
        egui::TopBottomPanel::top("status").show(ctx, |ui| {
            ui.add_space(4.0);
            panels::status::show(ui, &self.state);
            ui.add_space(4.0);
        });

        // Sidebar (left)
        egui::SidePanel::left("sidebar")
            .default_width(260.0)
            .min_width(220.0)
            .show(ctx, |ui| {
                ui.add_space(8.0);
                panels::sidebar::show(ui, &mut self.state);
            });

        // Viewport (center)
        egui::CentralPanel::default().show(ctx, |ui| {
            panels::viewport::show(ui, &mut self.state, ctx);
        });

        // Repaint while inference is running
        if matches!(
            self.state.inference,
            InferenceState::Loading | InferenceState::Denoising { .. } | InferenceState::Decoding
        ) {
            ctx.request_repaint();
        }
    }
}

impl LtxApp {
    fn poll_events(&mut self) {
        if let Some(rx) = &self.state.event_rx {
            while let Ok(event) = rx.try_recv() {
                match event {
                    GuiEvent::Progress { step, total, sigma } => {
                        self.state.inference = InferenceState::Denoising { step, total, sigma };
                    }
                    GuiEvent::Decoding => {
                        self.state.inference = InferenceState::Decoding;
                    }
                    GuiEvent::FramesReady(frames) => {
                        self.state.frames_display = frames;
                        self.state.current_frame = 0;
                        self.state.inference = InferenceState::Done;
                    }
                    GuiEvent::Error(e) => {
                        self.state.inference = InferenceState::Error(e);
                    }
                }
            }
        }
    }
}
