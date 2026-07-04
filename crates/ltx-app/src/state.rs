use std::path::PathBuf;
use std::sync::mpsc;

pub const DEFAULT_HEIGHT: i64 = 16;
pub const DEFAULT_WIDTH: i64 = 16;
pub const DEFAULT_FRAMES: i64 = 4;
pub const DEFAULT_STEPS: usize = 20;
pub const DEFAULT_CFG: f64 = 7.5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerKind {
    Ltx2,
    LinearQuadratic,
    Beta,
}

impl SchedulerKind {
    pub const ALL: &'static [SchedulerKind] = &[
        SchedulerKind::Ltx2,
        SchedulerKind::LinearQuadratic,
        SchedulerKind::Beta,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Ltx2 => "LTX-2",
            Self::LinearQuadratic => "Linear-Quadratic",
            Self::Beta => "Beta",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    Cpu,
    Cuda,
}

impl DeviceKind {
    pub const ALL: &'static [DeviceKind] = &[DeviceKind::Cpu, DeviceKind::Cuda];

    pub fn label(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Cuda => "CUDA",
        }
    }
}

#[derive(Debug, Clone)]
pub struct InferenceParams {
    #[allow(dead_code)]
    pub prompt: String,
    pub weights_path: Option<PathBuf>,
    pub height: i64,
    pub width: i64,
    pub frames: i64,
    pub steps: usize,
    pub cfg_scale: f64,
    pub scheduler: SchedulerKind,
    pub device: DeviceKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum InferenceState {
    Idle,
    Loading,
    Denoising { step: usize, total: usize, sigma: f64 },
    Decoding,
    Done,
    Error(String),
}

pub enum GuiEvent {
    Progress { step: usize, total: usize, sigma: f64 },
    Decoding,
    FramesReady(Vec<Vec<u8>>),
    Error(String),
}

pub enum GuiCommand {
    Cancel,
}

pub struct AppState {
    pub prompt: String,
    pub weights_path: Option<PathBuf>,
    pub height: i64,
    pub width: i64,
    pub frames: i64,
    pub steps: usize,
    pub cfg_scale: f64,
    pub scheduler: SchedulerKind,
    pub device: DeviceKind,
    pub inference: InferenceState,
    pub frames_display: Vec<Vec<u8>>,
    pub current_frame: usize,
    pub playing: bool,
    pub fps: f64,
    pub output_dir: Option<PathBuf>,
    pub event_rx: Option<mpsc::Receiver<GuiEvent>>,
    pub cmd_tx: Option<mpsc::Sender<GuiCommand>>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            prompt: String::from("a colorful abstract pattern"),
            weights_path: None,
            height: DEFAULT_HEIGHT,
            width: DEFAULT_WIDTH,
            frames: DEFAULT_FRAMES,
            steps: DEFAULT_STEPS,
            cfg_scale: DEFAULT_CFG,
            scheduler: SchedulerKind::Ltx2,
            device: DeviceKind::Cpu,
            inference: InferenceState::Idle,
            frames_display: Vec::new(),
            current_frame: 0,
            playing: false,
            fps: 8.0,
            output_dir: None,
            event_rx: None,
            cmd_tx: None,
        }
    }
}
