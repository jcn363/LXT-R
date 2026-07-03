use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum Modality {
    Video,
    Audio,
    Image,
}

impl Modality {
    pub fn is_video(&self) -> bool {
        matches!(self, Modality::Video)
    }

    pub fn is_audio(&self) -> bool {
        matches!(self, Modality::Audio)
    }

    pub fn is_image(&self) -> bool {
        matches!(self, Modality::Image)
    }

    pub fn ndim(&self) -> i64 {
        match self {
            Modality::Video => 5,  // B, C, F, H, W
            Modality::Audio => 4,  // B, C, T, F
            Modality::Image => 4,  // B, C, H, W
        }
    }
}
