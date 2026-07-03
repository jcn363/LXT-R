use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct VideoLatentShape {
    pub batch: i64,
    pub channels: i64,
    pub frames: i64,
    pub height: i64,
    pub width: i64,
}

impl VideoLatentShape {
    pub fn new(batch: i64, channels: i64, frames: i64, height: i64, width: i64) -> Self {
        Self { batch, channels, frames, height, width }
    }

    pub fn spatial_dim(&self) -> i64 {
        self.height * self.width
    }

    pub fn temporal_dim(&self) -> i64 {
        self.frames
    }

    pub fn flatten_spatial(&self) -> i64 {
        self.frames * self.height * self.width
    }

    pub fn to_vec(&self) -> Vec<i64> {
        vec![self.batch, self.channels, self.frames, self.height, self.width]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct AudioLatentShape {
    pub batch: i64,
    pub channels: i64,
    pub time: i64,
    pub features: i64,
}

impl AudioLatentShape {
    pub fn new(batch: i64, channels: i64, time: i64, features: i64) -> Self {
        Self { batch, channels, time, features }
    }

    pub fn to_vec(&self) -> Vec<i64> {
        vec![self.batch, self.channels, self.time, self.features]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct PatchGridBounds {
    pub min_t: i64,
    pub max_t: i64,
    pub min_h: i64,
    pub max_h: i64,
    pub min_w: i64,
    pub max_w: i64,
}

impl PatchGridBounds {
    pub fn new(min_t: i64, max_t: i64, min_h: i64, max_h: i64, min_w: i64, max_w: i64) -> Self {
        Self { min_t, max_t, min_h, max_h, min_w, max_w }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct TilingConfig {
    pub tile_size_px: i64,
    pub tile_overlap_px: i64,
    pub tile_size_frames: i64,
    pub tile_overlap_frames: i64,
}

impl Default for TilingConfig {
    fn default() -> Self {
        Self {
            tile_size_px: crate::constants::DEFAULT_TILE_SIZE_PX,
            tile_overlap_px: crate::constants::DEFAULT_TILE_OVERLAP_PX,
            tile_size_frames: crate::constants::DEFAULT_TILE_SIZE_FRAMES,
            tile_overlap_frames: crate::constants::DEFAULT_TILE_OVERLAP_FRAMES,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct TransformerArgs {
    pub num_layers: i64,
    pub num_heads: i64,
    pub head_dim: i64,
    pub hidden_dim: i64,
    pub intermediate_dim: i64,
    pub context_dim: Option<i64>,
    pub use_rope: bool,
    pub rope_type: String,
    pub max_seq_len: i64,
}

impl Default for TransformerArgs {
    fn default() -> Self {
        Self {
            num_layers: 28,
            num_heads: 8,
            head_dim: 128,
            hidden_dim: 1024,
            intermediate_dim: 4096,
            context_dim: None,
            use_rope: true,
            rope_type: "interleaved".to_string(),
            max_seq_len: 2048,
        }
    }
}
