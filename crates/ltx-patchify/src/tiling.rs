use tch::Tensor;

use ltx_types::{TilingConfig, MIN_SPATIAL_OVERLAP_PX, MIN_TEMPORAL_OVERLAP_FRAMES};

/// A single tile in a tiled processing layout.
///
/// Each tile specifies the pixel-space crop region and its corresponding
/// position in the latent grid.
#[derive(Debug, Clone, PartialEq)]
pub struct Tile {
    /// Latent-space temporal start index.
    pub t_start: i64,
    /// Latent-space temporal end index (exclusive).
    pub t_end: i64,
    /// Latent-space height start index.
    pub h_start: i64,
    /// Latent-space height end index (exclusive).
    pub h_end: i64,
    /// Latent-space width start index.
    pub w_start: i64,
    /// Latent-space width end index (exclusive).
    pub w_end: i64,
}

impl Tile {
    pub fn new(t_start: i64, t_end: i64, h_start: i64, h_end: i64, w_start: i64, w_end: i64) -> Self {
        Self { t_start, t_end, h_start, h_end, w_start, w_end }
    }

    /// Temporal extent of this tile (in latent frames).
    pub fn temporal_len(&self) -> i64 {
        self.t_end - self.t_start
    }

    /// Height extent of this tile (in latent pixels).
    pub fn height_len(&self) -> i64 {
        self.h_end - self.h_start
    }

    /// Width extent of this tile (in latent pixels).
    pub fn width_len(&self) -> i64 {
        self.w_end - self.w_start
    }
}

/// Generate tile grid coordinates from a tiling config and latent spatial dims.
///
/// Returns a `Vec<Tile>` covering the full latent volume with the specified
/// tile sizes and overlaps.
pub fn compute_tile_grid(
    config: &TilingConfig,
    latent_t: i64,
    latent_h: i64,
    latent_w: i64,
    time_scale: i64,
    height_scale: i64,
    width_scale: i64,
) -> Vec<Tile> {
    let tile_size_t = config.tile_size_frames / time_scale;
    let tile_overlap_t = config.tile_overlap_frames / time_scale;
    let tile_size_h = config.tile_size_px / height_scale;
    let tile_overlap_h = config.tile_overlap_px / width_scale;
    let tile_size_w = config.tile_size_px / width_scale;
    let tile_overlap_w = config.tile_overlap_px / width_scale;

    let mut tiles = Vec::new();
    let mut t = 0i64;
    while t < latent_t {
        let t_end = (t + tile_size_t).min(latent_t);
        let mut h = 0i64;
        while h < latent_h {
            let h_end = (h + tile_size_h).min(latent_h);
            let mut w = 0i64;
            while w < latent_w {
                let w_end = (w + tile_size_w).min(latent_w);
                tiles.push(Tile::new(t, t_end, h, h_end, w, w_end));
                w = if w_end == latent_w { latent_w } else { w_end - tile_overlap_w };
            }
            h = if h_end == latent_h { latent_h } else { h_end - tile_overlap_h };
        }
        t = if t_end == latent_t { latent_t } else { t_end - tile_overlap_t };
    }
    tiles
}

/// Compute trapezoidal blending weights for temporal overlap between tiles.
///
/// The weight ramps linearly from 0.0 at the overlap edge to 1.0 in the
/// non-overlapping region, producing smooth blending when tiles are composited.
///
/// # Arguments
/// * `tile_len` - Total tile length in latent frames
/// * `overlap_len` - Overlap region length in latent frames
/// * `is_first` - Whether this tile is the first in the temporal sequence
/// * `is_last` - Whether this tile is the last in the temporal sequence
pub fn trapezoidal_mask(
    tile_len: i64,
    overlap_len: i64,
    is_first: bool,
    is_last: bool,
) -> Tensor {
    let weights = Tensor::ones([tile_len], (tch::Kind::Float, tch::Device::Cpu));

    if !is_first && overlap_len > 0 {
        let ramp = Tensor::arange(overlap_len, (tch::Kind::Float, tch::Device::Cpu))
            / overlap_len as f64;
        let left = weights.narrow(0, 0, overlap_len) * &ramp;
        let _ = weights.narrow(0, 0, overlap_len).copy_(&left);
    }

    if !is_last && overlap_len > 0 {
        let start = tile_len - overlap_len;
        let ramp = 1.0
            - Tensor::arange(overlap_len, (tch::Kind::Float, tch::Device::Cpu))
                / overlap_len as f64;
        let right = weights.narrow(0, start, overlap_len) * &ramp;
        let _ = weights.narrow(0, start, overlap_len).copy_(&right);
    }

    weights
}

/// Validate tiling config against minimum overlap constraints.
///
/// Returns `Ok(())` if overlaps meet minimums, or an error message listing
/// which constraints are violated.
pub fn validate_tiling_config(config: &TilingConfig) -> Result<(), String> {
    let mut errors = Vec::new();

    if config.tile_overlap_px < MIN_SPATIAL_OVERLAP_PX {
        errors.push(format!(
            "tile_overlap_px ({}) < MIN_SPATIAL_OVERLAP_PX ({})",
            config.tile_overlap_px, MIN_SPATIAL_OVERLAP_PX,
        ));
    }
    if config.tile_overlap_frames < MIN_TEMPORAL_OVERLAP_FRAMES {
        errors.push(format!(
            "tile_overlap_frames ({}) < MIN_TEMPORAL_OVERLAP_FRAMES ({})",
            config.tile_overlap_frames, MIN_TEMPORAL_OVERLAP_FRAMES,
        ));
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("; "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_tile_extents() {
        let tile = Tile::new(0, 8, 0, 16, 0, 16);
        assert_eq!(tile.temporal_len(), 8);
        assert_eq!(tile.height_len(), 16);
        assert_eq!(tile.width_len(), 16);
    }

    #[test]
    fn test_compute_tile_grid_single_tile() {
        let config = TilingConfig {
            tile_size_px: 512,
            tile_overlap_px: 64,
            tile_size_frames: 64,
            tile_overlap_frames: 24,
        };
        // latent 4x16x16 with default scales (8,32,32)
        // tile_size_t = 64/8 = 8, tile_size_h = 512/32 = 16, tile_size_w = 512/32 = 16
        let tiles = compute_tile_grid(&config, 4, 16, 16, 8, 32, 32);
        // Should be 1 tile since latent dims fit within tile
        assert_eq!(tiles.len(), 1);
        assert_eq!(tiles[0], Tile::new(0, 4, 0, 16, 0, 16));
    }

    #[test]
    fn test_compute_tile_grid_multiple() {
        let config = TilingConfig {
            tile_size_px: 256,
            tile_overlap_px: 64,
            tile_size_frames: 32,
            tile_overlap_frames: 16,
        };
        // tile_size_t = 32/8 = 4, tile_size_h = 256/32 = 8
        // latent 8x16x16 → need 2x2 tiles
        let tiles = compute_tile_grid(&config, 8, 16, 16, 8, 32, 32);
        assert!(tiles.len() > 1);
    }

    #[test]
    fn test_trapezoidal_mask_no_overlap() {
        let mask = trapezoidal_mask(8, 0, true, true);
        assert_eq!(mask.size(), vec![8]);
        let expected = Tensor::ones([8], (tch::Kind::Float, tch::Device::Cpu));
        assert!(mask.allclose(&expected, 1e-6, 1e-6, false));
    }

    #[test]
    fn test_trapezoidal_mask_first_tile() {
        let mask = trapezoidal_mask(8, 2, true, false);
        assert_eq!(mask.size(), vec![8]);
        // First tile: no ramp on left, ramp down on right
        assert_eq!(mask.double_value(&[0]), 1.0);
        assert_eq!(mask.double_value(&[6]), 0.5);
        assert_eq!(mask.double_value(&[7]), 0.0);
    }

    #[test]
    fn test_trapezoidal_mask_last_tile() {
        let mask = trapezoidal_mask(8, 2, false, true);
        assert_eq!(mask.size(), vec![8]);
        // Last tile: ramp up on left, no ramp on right
        assert_eq!(mask.double_value(&[0]), 0.0);
        assert_eq!(mask.double_value(&[1]), 0.5);
        assert_eq!(mask.double_value(&[7]), 1.0);
    }

    #[test]
    fn test_validate_tiling_config_ok() {
        let config = TilingConfig::default();
        assert!(validate_tiling_config(&config).is_ok());
    }

    #[test]
    fn test_validate_tiling_config_bad_overlap() {
        let config = TilingConfig {
            tile_overlap_px: 32,
            tile_overlap_frames: 8,
            ..Default::default()
        };
        let err = validate_tiling_config(&config).unwrap_err();
        assert!(err.contains("tile_overlap_px"));
        assert!(err.contains("tile_overlap_frames"));
    }
}
