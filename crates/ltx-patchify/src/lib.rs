pub mod audio_patchifier;
pub mod coords;
pub mod ops;
pub mod tiling;
pub mod video_patchifier;

// Re-export for single import path
pub use audio_patchifier::{AudioPatchifier, AudioTiming};
pub use coords::{get_patch_grid_bounds, get_pixel_coords};
pub use ops::{
    patchify_4d, patchify_5d, patchify_audio, unpatchify_4d, unpatchify_5d, unpatchify_audio,
};
pub use tiling::{compute_tile_grid, trapezoidal_mask, validate_tiling_config, Tile};
pub use video_patchifier::VideoLatentPatchifier;
