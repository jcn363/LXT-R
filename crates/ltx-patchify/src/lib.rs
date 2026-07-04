//! Patchification operations for the LTX-2.3 Rust rewrite.
//!
//! Provides patchify/unpatchify ops for 4D, 5D, and audio tensors,
//! video and audio patchifiers, tiling utilities, and coordinate helpers.

pub mod audio_patchifier;
pub mod coords;
pub mod ops;
pub mod tiling;
pub mod video_patchifier;

// Re-export for single import path
pub use audio_patchifier::{AudioPatchifier, AudioTiming};
pub use coords::{
    get_patch_grid_bounds, get_pixel_coords, num_patches, patch_bounds_to_pixel_bounds,
};
pub use ops::{
    patchify_4d, patchify_5d, patchify_audio, unpatchify_4d, unpatchify_5d, unpatchify_audio,
};
pub use tiling::{compute_tile_grid, trapezoidal_mask, validate_tiling_config, Tile};
pub use video_patchifier::VideoLatentPatchifier;
