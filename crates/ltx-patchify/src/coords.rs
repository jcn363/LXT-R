use ltx_types::PatchGridBounds;

/// Compute pixel-space coordinates for each patch in a grid.
///
/// Returns a vector of `(pixel_x, pixel_y)` tuples, one per spatial patch
/// position. The coordinates represent the top-left corner of each patch in
/// the original pixel space.
///
/// # Arguments
/// * `latent_h` - Height of the latent grid (number of patches along height)
/// * `latent_w` - Width of the latent grid (number of patches along width)
/// * `height_scale` - Pixel-to-latent scale factor for height
/// * `width_scale` - Pixel-to-latent scale factor for width
pub fn get_pixel_coords(
    latent_h: i64,
    latent_w: i64,
    height_scale: i64,
    width_scale: i64,
) -> Vec<(i64, i64)> {
    let mut coords = Vec::with_capacity((latent_h * latent_w) as usize);
    for h in 0..latent_h {
        for w in 0..latent_w {
            coords.push((w * width_scale, h * height_scale));
        }
    }
    coords
}

/// Compute the patch grid bounds for a given latent volume.
///
/// Returns bounds indicating the min/max patch indices in each dimension,
/// useful for determining which patches need to be processed in a tiled
/// fashion.
pub fn get_patch_grid_bounds(
    latent_t: i64,
    latent_h: i64,
    latent_w: i64,
) -> PatchGridBounds {
    PatchGridBounds::new(0, latent_t, 0, latent_h, 0, latent_w)
}

/// Compute pixel-space bounds from patch grid bounds.
///
/// Converts latent patch indices back to pixel coordinates using the
/// standard scale factors.
pub fn patch_bounds_to_pixel_bounds(
    bounds: &PatchGridBounds,
    time_scale: i64,
    height_scale: i64,
    width_scale: i64,
) -> (i64, i64, i64, i64, i64, i64) {
    (
        bounds.min_t * time_scale,
        bounds.max_t * time_scale,
        bounds.min_h * height_scale,
        bounds.max_h * height_scale,
        bounds.min_w * width_scale,
        bounds.max_w * width_scale,
    )
}

/// Number of patches along each axis given pixel dimensions and scale.
pub fn num_patches(
    pixel_t: i64,
    pixel_h: i64,
    pixel_w: i64,
    time_scale: i64,
    height_scale: i64,
    width_scale: i64,
) -> (i64, i64, i64) {
    (
        pixel_t / time_scale,
        pixel_h / height_scale,
        pixel_w / width_scale,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ltx_types::{DEFAULT_HEIGHT_SCALE, DEFAULT_TIME_SCALE, DEFAULT_WIDTH_SCALE};

    #[test]
    fn test_get_pixel_coords() {
        let coords = get_pixel_coords(2, 2, 32, 32);
        assert_eq!(coords.len(), 4);
        // Row-major: (0,0), (32,0), (0,32), (32,32)
        assert_eq!(coords[0], (0, 0));
        assert_eq!(coords[1], (32, 0));
        assert_eq!(coords[2], (0, 32));
        assert_eq!(coords[3], (32, 32));
    }

    #[test]
    fn test_get_patch_grid_bounds() {
        let bounds = get_patch_grid_bounds(4, 16, 16);
        assert_eq!(bounds.min_t, 0);
        assert_eq!(bounds.max_t, 4);
        assert_eq!(bounds.min_h, 0);
        assert_eq!(bounds.max_h, 16);
        assert_eq!(bounds.min_w, 0);
        assert_eq!(bounds.max_w, 16);
    }

    #[test]
    fn test_patch_bounds_to_pixel_bounds() {
        let bounds = PatchGridBounds::new(0, 4, 0, 16, 0, 16);
        let (pt0, pt1, ph0, ph1, pw0, pw1) =
            patch_bounds_to_pixel_bounds(&bounds, DEFAULT_TIME_SCALE, DEFAULT_HEIGHT_SCALE, DEFAULT_WIDTH_SCALE);
        assert_eq!((pt0, pt1), (0, 32));
        assert_eq!((ph0, ph1), (0, 512));
        assert_eq!((pw0, pw1), (0, 512));
    }

    #[test]
    fn test_num_patches() {
        let (t, h, w) = num_patches(64, 512, 512, DEFAULT_TIME_SCALE, DEFAULT_HEIGHT_SCALE, DEFAULT_WIDTH_SCALE);
        assert_eq!((t, h, w), (8, 16, 16));
    }
}
