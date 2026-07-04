use tch::Tensor;

/// Pixel shuffle for N-dimensional spatial upsampling.
///
/// Rearranges elements from `(B, C * r^n, D1, D2, ...)` to `(B, C, D1 * r, D2 * r, ...)`
/// where `n` is the number of spatial dimensions and `r` is the upscale factor.
///
/// This is the inverse of space-to-depth: channel groups are unrolled into
/// spatial positions, effectively trading channel capacity for resolution.
///
/// Pure tensor op — no learnable parameters.
pub struct PixelShuffleND {
    upscale_factor: i64,
    num_spatial_dims: i64,
}

impl std::fmt::Debug for PixelShuffleND {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PixelShuffleND")
            .field("upscale_factor", &self.upscale_factor)
            .field("num_spatial_dims", &self.num_spatial_dims)
            .finish()
    }
}

impl PixelShuffleND {
    pub fn new(upscale_factor: i64, num_spatial_dims: i64) -> Self {
        assert!(upscale_factor > 0, "upscale_factor must be positive");
        assert!(
            (2..=3).contains(&num_spatial_dims),
            "num_spatial_dims must be 2 or 3"
        );
        Self {
            upscale_factor,
            num_spatial_dims,
        }
    }

    pub fn upscale_factor(&self) -> i64 {
        self.upscale_factor
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        match self.num_spatial_dims {
            2 => self.forward_2d(x),
            3 => self.forward_3d(x),
            _ => unreachable!(),
        }
    }

    /// 2D pixel shuffle: `(B, C*r^2, H, W)` → `(B, C, H*r, W*r)`.
    fn forward_2d(&self, x: &Tensor) -> Tensor {
        let r = self.upscale_factor;
        let (b, c_mul_r2, h, w) = x.size4().expect("forward_2d: tensor must be 4D");
        let c = c_mul_r2 / (r * r);
        assert_eq!(
            c_mul_r2,
            c * r * r,
            "input channels ({c_mul_r2}) must be divisible by upscale_factor^2 ({})",
            r * r
        );

        // (B, C, r, r, H, W) → (B, C, H, r, W, r) → (B, C, H*r, W*r)
        x.reshape([b, c, r, r, h, w])
            .permute([0, 1, 4, 2, 5, 3])
            .reshape([b, c, h * r, w * r])
    }

    /// 3D pixel shuffle: `(B, C*r^2, T, H, W)` → `(B, C, T, H*r, W*r)`.
    ///
    /// Only upsamples the spatial (H, W) dims; the time axis is preserved.
    fn forward_3d(&self, x: &Tensor) -> Tensor {
        let r = self.upscale_factor;
        let (b, c_mul_r2, t, h, w) = x.size5().expect("forward_3d: tensor must be 5D");
        let c = c_mul_r2 / (r * r);
        assert_eq!(
            c_mul_r2,
            c * r * r,
            "input channels ({c_mul_r2}) must be divisible by upscale_factor^2 ({})",
            r * r
        );

        // (B, C, r, r, T, H, W) → (B, C, T, H, r, W, r) → (B, C, T, H*r, W*r)
        x.reshape([b, c, r, r, t, h, w])
            .permute([0, 1, 4, 2, 5, 3, 6])
            .reshape([b, c, t, h * r, w * r])
    }
}

impl tch::nn::ModuleT for PixelShuffleND {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        self.forward(xs)
    }
}
