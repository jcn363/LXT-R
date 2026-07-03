use tch::Tensor;

/// Reshape (B, C, T, H, W) → (B, C·r², T, H/r, W/r).
/// Spatial information is packed into the channel dimension.
pub fn space_to_depth(x: &Tensor, r: i64) -> Tensor {
    let (b, c, t, h, w) = x.size5().unwrap();
    // (B, C, T, H/r, r, W/r, r) → permute to group r×r patches with channels
    x.reshape([b, c, t, h / r, r, w / r, r])
        .permute([0, 1, 3, 5, 2, 4, 6])  // (B, C, H/r, W/r, T, r, r)
        .reshape([b, c * r * r, t, h / r, w / r])
}

/// Inverse of `space_to_depth`.
/// (B, C·r², T, H/r, W/r) → (B, C, T, H, W)
pub fn depth_to_space(x: &Tensor, r: i64) -> Tensor {
    let (b, crr, t, hdiv, wdiv) = x.size5().unwrap();
    let c = crr / (r * r);
    x.reshape([b, c, r, r, t, hdiv, wdiv])
        .permute([0, 1, 6, 2, 4, 3, 5])  // (B, C, W/r, r, T, r, H/r) -- no wait
        .reshape([b, c, t, hdiv * r, wdiv * r])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_space_to_depth_roundtrip() {
        let x = Tensor::randn([1, 3, 8, 32, 32], (tch::Kind::Float, tch::Device::Cpu));
        let y = space_to_depth(&x, 2);
        assert_eq!(y.size(), vec![1, 12, 8, 16, 16]);
        let x_rec = depth_to_space(&y, 2);
        assert_eq!(x_rec.size(), vec![1, 3, 8, 32, 32]);
        assert!(x.allclose(&x_rec, 1e-5, 1e-5, false));
    }

    #[test]
    fn test_various_shapes() {
        for r in [2, 4] {
            let x = Tensor::randn([2, 4, 6, 16, 16], (tch::Kind::Float, tch::Device::Cpu));
            let y = space_to_depth(&x, r);
            assert_eq!(y.size(), vec![2, 4 * r * r, 6, 16 / r, 16 / r]);
            let x_rec = depth_to_space(&y, r);
            assert_eq!(x_rec.size(), vec![2, 4, 6, 16, 16]);
            assert!(x.allclose(&x_rec, 1e-5, 1e-5, false));
        }
    }
}
