use tch::Tensor;

/// Patchify 5D video tensor `(B,C,F,H,W)` → `(B, T, D)`.
///
/// Reshapes and permutes so that spatial-temporal patches are flattened into the
/// sequence dimension `T = (F/p1)*(H/p2)*(W/p3)` and the channel-plus-patch
/// elements form the feature dimension `D = C*p1*p2*p3`.
pub fn patchify_5d(x: &Tensor, p1: i64, p2: i64, p3: i64) -> Tensor {
    let (b, c, f, h, w) = x.size5().expect("patchify_5d: tensor must be 5D");
    assert!(f % p1 == 0, "f={f} not divisible by p1={p1}");
    assert!(h % p2 == 0, "h={h} not divisible by p2={p2}");
    assert!(w % p3 == 0, "w={w} not divisible by p3={p3}");
    x.reshape([b, c, f / p1, p1, h / p2, p2, w / p3, p3])
        .permute([0, 2, 4, 6, 1, 3, 5, 7])
        .reshape([b, (f / p1) * (h / p2) * (w / p3), c * p1 * p2 * p3])
}

/// Unpatchify 5D video tensor back to `(B, C, F*p1, H*p2, W*p3)`.
#[allow(clippy::too_many_arguments)]
pub fn unpatchify_5d(
    x: &Tensor,
    b: i64,
    c: i64,
    f: i64,
    h: i64,
    w: i64,
    p1: i64,
    p2: i64,
    p3: i64,
) -> Tensor {
    // x has shape [B, (f/p1)*(h/p2)*(w/p3), c*p1*p2*p3]
    // We need to reconstruct the grid: [B, f/p1, h/p2, w/p3, c, p1, p2, p3]
    let fp = f / p1;
    let hp = h / p2;
    let wp = w / p3;
    x.reshape([b, fp, hp, wp, c, p1, p2, p3])
        .permute([0, 4, 1, 5, 2, 6, 3, 7])
        .reshape([b, c, f, h, w])
}

/// Patchify 4D tensor `(B,C,H,W)` → `(B, C*p*p, H/p, W/p)`.
///
/// Each `p×p` spatial block is folded into the channel dimension, producing a
/// compact representation with reduced spatial extent.
pub fn patchify_4d(x: &Tensor, p: i64) -> Tensor {
    let (b, c, h, w) = x.size4().expect("patchify_4d: tensor must be 4D");
    assert!(h % p == 0, "h={h} not divisible by p={p}");
    assert!(w % p == 0, "w={w} not divisible by p={p}");
    x.reshape([b, c, h / p, p, w / p, p])
        .permute([0, 1, 3, 5, 2, 4])
        .reshape([b, c * p * p, h / p, w / p])
}

/// Unpatchify 4D tensor back to `(B, C, H, W)`.
pub fn unpatchify_4d(x: &Tensor, b: i64, c: i64, h: i64, w: i64, p: i64) -> Tensor {
    let (_b_actual, _, hp, wp) = x.size4().expect("unpatchify_4d: tensor must be 4D");
    x.reshape([b, c, p, p, hp, wp])
        .permute([0, 1, 4, 2, 5, 3])
        .reshape([b, c, h, w])
}

/// Patchify audio tensor `(B,C,T,F)` → `(B, T, C*F)`.
///
/// Each time step's channel × frequency features are concatenated into a single
/// vector per position.
pub fn patchify_audio(x: &Tensor) -> Tensor {
    let (b, c, t, f) = x.size4().expect("patchify_audio: tensor must be 4D");
    x.reshape([b, c, t, f])
        .permute([0, 2, 1, 3])
        .reshape([b, t, c * f])
}

/// Unpatchify audio tensor back to `(B, C, T, F)`.
pub fn unpatchify_audio(x: &Tensor, c: i64, f: i64) -> Tensor {
    let (b, t, _) = x.size3().expect("unpatchify_audio: tensor must be 3D");
    x.reshape([b, t, c, f]).permute([0, 2, 1, 3])
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn test_patchify_5d_roundtrip() {
        let x = Tensor::randn([1, 4, 8, 16, 16], (tch::Kind::Float, Device::Cpu));
        let p1 = 2i64;
        let p2 = 4i64;
        let p3 = 4i64;
        let patched = patchify_5d(&x, p1, p2, p3);
        // Output is [B, T, D] where T=(F/p1)*(H/p2)*(W/p3), D=C*p1*p2*p3
        assert_eq!(patched.size(), vec![1, 4 * 4 * 4, 4 * p1 * p2 * p3]);
        let unp = unpatchify_5d(&patched, 1, 4, 8, 16, 16, p1, p2, p3);
        assert_eq!(unp.size(), vec![1, 4, 8, 16, 16]);
        assert!(x.allclose(&unp, 1e-6, 1e-6, false));
    }

    #[test]
    fn test_patchify_4d_roundtrip() {
        let x = Tensor::randn([1, 3, 32, 32], (tch::Kind::Float, Device::Cpu));
        let p = 8i64;
        let patched = patchify_4d(&x, p);
        assert_eq!(patched.size(), vec![1, 3 * p * p, 32 / p, 32 / p]);
        let unp = unpatchify_4d(&patched, 1, 3, 32, 32, p);
        assert_eq!(unp.size(), vec![1, 3, 32, 32]);
        assert!(x.allclose(&unp, 1e-6, 1e-6, false));
    }

    #[test]
    fn test_patchify_audio_roundtrip() {
        let x = Tensor::randn([1, 64, 128, 128], (tch::Kind::Float, Device::Cpu));
        let patched = patchify_audio(&x);
        assert_eq!(patched.size(), vec![1, 128, 64 * 128]);
        let unp = unpatchify_audio(&patched, 64, 128);
        assert_eq!(unp.size(), vec![1, 64, 128, 128]);
        assert!(x.allclose(&unp, 1e-6, 1e-6, false));
    }

    // ── Golden tests (Python reference) ──────────────────────────────────

    /// Golden test: patchify_5d roundtrip matches Python reference.
    #[test]
    fn test_golden_patchify_5d() {
        let input = ltx_test_utils::load_golden("crates/goldens/patchify_5d.safetensors", "input");
        let expected_recovered =
            ltx_test_utils::load_golden("crates/goldens/patchify_5d.safetensors", "recovered");

        let (b, c, f, h, w) = (
            input.size()[0],
            input.size()[1],
            input.size()[2],
            input.size()[3],
            input.size()[4],
        );
        let (p1, p2, p3) = (2i64, 4i64, 4i64);
        let patched = patchify_5d(&input, p1, p2, p3);
        let recovered = unpatchify_5d(&patched, b, c, f, h, w, p1, p2, p3);

        ltx_test_utils::assert_allclose(&recovered, &expected_recovered, 1e-6, 1e-6);
    }

    /// Golden test: patchify_4d roundtrip matches Python reference.
    #[test]
    fn test_golden_patchify_4d() {
        let input = ltx_test_utils::load_golden("crates/goldens/patchify_4d.safetensors", "input");
        let expected_patched =
            ltx_test_utils::load_golden("crates/goldens/patchify_4d.safetensors", "patched");

        let p = 8i64;
        let patched = patchify_4d(&input, p);

        ltx_test_utils::assert_allclose(&patched, &expected_patched, 1e-6, 1e-6);
    }
}
