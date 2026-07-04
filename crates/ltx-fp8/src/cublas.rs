use std::ffi::c_void;
use tch::Tensor;

/// Handle to a cuBLAS library instance for FP8 GEMM operations.
///
/// THE ONLY cuBLAS FP8 GEMM wrapper in the LTX codebase.
/// CUDA FFI is isolated here — no other crate calls cuBLAS directly.
pub struct CublasFp8Handle {
    #[allow(dead_code)]
    handle: *mut c_void,
}

unsafe impl Send for CublasFp8Handle {}
unsafe impl Sync for CublasFp8Handle {}

impl CublasFp8Handle {
    /// Create a new cuBLAS handle.
    ///
    /// Returns `None` if CUDA is not available.
    pub fn new() -> Option<Self> {
        // CUDA FFI not yet wired — returns None on all platforms.
        // To enable: link against libcublas and call cublasCreate via extern "C".
        None
    }

    /// Perform FP8 GEMM: `out = a @ b` with per-tensor scaling.
    ///
    /// `scale_a` and `scale_b` are the FP8 dequantization scales for the respective operands.
    /// `out` specifies the output dtype.
    ///
    /// THE ONLY FP8 GEMM implementation in the LTX codebase.
    pub fn gemm_fp8(
        &self,
        _a: &Tensor,
        _b: &Tensor,
        _scale_a: f32,
        _scale_b: f32,
        _out: tch::Kind,
    ) -> Tensor {
        unimplemented!("cuBLAS FP8 GEMM — requires CUDA toolchain")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cublas_handle_not_available_on_cpu() {
        assert!(CublasFp8Handle::new().is_none());
    }

    #[test]
    #[should_panic(expected = "cuBLAS FP8 GEMM")]
    fn test_gemm_fp8_unimplemented() {
        let handle = CublasFp8Handle {
            handle: std::ptr::null_mut(),
        };
        let a = Tensor::zeros([2, 3], (tch::Kind::Float, tch::Device::Cpu));
        let b = Tensor::zeros([3, 4], (tch::Kind::Float, tch::Device::Cpu));
        let _ = handle.gemm_fp8(&a, &b, 1.0, 1.0, tch::Kind::Float);
    }
}
