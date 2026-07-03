//! CUDA kernel stubs.
//!
//! The actual CUDA C kernel implementations live in a separate `kernels.cu` file
//! and are compiled via `build.rs`. This module provides the Rust-side FFI declarations
//! and safe wrappers that will call into the compiled CUDA code.
//!
//! When CUDA is not available, these functions return errors or use CPU fallbacks.

/// Check if CUDA kernels are available at runtime.
pub fn cuda_kernels_available() -> bool {
    // CUDA kernels not yet compiled — always returns false.
    // To enable: build with `--features cuda` and link kernels.cu via build.rs.
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernels_not_available_on_cpu() {
        assert!(!cuda_kernels_available());
    }
}
