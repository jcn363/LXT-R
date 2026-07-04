//! Test utilities for the LTX-2.3 Rust rewrite.
//!
//! Provides golden file loading (safetensors), tensor comparison assertions,
//! and common test fixtures (VarStore creation, deterministic tensors).

pub mod assertions;
pub mod fixtures;
pub mod golden;

pub use assertions::{assert_allclose, assert_allclose_default};
pub use fixtures::{make_seed_tensor, make_vs};
pub use golden::load_golden;
