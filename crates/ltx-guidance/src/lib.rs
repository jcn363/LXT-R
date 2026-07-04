//! Guidance configuration for the LTX-2.3 Rust rewrite.
//!
//! Provides perturbation configs for STG (Spatiotemporal Guidance)
//! and other guidance methods.

pub mod perturbations;

pub use perturbations::{PerturbationConfig, PerturbationKind, StgPerturbationConfig};
