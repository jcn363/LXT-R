//! Shared types, constants, and protocols for the LTX-2.3 Rust rewrite.
//!
//! This crate is the foundation of the workspace — every other crate depends on it.
//! It defines constants, shapes, enums, traits (protocols), and utility functions
//! that are used throughout the codebase.
//!
//! # SSOT Enforcement
//! All numeric constants live in [`constants`]. No other crate defines its own constants.

pub mod constants;
pub mod enums;
pub mod modality;
pub mod protocols;
pub mod shapes;
pub mod tools;
pub mod utils;

pub use constants::*;
pub use enums::*;
pub use modality::*;
pub use protocols::*;
pub use shapes::*;
pub use tools::*;
