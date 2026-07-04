//! Transformer model for the LTX-2.3 Rust rewrite.
//!
//! Provides the DiT (Diffusion Transformer) architecture including
//! BasicAVTransformerBlock, LTXModel, FeedForward, and text projection.
pub mod block;
pub mod configurator;
pub mod feed_forward;
pub mod model;
pub mod text_projection;

pub use configurator::from_config;
pub use model::LTXModel;
