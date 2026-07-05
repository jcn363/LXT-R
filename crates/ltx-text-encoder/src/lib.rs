//! Text encoder for the LTX-2.3 Rust rewrite.
//!
//! Provides GemmaTextEncoder (48-layer transformer) and SigLIPVisionTower (27-layer)
//! for encoding text and image inputs into embeddings.

pub mod config;
pub mod configurator;
pub mod embeddings_connector;
pub mod embeddings_processor;
pub mod encoder;
pub mod feature_extractor;
pub mod gemma3_text;
pub mod image_processor;
pub mod prompt_enhancement;
pub mod siglip;
pub mod tokenizer;

pub mod t5_encoder;

pub use config::{Gemma3ConfigData, LTXVTextEncoderConfig, SigLIPConfigData, T5ConfigData};
pub use configurator::{default_config, from_config};
pub use encoder::{GemmaTextEncoder, T5TextEncoder};
pub use tokenizer::LTXVGemmaTokenizer;
