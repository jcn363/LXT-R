//! Conditioning utilities for the LTX-2.3 Rust rewrite.
//!
//! Provides attention masks (causal, padding, cross-attention),
//! conditioning types (LatentCond, ReferenceVideo, Keyframe),
//! and the ConditioningItem trait.

pub mod item;
pub mod mask_utils;
pub mod types;

pub use item::ConditioningItem;
pub use mask_utils::{causal_mask, causal_padding_mask, cross_attention_mask, padding_mask};
pub use types::{Keyframe, LatentCond, ReferenceVideo};
