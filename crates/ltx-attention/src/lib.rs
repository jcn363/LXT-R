//! Attention mechanisms for the LTX-2.3 Rust rewrite.
//!
//! Provides RoPE (Rotary Position Embeddings), SDPA (Scaled Dot-Product Attention),
//! TransformerAttention (Q/K/V with gating), SimpleAttnBlock (Conv2d-based), and a factory.

pub mod factory;
pub mod rope;
pub mod sdpa;
pub mod simple_attn;
pub mod transformer_attn;

pub use factory::make_attention;
pub use rope::{apply_rotary_emb, precompute_freqs_cis, RopeType};
pub use sdpa::scaled_dot_product_attention;
pub use simple_attn::SimpleAttnBlock;
pub use transformer_attn::TransformerAttention;
