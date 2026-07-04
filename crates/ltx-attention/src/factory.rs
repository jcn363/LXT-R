use tch::nn::ModuleT;

use crate::rope::RopeType;
use crate::simple_attn::SimpleAttnBlock;
use crate::transformer_attn::TransformerAttention;

pub fn make_attention(
    attn_type: &str,
    dim: i64,
    heads: i64,
    head_dim: i64,
    context_dim: Option<i64>,
    rope_type: RopeType,
) -> Result<Box<dyn ModuleT + Send>, String> {
    match attn_type {
        "transformer" => Ok(Box::new(TransformerAttention::new(
            dim, heads, head_dim, context_dim, rope_type,
        ))),
        "simple" => Ok(Box::new(SimpleAttnBlock::new(dim))),
        "gated" => Ok(Box::new(TransformerAttention::new_gated(
            dim, heads, head_dim, context_dim, rope_type,
        ))),
        _ => Err(format!("Unknown attention type: {}", attn_type)),
    }
}
