use ltx_types::{NORM_EPS, ROPE_THETA};

use crate::config::{LTXVTextEncoderConfig, T5ConfigData};
use crate::encoder::GemmaTextEncoder;
use crate::tokenizer::LTXVGemmaTokenizer;

/// Build a GemmaTextEncoder from configuration data.
pub fn from_config(
    vs: &tch::nn::Path,
    config: &LTXVTextEncoderConfig,
    tokenizer_path: &str,
) -> Result<GemmaTextEncoder, Box<dyn std::error::Error + Send + Sync>> {
    let tokenizer = LTXVGemmaTokenizer::from_file(tokenizer_path, config.max_text_length as usize)?;
    Ok(GemmaTextEncoder::new(vs, config, tokenizer))
}

/// Create a default T5 configuration matching T5-XXL.
pub fn default_t5_config() -> T5ConfigData {
    T5ConfigData {
        d_model: 4096,
        d_ff: 10240,
        d_kv: 64,
        num_heads: 64,
        num_layers: 24,
        vocab_size: 32128,
        layer_norm_epsilon: NORM_EPS,
        dropout_rate: 0.1,
        relative_attention_num_buckets: 32,
        relative_attention_max_distance: 128,
        is_gated_act: true,
        dense_act_fn: String::from("gelu_new"),
    }
}

/// Create a default configuration matching the LTX-V Gemma3-12B + SigLIP-L setup.
pub fn default_config() -> LTXVTextEncoderConfig {
    LTXVTextEncoderConfig {
        gemma3: crate::config::Gemma3ConfigData {
            hidden_size: 3840,
            intermediate_size: 14336,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 256,
            num_hidden_layers: 48,
            vocab_size: 262144,
            rms_norm_eps: NORM_EPS,
            hidden_act: String::from("gelu_pytorch_tanh"),
            rope_theta: ROPE_THETA,
            max_position_embeddings: 131072,
        },
        siglip: crate::config::SigLIPConfigData {
            hidden_size: 1024,
            intermediate_size: 4096,
            num_attention_heads: 16,
            num_hidden_layers: 27,
            image_size: 384,
            patch_size: 14,
            hidden_act: String::from("gelu_pytorch_tanh"),
        },
        max_text_length: 512,
    }
}
