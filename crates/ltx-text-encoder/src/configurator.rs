use ltx_types::{NORM_EPS, ROPE_THETA};

use crate::config::LTXVTextEncoderConfig;
use crate::encoder::GemmaTextEncoder;
use crate::tokenizer::LTXVGemmaTokenizer;

/// Build a GemmaTextEncoder from configuration data.
#[must_use = "caller must handle encoder build error"]
pub fn from_config(
    vs: &tch::nn::Path,
    config: &LTXVTextEncoderConfig,
    tokenizer_path: &str,
) -> Result<GemmaTextEncoder, Box<dyn std::error::Error + Send + Sync>> {
    let tokenizer = LTXVGemmaTokenizer::from_file(tokenizer_path, config.max_text_length as usize)?;
    Ok(GemmaTextEncoder::new(vs, config, tokenizer))
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
