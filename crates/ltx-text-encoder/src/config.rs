use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Gemma3ConfigData {
    pub hidden_size: i64,
    pub intermediate_size: i64,
    pub num_attention_heads: i64,
    pub num_key_value_heads: i64,
    pub head_dim: i64,
    pub num_hidden_layers: i64,
    pub vocab_size: i64,
    pub rms_norm_eps: f64,
    pub hidden_act: String,
    pub rope_theta: f64,
    pub max_position_embeddings: i64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SigLIPConfigData {
    pub hidden_size: i64,
    pub intermediate_size: i64,
    pub num_attention_heads: i64,
    pub num_hidden_layers: i64,
    pub image_size: i64,
    pub patch_size: i64,
    pub hidden_act: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LTXVTextEncoderConfig {
    pub gemma3: Gemma3ConfigData,
    pub siglip: SigLIPConfigData,
    pub max_text_length: i64,
}
