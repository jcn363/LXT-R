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

#[derive(Debug, Clone, Deserialize)]
pub struct T5ConfigData {
    pub d_model: i64,
    pub d_ff: i64,
    pub d_kv: i64,
    pub num_heads: i64,
    pub num_layers: i64,
    pub vocab_size: i64,
    pub layer_norm_epsilon: f64,
    #[serde(default)]
    pub dropout_rate: f64,
    #[serde(default = "default_num_buckets")]
    pub relative_attention_num_buckets: i64,
    #[serde(default = "default_max_distance")]
    pub relative_attention_max_distance: i64,
    #[serde(default)]
    pub is_gated_act: bool,
    #[serde(default)]
    pub dense_act_fn: String,
}

fn default_num_buckets() -> i64 { 32 }
fn default_max_distance() -> i64 { 128 }
