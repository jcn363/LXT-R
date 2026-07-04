use ltx_text_encoder::Gemma3ConfigData;

#[test]
fn test_gemma3_config_deserialize() {
    let json = r#"{
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "num_hidden_layers": 2,
        "vocab_size": 1000,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "rope_theta": 10000.0,
        "max_position_embeddings": 512
    }"#;
    let config: Gemma3ConfigData = serde_json::from_str(json).unwrap();
    assert_eq!(config.hidden_size, 64);
    assert_eq!(config.intermediate_size, 128);
    assert_eq!(config.num_attention_heads, 4);
    assert_eq!(config.num_key_value_heads, 2);
    assert_eq!(config.head_dim, 16);
    assert_eq!(config.num_hidden_layers, 2);
    assert_eq!(config.vocab_size, 1000);
    assert!((config.rms_norm_eps - 1e-6).abs() < 1e-9);
    assert_eq!(config.hidden_act, "silu");
    assert!((config.rope_theta - 10000.0).abs() < 0.01);
    assert_eq!(config.max_position_embeddings, 512);
}

#[test]
fn test_gemma3_config_clone() {
    let json = r#"{
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "num_hidden_layers": 1,
        "vocab_size": 500,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "rope_theta": 10000.0,
        "max_position_embeddings": 256
    }"#;
    let config: Gemma3ConfigData = serde_json::from_str(json).unwrap();
    let config2 = config.clone();
    assert_eq!(config.hidden_size, config2.hidden_size);
    assert_eq!(config.vocab_size, config2.vocab_size);
}

#[test]
fn test_gemma3_config_debug() {
    let json = r#"{
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "num_hidden_layers": 1,
        "vocab_size": 500,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "rope_theta": 10000.0,
        "max_position_embeddings": 256
    }"#;
    let config: Gemma3ConfigData = serde_json::from_str(json).unwrap();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("Gemma3ConfigData"));
    assert!(debug_str.contains("hidden_size: 32"));
}
