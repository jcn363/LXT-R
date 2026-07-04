use ltx_types::NormLayerType;
use ltx_video_vae::configurator::VideoVAEConfig;

#[test]
fn test_vae_config_deserialize_minimal() {
    let json = r#"{
        "in_channels": 3,
        "base_channels": 128,
        "channel_multipliers": [1, 2, 4],
        "num_res_blocks": 2,
        "latent_channels": 16
    }"#;
    let cfg: VideoVAEConfig = serde_json::from_str(json).unwrap();
    assert_eq!(cfg.in_channels, 3);
    assert_eq!(cfg.base_channels, 128);
    assert_eq!(cfg.channel_multipliers, vec![1, 2, 4]);
    assert_eq!(cfg.num_res_blocks, 2);
    assert_eq!(cfg.latent_channels, 16);
    // Default values
    assert!(!cfg.causal);
    assert_eq!(cfg.norm_type, NormLayerType::Group);
    assert_eq!(cfg.spatial_downsample_factor, 8);
}

#[test]
fn test_vae_config_deserialize_full() {
    let json = r#"{
        "in_channels": 3,
        "base_channels": 64,
        "channel_multipliers": [1, 2],
        "num_res_blocks": 1,
        "latent_channels": 8,
        "norm_num_groups": 32,
        "causal": true,
        "norm_type": "Pixel",
        "spatial_downsample_factor": 4
    }"#;
    let cfg: VideoVAEConfig = serde_json::from_str(json).unwrap();
    assert!(cfg.causal);
    assert_eq!(cfg.norm_num_groups, 32);
    assert_eq!(cfg.spatial_downsample_factor, 4);
}

#[test]
fn test_vae_config_clone() {
    let json = r#"{
        "in_channels": 3,
        "base_channels": 128,
        "channel_multipliers": [1, 2, 4],
        "num_res_blocks": 2,
        "latent_channels": 16
    }"#;
    let cfg: VideoVAEConfig = serde_json::from_str(json).unwrap();
    let cfg2 = cfg.clone();
    assert_eq!(cfg.in_channels, cfg2.in_channels);
    assert_eq!(cfg.channel_multipliers, cfg2.channel_multipliers);
}

#[test]
fn test_vae_config_debug() {
    let json = r#"{
        "in_channels": 3,
        "base_channels": 64,
        "channel_multipliers": [1, 2],
        "num_res_blocks": 1,
        "latent_channels": 8
    }"#;
    let cfg: VideoVAEConfig = serde_json::from_str(json).unwrap();
    let debug_str = format!("{:?}", cfg);
    assert!(debug_str.contains("VideoVAEConfig"));
    assert!(debug_str.contains("in_channels: 3"));
}
