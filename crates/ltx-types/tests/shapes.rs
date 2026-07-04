use ltx_types::*;

#[test]
fn test_video_latent_shape_new() {
    let shape = VideoLatentShape::new(2, 16, 8, 32, 32);
    assert_eq!(shape.batch, 2);
    assert_eq!(shape.channels, 16);
    assert_eq!(shape.frames, 8);
    assert_eq!(shape.height, 32);
    assert_eq!(shape.width, 32);
}

#[test]
fn test_video_latent_shape_spatial_dim() {
    let shape = VideoLatentShape::new(1, 16, 8, 32, 32);
    assert_eq!(shape.spatial_dim(), 32 * 32);
}

#[test]
fn test_video_latent_shape_temporal_dim() {
    let shape = VideoLatentShape::new(1, 16, 8, 32, 32);
    assert_eq!(shape.temporal_dim(), 8);
}

#[test]
fn test_video_latent_shape_flatten_spatial() {
    let shape = VideoLatentShape::new(1, 16, 8, 32, 32);
    assert_eq!(shape.flatten_spatial(), 8 * 32 * 32);
}

#[test]
fn test_video_latent_shape_to_vec() {
    let shape = VideoLatentShape::new(2, 16, 8, 32, 32);
    assert_eq!(shape.to_vec(), vec![2, 16, 8, 32, 32]);
}

#[test]
fn test_video_latent_shape_clone_copy() {
    let a = VideoLatentShape::new(1, 16, 8, 32, 32);
    let b = a;
    let c = a;
    assert_eq!(a, b);
    assert_eq!(a, c);
}

#[test]
fn test_video_latent_shape_partial_eq() {
    let a = VideoLatentShape::new(1, 16, 8, 32, 32);
    let b = VideoLatentShape::new(1, 16, 8, 32, 32);
    let c = VideoLatentShape::new(2, 16, 8, 32, 32);
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn test_audio_latent_shape_new() {
    let shape = AudioLatentShape::new(2, 8, 128, 64);
    assert_eq!(shape.batch, 2);
    assert_eq!(shape.channels, 8);
    assert_eq!(shape.time, 128);
    assert_eq!(shape.features, 64);
}

#[test]
fn test_audio_latent_shape_to_vec() {
    let shape = AudioLatentShape::new(2, 8, 128, 64);
    assert_eq!(shape.to_vec(), vec![2, 8, 128, 64]);
}

#[test]
fn test_audio_latent_shape_clone_copy() {
    let a = AudioLatentShape::new(1, 8, 128, 64);
    let b = a;
    let c = a;
    assert_eq!(a, b);
    assert_eq!(a, c);
}

#[test]
fn test_audio_latent_shape_partial_eq() {
    let a = AudioLatentShape::new(1, 8, 128, 64);
    let b = AudioLatentShape::new(1, 8, 128, 64);
    let c = AudioLatentShape::new(2, 8, 128, 64);
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn test_patch_grid_bounds_new() {
    let bounds = PatchGridBounds::new(0, 8, 0, 32, 0, 32);
    assert_eq!(bounds.min_t, 0);
    assert_eq!(bounds.max_t, 8);
    assert_eq!(bounds.min_h, 0);
    assert_eq!(bounds.max_h, 32);
    assert_eq!(bounds.min_w, 0);
    assert_eq!(bounds.max_w, 32);
}

#[test]
fn test_patch_grid_bounds_clone_copy() {
    let a = PatchGridBounds::new(0, 8, 0, 32, 0, 32);
    let b = a;
    let c = a;
    assert_eq!(a, b);
    assert_eq!(a, c);
}

#[test]
fn test_patch_grid_bounds_partial_eq() {
    let a = PatchGridBounds::new(0, 8, 0, 32, 0, 32);
    let b = PatchGridBounds::new(0, 8, 0, 32, 0, 32);
    let c = PatchGridBounds::new(1, 8, 0, 32, 0, 32);
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn test_tiling_config_default() {
    let config = TilingConfig::default();
    assert_eq!(config.tile_size_px, DEFAULT_TILE_SIZE_PX);
    assert_eq!(config.tile_overlap_px, DEFAULT_TILE_OVERLAP_PX);
    assert_eq!(config.tile_size_frames, DEFAULT_TILE_SIZE_FRAMES);
    assert_eq!(config.tile_overlap_frames, DEFAULT_TILE_OVERLAP_FRAMES);
}

#[test]
fn test_tiling_config_custom() {
    let config = TilingConfig {
        tile_size_px: 256,
        tile_overlap_px: 32,
        tile_size_frames: 32,
        tile_overlap_frames: 16,
    };
    assert_eq!(config.tile_size_px, 256);
    assert_eq!(config.tile_overlap_px, 32);
    assert_eq!(config.tile_size_frames, 32);
    assert_eq!(config.tile_overlap_frames, 16);
}

#[test]
fn test_tiling_config_clone() {
    let a = TilingConfig::default();
    let b = a.clone();
    assert_eq!(a, b);
}

#[test]
fn test_tiling_config_partial_eq() {
    let a = TilingConfig::default();
    let b = TilingConfig::default();
    let c = TilingConfig {
        tile_size_px: 256,
        tile_overlap_px: 64,
        tile_size_frames: 64,
        tile_overlap_frames: 24,
    };
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn test_transformer_args_default() {
    let args = TransformerArgs::default();
    assert_eq!(args.num_layers, 28);
    assert_eq!(args.num_heads, 8);
    assert_eq!(args.head_dim, 128);
    assert_eq!(args.hidden_dim, 1024);
    assert_eq!(args.intermediate_dim, 4096);
    assert_eq!(args.context_dim, None);
    assert!(args.use_rope);
    assert_eq!(args.rope_type, "interleaved");
    assert_eq!(args.max_seq_len, 2048);
}

#[test]
fn test_transformer_args_custom() {
    let args = TransformerArgs {
        num_layers: 12,
        num_heads: 16,
        head_dim: 64,
        hidden_dim: 512,
        intermediate_dim: 2048,
        context_dim: Some(768),
        use_rope: false,
        rope_type: "split".to_string(),
        max_seq_len: 1024,
    };
    assert_eq!(args.num_layers, 12);
    assert_eq!(args.num_heads, 16);
    assert_eq!(args.head_dim, 64);
    assert_eq!(args.hidden_dim, 512);
    assert_eq!(args.intermediate_dim, 2048);
    assert_eq!(args.context_dim, Some(768));
    assert!(!args.use_rope);
    assert_eq!(args.rope_type, "split");
    assert_eq!(args.max_seq_len, 1024);
}

#[test]
fn test_transformer_args_clone() {
    let a = TransformerArgs::default();
    let b = a.clone();
    assert_eq!(a, b);
}

#[test]
fn test_transformer_args_partial_eq() {
    let a = TransformerArgs::default();
    let b = TransformerArgs::default();
    let c = TransformerArgs {
        num_layers: 12,
        ..Default::default()
    };
    assert_eq!(a, b);
    assert_ne!(a, c);
}
