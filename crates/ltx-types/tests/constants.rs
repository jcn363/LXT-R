use ltx_types::*;

#[test]
fn test_norm_eps() {
    assert_eq!(NORM_EPS, 1e-6);
}

#[test]
fn test_stability_eps() {
    assert_eq!(STABILITY_EPS, 1e-8);
}

#[test]
fn test_fp8_max() {
    assert_eq!(FP8_MAX, 448.0);
}

#[test]
fn test_fp8_min() {
    assert_eq!(FP8_MIN, -448.0);
}

#[test]
fn test_rope_theta() {
    assert_eq!(ROPE_THETA, 10_000.0);
}

#[test]
fn test_rope_freq_scale() {
    assert_eq!(ROPE_FREQ_SCALE, std::f64::consts::FRAC_PI_2);
}

#[test]
fn test_lrelu_slope() {
    assert_eq!(LRELU_SLOPE, 0.1);
}

#[test]
fn test_default_max_shift() {
    assert_eq!(DEFAULT_MAX_SHIFT, 2.05);
}

#[test]
fn test_default_base_shift() {
    assert_eq!(DEFAULT_BASE_SHIFT, 0.95);
}

#[test]
fn test_default_terminal() {
    assert_eq!(DEFAULT_TERMINAL, 0.1);
}

#[test]
fn test_timestep_scale_multiplier() {
    assert_eq!(TIMESTEP_SCALE_MULTIPLIER, 1000);
}

#[test]
fn test_default_max_pos() {
    assert_eq!(DEFAULT_MAX_POS, [20, 2048, 2048]);
}

#[test]
fn test_default_audio_max_pos() {
    assert_eq!(DEFAULT_AUDIO_MAX_POS, [20]);
}

#[test]
fn test_min_spatial_overlap_px() {
    assert_eq!(MIN_SPATIAL_OVERLAP_PX, 64);
}

#[test]
fn test_min_temporal_overlap_frames() {
    assert_eq!(MIN_TEMPORAL_OVERLAP_FRAMES, 16);
}

#[test]
fn test_default_tile_size_px() {
    assert_eq!(DEFAULT_TILE_SIZE_PX, 512);
}

#[test]
fn test_default_tile_overlap_px() {
    assert_eq!(DEFAULT_TILE_OVERLAP_PX, 64);
}

#[test]
fn test_default_tile_size_frames() {
    assert_eq!(DEFAULT_TILE_SIZE_FRAMES, 64);
}

#[test]
fn test_default_tile_overlap_frames() {
    assert_eq!(DEFAULT_TILE_OVERLAP_FRAMES, 24);
}

#[test]
fn test_default_time_scale() {
    assert_eq!(DEFAULT_TIME_SCALE, 8);
}

#[test]
fn test_default_height_scale() {
    assert_eq!(DEFAULT_HEIGHT_SCALE, 32);
}

#[test]
fn test_default_width_scale() {
    assert_eq!(DEFAULT_WIDTH_SCALE, 32);
}

#[test]
fn test_vae_norm_num_groups() {
    assert_eq!(VAE_NORM_NUM_GROUPS, 32);
}

#[test]
fn test_lora_deltas_dtype_if_fp8() {
    assert_eq!(LORA_DELTAS_DTYPE_IF_FP8, "bfloat16");
}

#[test]
fn test_attention_gate_scale() {
    assert_eq!(ATTENTION_GATE_SCALE, 2.0);
}

#[test]
fn test_projection_eps() {
    assert_eq!(PROJECTION_EPS, 1e-8);
}