/// Normalization epsilon for RMSNorm, GroupNorm, and all norm layers.
pub const NORM_EPS: f64 = 1e-6;

/// Small epsilon for numerical stability (clamp_min, division safety).
pub const STABILITY_EPS: f64 = 1e-8;

/// FP8 E4M3FN maximum representable value.
pub const FP8_MAX: f64 = 448.0;

/// FP8 E4M3FN minimum representable value.
pub const FP8_MIN: f64 = -448.0;

/// Default RoPE theta (base frequency).
pub const ROPE_THETA: f64 = 10_000.0;

/// RoPE frequency scaling factor (pi / 2).
pub const ROPE_FREQ_SCALE: f64 = std::f64::consts::FRAC_PI_2;

/// LeakyReLU slope used in audio VAE ResBlocks.
pub const LRELU_SLOPE: f64 = 0.1;

/// Default scheduler parameters.
pub const DEFAULT_MAX_SHIFT: f64 = 2.05;
pub const DEFAULT_BASE_SHIFT: f64 = 0.95;
pub const DEFAULT_TERMINAL: f64 = 0.1;

/// Default timestep scale multiplier.
pub const TIMESTEP_SCALE_MULTIPLIER: i64 = 1000;

/// Default positional embedding max positions (time, height, width).
pub const DEFAULT_MAX_POS: [i64; 3] = [20, 2048, 2048];
pub const DEFAULT_AUDIO_MAX_POS: [i64; 1] = [20];

/// Tiling minimums.
pub const MIN_SPATIAL_OVERLAP_PX: i64 = 64;
pub const MIN_TEMPORAL_OVERLAP_FRAMES: i64 = 16;

/// Tiling defaults.
pub const DEFAULT_TILE_SIZE_PX: i64 = 512;
pub const DEFAULT_TILE_OVERLAP_PX: i64 = 64;
pub const DEFAULT_TILE_SIZE_FRAMES: i64 = 64;
pub const DEFAULT_TILE_OVERLAP_FRAMES: i64 = 24;

/// Scale factors for latent ↔ pixel conversion.
pub const DEFAULT_TIME_SCALE: i64 = 8;
pub const DEFAULT_HEIGHT_SCALE: i64 = 32;
pub const DEFAULT_WIDTH_SCALE: i64 = 32;

/// Video VAE normalization groups.
pub const VAE_NORM_NUM_GROUPS: i64 = 32;

/// LoRA delta dtype when model is FP8 — stored as string, resolved at runtime.
pub const LORA_DELTAS_DTYPE_IF_FP8: &str = "bfloat16";

/// Attention gate multiplier.
pub const ATTENTION_GATE_SCALE: f64 = 2.0;

/// Projection coefficient epsilon (avoid division by zero).
pub const PROJECTION_EPS: f64 = 1e-8;
