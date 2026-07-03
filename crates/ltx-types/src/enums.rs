use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum NormLayerType {
    Group,
    Pixel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum PaddingModeType {
    Zeros,
    Reflect,
    Replicate,
    Circular,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum ActivationType {
    SiLU,
    LeakyReLU,
    GELU,
    ReLU,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum ConvDimension {
    Conv1d,
    Conv2d,
    Conv3d,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum AttentionBackend {
    Sdpa,
    FlashAttention2,
    XFormers,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum DType {
    Float32,
    Float16,
    BFloat16,
    Float8E4m3fn,
}

impl DType {
    pub fn to_tch_kind(self) -> tch::Kind {
        match self {
            DType::Float32 => tch::Kind::Float,
            DType::Float16 => tch::Kind::Half,
            DType::BFloat16 => tch::Kind::BFloat16,
            DType::Float8E4m3fn => tch::Kind::BFloat16, // FP8 handled at quantization layer
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "float32" | "f32" => Some(DType::Float32),
            "float16" | "f16" => Some(DType::Float16),
            "bfloat16" | "bf16" => Some(DType::BFloat16),
            "float8_e4m3fn" | "fp8" => Some(DType::Float8E4m3fn),
            _ => None,
        }
    }
}
