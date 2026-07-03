use serde::Deserialize;

use ltx_types::DType;

#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationPolicy {
    pub weight_dtype: DType,
    pub activate_fp8: bool,
    pub group_size: Option<i64>,
}

impl Default for QuantizationPolicy {
    fn default() -> Self {
        Self {
            weight_dtype: DType::BFloat16,
            activate_fp8: false,
            group_size: None,
        }
    }
}

impl QuantizationPolicy {
    pub fn fp8_per_tensor() -> Self {
        Self {
            weight_dtype: DType::Float8E4m3fn,
            activate_fp8: false,
            group_size: None,
        }
    }

    pub fn is_fp8(&self) -> bool {
        matches!(self.weight_dtype, DType::Float8E4m3fn)
    }
}
