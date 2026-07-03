use std::collections::HashMap;
use std::path::Path;
use tch::Tensor;

use crate::primitives::{StateDict, StateDictLoader};

/// Loader for .safetensors format files.
pub struct SafetensorsStateDictLoader;

impl SafetensorsStateDictLoader {
    pub fn new() -> Self {
        Self
    }
}

impl Default for SafetensorsStateDictLoader {
    fn default() -> Self {
        Self
    }
}

fn safetensors_dtype_to_tch(dtype: safetensors::Dtype) -> tch::Kind {
    match dtype {
        safetensors::Dtype::F32 => tch::Kind::Float,
        safetensors::Dtype::F64 => tch::Kind::Double,
        safetensors::Dtype::F16 => tch::Kind::Half,
        safetensors::Dtype::BF16 => tch::Kind::BFloat16,
        safetensors::Dtype::I64 => tch::Kind::Int64,
        safetensors::Dtype::I32 => tch::Kind::Int,
        safetensors::Dtype::I16 => tch::Kind::Int16,
        safetensors::Dtype::I8 => tch::Kind::Int8,
        safetensors::Dtype::U8 => tch::Kind::Uint8,
        safetensors::Dtype::BOOL => tch::Kind::Bool,
        // FP8 and unsigned int types — store as BF16 (FP8 proxy) or closest tch kind
        safetensors::Dtype::F8_E4M3 | safetensors::Dtype::F8_E5M2 => tch::Kind::BFloat16,
        safetensors::Dtype::U16 => tch::Kind::Int16,
        safetensors::Dtype::U32 => tch::Kind::Int,
        safetensors::Dtype::U64 => tch::Kind::Int64,
        _ => tch::Kind::Float,
    }
}

impl StateDictLoader for SafetensorsStateDictLoader {
    fn load(&self, path: &Path) -> Result<StateDict, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;
        let safetensor = safetensors::SafeTensors::deserialize(&bytes)?;
        let mut state_dict = HashMap::new();

        for name in safetensor.names() {
            let tensor_view = safetensor.tensor(name)?;
            let kind = safetensors_dtype_to_tch(tensor_view.dtype());
            let shape: Vec<i64> = tensor_view.shape().iter().map(|&s| s as i64).collect();
            let data = tensor_view.data();
            let tensor = Tensor::from_data_size(data, &shape, kind);
            state_dict.insert(name.to_string(), tensor);
        }

        Ok(state_dict)
    }

    fn supports_extension(&self, ext: &str) -> bool {
        ext == "safetensors"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supports_extension() {
        let loader = SafetensorsStateDictLoader::new();
        assert!(loader.supports_extension("safetensors"));
        assert!(!loader.supports_extension("bin"));
    }
}
