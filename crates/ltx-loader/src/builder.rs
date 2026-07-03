use std::path::Path;

use crate::primitives::{StateDict, StateDictLoader};
use crate::safetensors_loader::SafetensorsStateDictLoader;
use crate::sd_ops::SDOps;

/// Builder for constructing a single-GPU model from a checkpoint.
pub struct SingleGPUModelBuilder {
    device: tch::Device,
    dtype: tch::Kind,
}

impl SingleGPUModelBuilder {
    pub fn new(device: tch::Device, dtype: tch::Kind) -> Self {
        Self { device, dtype }
    }

    /// Builder targeting CPU with float32.
    pub fn cpu() -> Self {
        Self::new(tch::Device::Cpu, tch::Kind::Float)
    }

    /// Builder targeting CUDA device 0 with bfloat16.
    pub fn cuda() -> Self {
        let device = tch::Device::Cuda(0);
        Self::new(device, tch::Kind::BFloat16)
    }

    /// Load a state dict from a file path, auto-detecting format.
    pub fn load_state_dict(&self, path: &Path) -> Result<StateDict, Box<dyn std::error::Error>> {
        let ext = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        let state_dict = if ext == "safetensors" {
            let loader = SafetensorsStateDictLoader::new();
            loader.load(path)?
        } else {
            return Err(format!("Unsupported checkpoint format: .{}", ext).into());
        };

        Ok(self.prepare_state_dict(state_dict))
    }

    /// Cast and move state dict tensors to the target device and dtype.
    fn prepare_state_dict(&self, state_dict: StateDict) -> StateDict {
        let casted = SDOps::cast_dtype(&state_dict, self.dtype);
        casted
            .iter()
            .map(|(k, v)| (k.clone(), v.to_device(self.device)))
            .collect()
    }

    pub fn device(&self) -> tch::Device {
        self.device
    }

    pub fn dtype(&self) -> tch::Kind {
        self.dtype
    }
}

impl Default for SingleGPUModelBuilder {
    fn default() -> Self {
        Self::cpu()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Tensor;

    #[test]
    fn test_builder_cpu() {
        let builder = SingleGPUModelBuilder::cpu();
        assert_eq!(builder.device(), tch::Device::Cpu);
        assert_eq!(builder.dtype(), tch::Kind::Float);
    }

    #[test]
    fn test_prepare_state_dict() {
        let builder = SingleGPUModelBuilder::new(tch::Device::Cpu, tch::Kind::BFloat16);
        let mut sd = std::collections::HashMap::new();
        sd.insert("w".to_string(), Tensor::ones([2, 2], (tch::Kind::Float, tch::Device::Cpu)));

        let prepared = builder.prepare_state_dict(sd);
        let w = prepared.get("w").unwrap();
        assert_eq!(w.kind(), tch::Kind::BFloat16);
    }
}
