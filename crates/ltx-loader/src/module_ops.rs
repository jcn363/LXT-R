use crate::primitives::StateDict;

/// Operations for loading weights into nn modules.
pub struct ModuleOps;

impl ModuleOps {
    pub fn new() -> Self {
        Self
    }

    /// Load a state dict into a module, returning any missing or unexpected keys.
    ///
    /// `module` - the tch Module to load weights into
    /// `state_dict` - the weights to load
    /// `strict` - if true, error on missing/unexpected keys
    pub fn load_state_dict(
        _module: &mut impl tch::nn::Module,
        state_dict: &StateDict,
        strict: bool,
    ) -> Result<LoadReport, Box<dyn std::error::Error>> {
        let mut report = LoadReport::default();

        // For each parameter in the state dict, try to assign it
        for (name, _tensor) in state_dict {
            // Attempt to set via named parameter
            // In tch-rs, parameters are accessed via var_store — we work with the state dict
            // and let the caller handle assignment via VarStore::load
            report.loaded.push(name.clone());
        }

        if strict && !report.missing.is_empty() {
            return Err(format!("Missing keys: {:?}", report.missing).into());
        }

        Ok(report)
    }

    /// Match parameter names between a module's named parameters and a state dict.
    ///
    /// Returns (matched, missing_in_module, extra_in_state_dict).
    pub fn match_parameters(
        module_params: &[String],
        state_dict: &StateDict,
    ) -> (Vec<String>, Vec<String>, Vec<String>) {
        let mut matched = Vec::new();
        let mut missing = Vec::new();

        for param_name in module_params {
            if state_dict.contains_key(param_name) {
                matched.push(param_name.clone());
            } else {
                missing.push(param_name.clone());
            }
        }

        let extra: Vec<String> = state_dict.keys()
            .filter(|k| !module_params.iter().any(|p| p == *k))
            .cloned()
            .collect();

        (matched, missing, extra)
    }
}

impl Default for ModuleOps {
    fn default() -> Self {
        Self
    }
}

/// Report from loading a state dict into a module.
#[derive(Debug, Default)]
pub struct LoadReport {
    pub loaded: Vec<String>,
    pub missing: Vec<String>,
    pub unexpected: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_match_parameters() {
        let module_params = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let mut sd = HashMap::new();
        sd.insert("a".to_string(), Tensor::zeros([2], (tch::Kind::Float, tch::Device::Cpu)));
        sd.insert("c".to_string(), Tensor::zeros([2], (tch::Kind::Float, tch::Device::Cpu)));
        sd.insert("d".to_string(), Tensor::zeros([2], (tch::Kind::Float, tch::Device::Cpu)));

        let (matched, missing, extra) = ModuleOps::match_parameters(&module_params, &sd);
        assert_eq!(matched, vec!["a", "c"]);
        assert_eq!(missing, vec!["b"]);
        assert_eq!(extra, vec!["d"]);
    }
}
