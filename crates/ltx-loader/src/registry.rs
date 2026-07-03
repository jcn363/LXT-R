use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::primitives::StateDict;
use crate::safetensors_loader::SafetensorsStateDictLoader;
use crate::primitives::StateDictLoader;

/// Registry mapping model names to their checkpoint paths and loaders.
pub struct StateDictRegistry {
    entries: HashMap<String, RegistryEntry>,
}

struct RegistryEntry {
    path: PathBuf,
    format: CheckpointFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointFormat {
    Safetensors,
    Bin,
    Auto,
}

impl StateDictRegistry {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Register a model with its checkpoint path.
    pub fn register(&mut self, name: &str, path: PathBuf, format: CheckpointFormat) {
        self.entries.insert(name.to_string(), RegistryEntry { path, format });
    }

    /// Get the checkpoint path for a model.
    pub fn get_path(&self, name: &str) -> Option<&Path> {
        self.entries.get(name).map(|e| e.path.as_path())
    }

    /// Load a state dict by model name.
    #[must_use = "caller must handle registry load error"]
    pub fn load(&self, name: &str) -> Result<StateDict, Box<dyn std::error::Error>> {
        let entry = self.entries.get(name)
            .ok_or_else(|| format!("Model '{}' not registered", name))?;

        let loader: Box<dyn StateDictLoader> = match entry.format {
            CheckpointFormat::Safetensors | CheckpointFormat::Auto => {
                Box::new(SafetensorsStateDictLoader::new())
            }
            CheckpointFormat::Bin => {
                return Err("Binary format not yet supported".into());
            }
        };

        loader.load(&entry.path)
    }

    /// List all registered model names.
    pub fn list(&self) -> Vec<&str> {
        self.entries.keys().map(|s| s.as_str()).collect()
    }

    /// Check if a model is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// Remove a model from the registry.
    pub fn unregister(&mut self, name: &str) -> bool {
        self.entries.remove(name).is_some()
    }
}

impl Default for StateDictRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_register_and_list() {
        let mut reg = StateDictRegistry::new();
        reg.register("model_a", PathBuf::from("/tmp/a.safetensors"), CheckpointFormat::Safetensors);
        reg.register("model_b", PathBuf::from("/tmp/b.safetensors"), CheckpointFormat::Safetensors);

        let mut list = reg.list();
        list.sort();
        assert_eq!(list, vec!["model_a", "model_b"]);
    }

    #[test]
    fn test_registry_contains() {
        let mut reg = StateDictRegistry::new();
        reg.register("test", PathBuf::from("/tmp/test.safetensors"), CheckpointFormat::Auto);
        assert!(reg.contains("test"));
        assert!(!reg.contains("other"));
    }

    #[test]
    fn test_registry_unregister() {
        let mut reg = StateDictRegistry::new();
        reg.register("test", PathBuf::from("/tmp/test.safetensors"), CheckpointFormat::Safetensors);
        assert!(reg.unregister("test"));
        assert!(!reg.contains("test"));
        assert!(!reg.unregister("test"));
    }
}
