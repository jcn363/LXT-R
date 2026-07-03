use std::collections::HashMap;
use tch::Tensor;

/// Type alias for a state dict: mapping weight names to tensors.
pub type StateDict = HashMap<String, Tensor>;

/// Trait for loading state dicts from various formats.
pub trait StateDictLoader {
    /// Load a state dict from the given path.
    fn load(&self, path: &std::path::Path) -> Result<StateDict, Box<dyn std::error::Error>>;

    /// Check if the loader supports the given file extension.
    fn supports_extension(&self, ext: &str) -> bool;
}
