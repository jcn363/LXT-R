use crate::primitives::StateDict;

/// Operations on state dicts for key matching, replacement, and filtering.
pub struct SDOps;

impl SDOps {
    pub fn new() -> Self {
        Self
    }

    /// Filter state dict to only include keys matching a prefix.
    pub fn filter_prefix(state_dict: &StateDict, prefix: &str) -> StateDict {
        state_dict
            .iter()
            .filter(|(k, _)| k.starts_with(prefix))
            .map(|(k, v)| (k.clone(), v.shallow_clone()))
            .collect()
    }

    /// Remove prefix from all keys in a state dict.
    pub fn strip_prefix(state_dict: &StateDict, prefix: &str) -> StateDict {
        state_dict
            .iter()
            .filter_map(|(k, v)| {
                k.strip_prefix(prefix)
                    .map(|stripped| (stripped.to_string(), v.shallow_clone()))
            })
            .collect()
    }

    /// Replace a substring in all keys of a state dict.
    pub fn rename_keys(state_dict: &StateDict, from: &str, to: &str) -> StateDict {
        state_dict
            .iter()
            .map(|(k, v)| {
                let new_key = k.replace(from, to);
                (new_key, v.shallow_clone())
            })
            .collect()
    }

    /// Merge two state dicts, with `override_dict` taking precedence on key conflicts.
    pub fn merge(base: &StateDict, override_dict: &StateDict) -> StateDict {
        let mut result: StateDict = base
            .iter()
            .map(|(k, v)| (k.clone(), v.shallow_clone()))
            .collect();
        for (k, v) in override_dict {
            result.insert(k.clone(), v.shallow_clone());
        }
        result
    }

    /// Find keys present in `source` but missing from `target`.
    pub fn missing_keys(source: &StateDict, target: &StateDict) -> Vec<String> {
        source
            .keys()
            .filter(|k| !target.contains_key(*k))
            .cloned()
            .collect()
    }

    /// Find keys present in both dicts but with mismatched shapes.
    pub fn shape_mismatches(a: &StateDict, b: &StateDict) -> Vec<(String, Vec<i64>, Vec<i64>)> {
        let mut mismatches = Vec::new();
        for (k, va) in a {
            if let Some(vb) = b.get(k) {
                if va.size() != vb.size() {
                    mismatches.push((k.clone(), va.size(), vb.size()));
                }
            }
        }
        mismatches
    }

    /// Cast all tensors in a state dict to the given dtype.
    pub fn cast_dtype(state_dict: &StateDict, kind: tch::Kind) -> StateDict {
        state_dict
            .iter()
            .map(|(k, v)| (k.clone(), v.to_kind(kind)))
            .collect()
    }
}

impl Default for SDOps {
    fn default() -> Self {
        Self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tch::Tensor;

    fn make_sd(keys: &[&str]) -> StateDict {
        keys.iter()
            .map(|k| {
                (
                    k.to_string(),
                    Tensor::zeros([2, 2], (tch::Kind::Float, tch::Device::Cpu)),
                )
            })
            .collect()
    }

    #[test]
    fn test_filter_prefix() {
        let sd = make_sd(&["a.x", "a.y", "b.x"]);
        let filtered = SDOps::filter_prefix(&sd, "a.");
        assert_eq!(filtered.len(), 2);
        assert!(filtered.contains_key("a.x"));
        assert!(filtered.contains_key("a.y"));
    }

    #[test]
    fn test_strip_prefix() {
        let sd = make_sd(&["model.layer.weight"]);
        let stripped = SDOps::strip_prefix(&sd, "model.");
        assert!(stripped.contains_key("layer.weight"));
        assert!(!stripped.contains_key("model.layer.weight"));
    }

    #[test]
    fn test_rename_keys() {
        let sd = make_sd(&["old_name"]);
        let renamed = SDOps::rename_keys(&sd, "old", "new");
        assert!(renamed.contains_key("new_name"));
    }

    #[test]
    fn test_missing_keys() {
        let a = make_sd(&["a", "b", "c"]);
        let b = make_sd(&["a", "b"]);
        let missing = SDOps::missing_keys(&a, &b);
        assert_eq!(missing, vec!["c"]);
    }

    #[test]
    fn test_shape_mismatches() {
        let mut a = HashMap::new();
        a.insert(
            "w".to_string(),
            Tensor::zeros([2, 3], (tch::Kind::Float, tch::Device::Cpu)),
        );
        let mut b = HashMap::new();
        b.insert(
            "w".to_string(),
            Tensor::zeros([2, 4], (tch::Kind::Float, tch::Device::Cpu)),
        );
        let mismatches = SDOps::shape_mismatches(&a, &b);
        assert_eq!(mismatches.len(), 1);
    }
}
