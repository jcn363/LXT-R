use std::path::PathBuf;
use tch::Tensor;

/// Load a tensor from a safetensors file as f32.
///
/// All dtypes are converted to f32 for comparison — golden tests
/// compare numerical values, not storage formats.
///
/// The path is resolved relative to the workspace root (where Cargo.toml is).
pub fn load_golden(path: &str, name: &str) -> Tensor {
    // Try to find the file relative to workspace root
    let workspace_root = find_workspace_root();
    let full_path = workspace_root.join(path);

    let data = std::fs::read(&full_path)
        .unwrap_or_else(|e| panic!("failed to read safetensors file at {:?}: {}", full_path, e));
    let tensors = safetensors::SafeTensors::deserialize(&data).expect("failed to deserialize");
    let view = tensors.tensor(name).expect("tensor not found");
    let dtype = view.dtype();
    let shape: Vec<i64> = view.shape().iter().map(|&d| d as i64).collect();
    let bytes = view.data();

    match dtype {
        safetensors::Dtype::F32 => {
            let floats: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Tensor::from_slice(&floats).reshape(&shape)
        }
        safetensors::Dtype::F16 => {
            let floats: Vec<f32> = bytes
                .chunks_exact(2)
                .map(|c| {
                    let h = u16::from_le_bytes([c[0], c[1]]);
                    let sign = ((h >> 15) as f32) * -2.0 + 1.0;
                    let exp = ((h >> 10) & 0x1F) as i32 - 15;
                    let mantissa = (h & 0x3FF) as f32 / 1024.0 + 1.0;
                    sign * 2f32.powi(exp) * mantissa
                })
                .collect();
            Tensor::from_slice(&floats).reshape(&shape)
        }
        _ => panic!("unsupported dtype for golden: {:?}", dtype),
    }
}

/// List all tensor names in a safetensors file.
pub fn list_golden_tensors(path: &str) -> Vec<String> {
    let workspace_root = find_workspace_root();
    let full_path = workspace_root.join(path);
    let data = std::fs::read(&full_path).expect("failed to read safetensors file");
    let tensors = safetensors::SafeTensors::deserialize(&data).expect("failed to deserialize");
    tensors.names().iter().map(|s| s.to_string()).collect()
}

/// Find the workspace root by looking for Cargo.toml.
fn find_workspace_root() -> PathBuf {
    // Try CARGO_MANIFEST_DIR first (for the crate being compiled)
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        let manifest_path = PathBuf::from(&manifest_dir);
        // The workspace root is typically 2 levels up from a crate (crates/<name>)
        if let Some(parent) = manifest_path.parent() {
            if let Some(workspace) = parent.parent() {
                if workspace.join("Cargo.toml").exists() {
                    return workspace.to_path_buf();
                }
            }
        }
    }

    // Fallback: try current directory
    PathBuf::from(".")
}
