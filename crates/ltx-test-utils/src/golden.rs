use tch::Tensor;

/// Load a tensor from a safetensors file as f32.
///
/// All dtypes are converted to f32 for comparison — golden tests
/// compare numerical values, not storage formats.
pub fn load_golden(path: &str, name: &str) -> Tensor {
    let data = std::fs::read(path).expect("failed to read safetensors file");
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
    let data = std::fs::read(path).expect("failed to read safetensors file");
    let tensors = safetensors::SafeTensors::deserialize(&data).expect("failed to deserialize");
    tensors.names().iter().map(|s| s.to_string()).collect()
}
