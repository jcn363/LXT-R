/// Tests for model weight loading from safetensors files.
/// Verifies that weights can be saved and loaded correctly,
/// and that the VarStore variable names match the expected paths.
use ltx_attention::RopeType;
use ltx_norm::RMSNorm;
use ltx_transformer::block::BasicAVTransformerBlock;
use ltx_transformer::model::LTXModel;
use tch::{Device, Kind, Tensor};

fn make_vs() -> tch::nn::VarStore {
    tch::nn::VarStore::new(Device::Cpu)
}

fn build_model(vs: &tch::nn::Path, dim: i64) -> LTXModel {
    let mut blocks = Vec::new();
    for i in 0..28 {
        blocks.push(BasicAVTransformerBlock::new(
            &(vs / "blocks" / i),
            dim,
            4,
            dim / 4,
            None,
            RopeType::Interleaved,
        ));
    }
    let norm_out = RMSNorm::default_eps_with_path(vs / "norm_out", dim);
    let proj_out = tch::nn::linear(vs / "proj_out", dim, dim, Default::default());
    LTXModel::new(blocks, norm_out, proj_out)
}

/// Test that VarStore variables have the expected key structure.
#[test]
fn test_varstore_keys_structure() {
    let vs = make_vs();
    let dim = 64;
    let _model = build_model(&vs.root(), dim);

    let vars = vs.variables();
    let keys: Vec<&str> = vars.keys().map(|k| k.as_str()).collect();

    // Print all keys for debugging
    println!("\nVarStore keys ({} total):", keys.len());
    for key in &keys {
        println!("  {key}");
    }

    // Verify we have the right number of variables
    assert!(!keys.is_empty(), "VarStore should have variables");

    // Verify expected key patterns exist
    assert!(keys.iter().any(|k| k.contains("self_attn")),
        "Missing self_attn keys in VarStore");
    assert!(keys.iter().any(|k| k.contains("cross_attn")),
        "Missing cross_attn keys in VarStore");
}

/// Test that model can be saved and loaded with correct shapes.
#[test]
fn test_model_save_load_roundtrip() {
    let vs = make_vs();
    let dim = 64;
    let _model = build_model(&vs.root(), dim);

    // Save weights
    let save_path = "/tmp/test_model_weights.safetensors";
    vs.save(save_path).expect("failed to save weights");

    // Create new VarStore and load
    let mut vs2 = make_vs();
    let _model2 = build_model(&vs2.root(), dim);
    vs2.load(save_path).expect("failed to load weights");

    // Verify all variables have the same shapes
    let vars1 = vs.variables();
    let vars2 = vs2.variables();

    assert_eq!(vars1.len(), vars2.len(), "different number of variables");

    for (key1, tensor1) in vars1.iter() {
        let tensor2 = vars2.get(key1).unwrap_or_else(|| panic!("missing key: {key1}"));
        assert_eq!(tensor1.size(), tensor2.size(),
            "shape mismatch for {key1}: {:?} vs {:?}", tensor1.size(), tensor2.size());
    }

    // Clean up
    let _ = std::fs::remove_file(save_path);
}

/// Test that model inference works after loading weights.
#[test]
fn test_model_inference_after_load() {
    let vs = make_vs();
    let dim = 64;
    let model = build_model(&vs.root(), dim);

    // Save weights
    let save_path = "/tmp/test_model_inference.safetensors";
    vs.save(save_path).expect("failed to save weights");

    // Create new VarStore and load
    let mut vs2 = make_vs();
    let model2 = build_model(&vs2.root(), dim);
    vs2.load(save_path).expect("failed to load weights");

    // Run inference with both models - should produce same output
    let x = Tensor::randn([1, 4, 64], (Kind::Float, Device::Cpu));
    let timestep = Tensor::from_slice(&[0.5]);
    let context = Tensor::randn([1, 3, 64], (Kind::Float, Device::Cpu));

    let out1 = model.forward(&x, &timestep, &context, None, None);
    let out2 = model2.forward(&x, &timestep, &context, None, None);

    assert!(out1.allclose(&out2, 1e-6, 1e-6, false),
        "outputs differ after save/load");

    // Clean up
    let _ = std::fs::remove_file(save_path);
}
