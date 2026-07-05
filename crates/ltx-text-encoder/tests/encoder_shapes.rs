use ltx_text_encoder::gemma3_text::Gemma3TextModel;
use ltx_text_encoder::config::Gemma3ConfigData;
use tch::{Device, Kind, Tensor};

fn small_config() -> Gemma3ConfigData {
    Gemma3ConfigData {
        hidden_size: 64,
        intermediate_size: 128,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 16,
        num_hidden_layers: 2,
        vocab_size: 1000,
        rms_norm_eps: 1e-6,
        hidden_act: String::from("silu"),
        rope_theta: 10000.0,
        max_position_embeddings: 256,
    }
}

#[test]
fn test_gemma3_text_model_output_shape() {
    let vs = tch::nn::VarStore::new(Device::Cpu);
    let config = small_config();
    let model = Gemma3TextModel::new(vs.root(), &config);

    // Input: batch=1, seq_len=8 token IDs
    let input_ids = Tensor::arange_start(0i64, 8, (Kind::Int64, Device::Cpu)).unsqueeze(0);
    let output = model.forward(&input_ids);

    // Output should be [B=1, seq_len=8, hidden_size=64]
    assert_eq!(output.size(), vec![1, 8, 64]);
}

#[test]
fn test_gemma3_text_model_varies_with_input() {
    let vs = tch::nn::VarStore::new(Device::Cpu);
    let config = small_config();
    let model = Gemma3TextModel::new(vs.root(), &config);

    let ids_a = Tensor::from_slice(&[1i64, 2, 3, 4]).unsqueeze(0);
    let ids_b = Tensor::from_slice(&[5i64, 6, 7, 8]).unsqueeze(0);

    let out_a = model.forward(&ids_a);
    let out_b = model.forward(&ids_b);

    // Different inputs should produce different outputs
    let diff = (&out_a - &out_b).abs().sum(Kind::Float).double_value(&[]);
    assert!(diff > 0.0, "different inputs should produce different outputs");
}

#[test]
fn test_gemma3_text_model_hidden_size() {
    let vs = tch::nn::VarStore::new(Device::Cpu);
    let config = small_config();
    let model = Gemma3TextModel::new(vs.root(), &config);
    assert_eq!(model.hidden_size(), 64);
}

#[test]
fn test_gemma3_text_model_fits_transformer_context() {
    let vs = tch::nn::VarStore::new(Device::Cpu);
    let config = small_config();
    let model = Gemma3TextModel::new(vs.root(), &config);

    // Simulate full pipeline: encode -> use as transformer context
    let input_ids = Tensor::from_slice(&[10i64, 20, 30, 40, 50]).unsqueeze(0);
    let context = model.forward(&input_ids);

    // Transformer expects [B, seq_len, context_dim]
    assert_eq!(context.size()[0], 1, "batch dim");
    assert_eq!(context.size()[1], 5, "seq_len matches input");
    assert_eq!(context.size()[2], 64, "hidden_size = context_dim");

    // Verify it can be used in matmul with transformer output
    let transformer_out = Tensor::randn([1, 5, 64], (Kind::Float, Device::Cpu));
    let cross_attn = transformer_out.matmul(&context.transpose(1, 2));
    assert_eq!(cross_attn.size(), vec![1, 5, 5]);
}
