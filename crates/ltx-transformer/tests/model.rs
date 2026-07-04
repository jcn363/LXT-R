use ltx_attention::RopeType;
use ltx_norm::RMSNorm;
use ltx_transformer::block::BasicAVTransformerBlock;
use ltx_transformer::model::LTXModel;
use tch::{Device, Kind, Tensor};

fn make_vs() -> tch::nn::VarStore {
    tch::nn::VarStore::new(Device::Cpu)
}

fn build_model(vs: &tch::nn::Path, num_layers: usize, dim: i64) -> LTXModel {
    let mut blocks = Vec::new();
    for i in 0..num_layers {
        blocks.push(BasicAVTransformerBlock::new(
            &(vs / "blocks" / i),
            dim,
            4,
            dim / 4,
            None,
            RopeType::Interleaved,
        ));
    }
    let norm_out = RMSNorm::default_eps(dim, vs.device());
    let proj_out = tch::nn::linear(vs / "proj_out", dim, dim, Default::default());
    LTXModel::new(blocks, norm_out, proj_out)
}

#[test]
fn test_model_single_block_output_shape() {
    let vs = make_vs();
    let model = build_model(&vs.root(), 1, 64);
    let latent = Tensor::randn([1, 8, 64], (Kind::Float, Device::Cpu));
    let timestep = Tensor::randn([1], (Kind::Float, Device::Cpu));
    let context = Tensor::randn([1, 5, 64], (Kind::Float, Device::Cpu));
    let out = model.forward(&latent, &timestep, &context, None, None);
    assert_eq!(out.size(), vec![1, 8, 64]);
}

#[test]
fn test_model_multi_block_output_shape() {
    let vs = make_vs();
    let model = build_model(&vs.root(), 4, 64);
    let latent = Tensor::randn([1, 6, 64], (Kind::Float, Device::Cpu));
    let timestep = Tensor::randn([1], (Kind::Float, Device::Cpu));
    let context = Tensor::randn([1, 10, 64], (Kind::Float, Device::Cpu));
    let out = model.forward(&latent, &timestep, &context, None, None);
    assert_eq!(out.size(), vec![1, 6, 64]);
}

#[test]
fn test_model_preserves_batch_and_seq() {
    let vs = make_vs();
    let model = build_model(&vs.root(), 2, 32);
    let (batch, seq) = (3, 7);
    let latent = Tensor::randn([batch, seq, 32], (Kind::Float, Device::Cpu));
    let timestep = Tensor::randn([batch], (Kind::Float, Device::Cpu));
    let context = Tensor::randn([batch, 4, 32], (Kind::Float, Device::Cpu));
    let out = model.forward(&latent, &timestep, &context, None, None);
    assert_eq!(out.size(), vec![batch, seq, 32]);
}

#[test]
fn test_model_debug_output() {
    let vs = make_vs();
    let model = build_model(&vs.root(), 3, 64);
    let debug_str = format!("{:?}", model);
    assert!(debug_str.contains("num_blocks"));
    assert!(debug_str.contains("3"));
}
