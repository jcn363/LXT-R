use ltx_attention::RopeType;
use ltx_transformer::block::BasicAVTransformerBlock;
use tch::{Device, Kind, Tensor};

fn make_vs() -> tch::nn::VarStore {
    tch::nn::VarStore::new(Device::Cpu)
}

#[test]
fn test_basic_block_output_shape() {
    let vs = make_vs();
    let block = BasicAVTransformerBlock::new(&vs.root(), 64, 4, 16, None, RopeType::Interleaved);
    let x = Tensor::randn([1, 8, 64], (Kind::Float, Device::Cpu));
    let timestep = Tensor::randn([1], (Kind::Float, Device::Cpu));
    let context = Tensor::randn([1, 5, 64], (Kind::Float, Device::Cpu));
    let out = block.forward(&x, &timestep, &context, None, None);
    assert_eq!(out.size(), vec![1, 8, 64]);
}

#[test]
fn test_basic_block_preserves_sequence_length() {
    let vs = make_vs();
    let block = BasicAVTransformerBlock::new(&vs.root(), 32, 2, 16, None, RopeType::Split);
    let seq_len = 12;
    let x = Tensor::randn([2, seq_len, 32], (Kind::Float, Device::Cpu));
    let timestep = Tensor::randn([2], (Kind::Float, Device::Cpu));
    let context = Tensor::randn([2, 6, 32], (Kind::Float, Device::Cpu));
    let out = block.forward(&x, &timestep, &context, None, None);
    assert_eq!(out.size(), vec![2, seq_len, 32]);
}

#[test]
fn test_basic_block_with_cross_context() {
    let vs = make_vs();
    let context_dim = 48;
    let block = BasicAVTransformerBlock::new(
        &vs.root(),
        64,
        4,
        16,
        Some(context_dim),
        RopeType::Interleaved,
    );
    let x = Tensor::randn([1, 6, 64], (Kind::Float, Device::Cpu));
    let timestep = Tensor::randn([1], (Kind::Float, Device::Cpu));
    let context = Tensor::randn([1, 4, context_dim], (Kind::Float, Device::Cpu));
    let out = block.forward(&x, &timestep, &context, None, None);
    assert_eq!(out.size(), vec![1, 6, 64]);
}
