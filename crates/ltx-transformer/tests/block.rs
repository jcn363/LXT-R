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

/// Bug verification: modulation chunks must broadcast across the sequence dimension.
/// The adaln modulation is [B, dim] after chunk, then unsqueezed to [B, 1, dim].
/// This means every position in the sequence gets the same modulation.
/// We verify the block produces correct output shape with varying sequence lengths.
#[test]
fn test_block_modulation_uniform_across_seq() {
    let vs = make_vs();
    let dim = 64;
    let block = BasicAVTransformerBlock::new(&vs.root(), dim, 4, 16, None, RopeType::Interleaved);

    let timestep = Tensor::from_slice(&[0.5]).to_kind(Kind::Float);
    let context = Tensor::randn([1, 4, dim], (Kind::Float, Device::Cpu));

    // Verify shape is correct for seq_len=2
    let x = Tensor::randn([1, 2, dim], (Kind::Float, Device::Cpu));
    let out = block.forward(&x, &timestep, &context, None, None);
    assert_eq!(out.size(), vec![1, 2, dim]);

    // Verify shape is correct for seq_len=50 (modulation must broadcast across all positions)
    let x50 = Tensor::randn([1, 50, dim], (Kind::Float, Device::Cpu));
    let out50 = block.forward(&x50, &timestep, &context, None, None);
    assert_eq!(out50.size(), vec![1, 50, dim]);
}

/// Stress test: large batch and sequence lengths with the block.
#[test]
fn test_block_large_batch_large_seq() {
    let vs = make_vs();
    let dim = 64;
    let block = BasicAVTransformerBlock::new(&vs.root(), dim, 4, 16, None, RopeType::Interleaved);

    let batch = 4;
    let seq = 20;
    let x = Tensor::randn([batch, seq, dim], (Kind::Float, Device::Cpu));
    let timestep = Tensor::randn([batch], (Kind::Float, Device::Cpu));
    let context = Tensor::randn([batch, 8, dim], (Kind::Float, Device::Cpu));

    let out = block.forward(&x, &timestep, &context, None, None);
    assert_eq!(out.size(), vec![batch, seq, dim]);
}

/// Regression: sinusoidal embedding NaN bug caused block output to be NaN.
/// The bug was `(-max_period).ln()` (log of negative = NaN) instead of
/// `-(max_period).ln()` (negate the log result). Fixed in sinusoidal.rs.
#[test]
fn test_block_output_finite_and_nonzero() {
    let vs = make_vs();
    let dim = 64;
    let block = BasicAVTransformerBlock::new(&vs.root(), dim, 4, 16, None, RopeType::Interleaved);

    let x = Tensor::randn([1, 4, dim], (Kind::Float, Device::Cpu));
    let timestep = Tensor::from_slice(&[0.5]).to_kind(Kind::Float);
    let context = Tensor::randn([1, 3, dim], (Kind::Float, Device::Cpu));

    let out = block.forward(&x, &timestep, &context, None, None);

    assert_eq!(out.size(), vec![1, 4, dim]);
    assert_eq!(out.isnan().any().double_value(&[]), 0.0, "block output contains NaN");
    assert!(out.abs().sum(Kind::Float).double_value(&[]) > 0.0, "block output is all zeros");
}
