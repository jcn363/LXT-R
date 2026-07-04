use tch::Device;

/// Create a CPU `VarStore` — the most common test fixture.
pub fn make_vs() -> tch::nn::VarStore {
    tch::nn::VarStore::new(Device::Cpu)
}

/// Create a deterministic random tensor with a fixed seed.
///
/// Use this when you need reproducible tensor values across runs,
/// e.g. for golden file comparisons or regression tests.
pub fn make_seed_tensor(shape: &[i64], seed: i64) -> tch::Tensor {
    tch::manual_seed(seed);
    tch::Tensor::randn(shape, (tch::Kind::Float, Device::Cpu))
}
