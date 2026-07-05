use ltx_text_encoder::embeddings_processor::EmbeddingsProcessor;
use tch::{Device, Kind, Tensor};

fn make_vs() -> tch::nn::VarStore {
    tch::nn::VarStore::new(Device::Cpu)
}

#[test]
fn test_embeddings_processor_output_shape() {
    let vs = make_vs();
    let ep = EmbeddingsProcessor::new(vs.root(), 256, 64);
    let x = Tensor::randn([1, 10, 256], (Kind::Float, Device::Cpu));
    let out = ep.forward(&x);
    assert_eq!(out.size(), vec![1, 10, 64]);
}

#[test]
fn test_embeddings_processor_hidden_size() {
    let vs = make_vs();
    let ep = EmbeddingsProcessor::new(vs.root(), 128, 32);
    assert_eq!(ep.hidden_size(), 32);
}

#[test]
fn test_embeddings_processor_extract_cls() {
    let vs = make_vs();
    let ep = EmbeddingsProcessor::new(vs.root(), 64, 32);
    let x = Tensor::randn([1, 8, 64], (Kind::Float, Device::Cpu));
    let cls = ep.extract_cls(&x);
    assert_eq!(cls.size(), vec![1, 64]);
}

#[test]
fn test_embeddings_processor_mean_pool() {
    let vs = make_vs();
    let ep = EmbeddingsProcessor::new(vs.root(), 64, 32);
    let x = Tensor::randn([2, 10, 64], (Kind::Float, Device::Cpu));
    let pooled = ep.mean_pool(&x);
    assert_eq!(pooled.size(), vec![2, 64]);
}

#[test]
fn test_embeddings_processor_batch_preservation() {
    let vs = make_vs();
    let ep = EmbeddingsProcessor::new(vs.root(), 32, 16);
    for b in [1, 3, 8] {
        let x = Tensor::randn([b, 5, 32], (Kind::Float, Device::Cpu));
        let out = ep.forward(&x);
        assert_eq!(out.size()[0], b);
    }
}

#[test]
fn test_embeddings_processor_seq_preservation() {
    let vs = make_vs();
    let ep = EmbeddingsProcessor::new(vs.root(), 32, 16);
    for seq in [1, 10, 50] {
        let x = Tensor::randn([1, seq, 32], (Kind::Float, Device::Cpu));
        let out = ep.forward(&x);
        assert_eq!(out.size()[1], seq);
    }
}
