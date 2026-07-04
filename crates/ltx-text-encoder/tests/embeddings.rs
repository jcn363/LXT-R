use ltx_text_encoder::embeddings_processor::EmbeddingsProcessor;
use tch::{Device, Kind, Tensor};

#[test]
fn test_embeddings_processor_output_shape() {
    let ep = EmbeddingsProcessor::new(256, 64);
    let x = Tensor::randn([1, 10, 256], (Kind::Float, Device::Cpu));
    let out = ep.forward(&x);
    assert_eq!(out.size(), vec![1, 10, 64]);
}

#[test]
fn test_embeddings_processor_hidden_size() {
    let ep = EmbeddingsProcessor::new(128, 32);
    assert_eq!(ep.hidden_size(), 32);
}

#[test]
fn test_embeddings_processor_extract_cls() {
    let ep = EmbeddingsProcessor::new(64, 32);
    let x = Tensor::randn([1, 8, 64], (Kind::Float, Device::Cpu));
    let cls = ep.extract_cls(&x);
    // CLS is the first token: shape (B, hidden_size)
    assert_eq!(cls.size(), vec![1, 64]);
}

#[test]
fn test_embeddings_processor_mean_pool() {
    let ep = EmbeddingsProcessor::new(64, 32);
    let x = Tensor::randn([2, 10, 64], (Kind::Float, Device::Cpu));
    let pooled = ep.mean_pool(&x);
    // Mean pooling over seq dim: (B, hidden_size)
    assert_eq!(pooled.size(), vec![2, 64]);
}

#[test]
fn test_embeddings_processor_batch_preservation() {
    let ep = EmbeddingsProcessor::new(32, 16);
    for b in [1, 3, 8] {
        let x = Tensor::randn([b, 5, 32], (Kind::Float, Device::Cpu));
        let out = ep.forward(&x);
        assert_eq!(out.size()[0], b);
    }
}

#[test]
fn test_embeddings_processor_seq_preservation() {
    let ep = EmbeddingsProcessor::new(32, 16);
    for seq in [1, 10, 50] {
        let x = Tensor::randn([1, seq, 32], (Kind::Float, Device::Cpu));
        let out = ep.forward(&x);
        assert_eq!(out.size()[1], seq);
    }
}
