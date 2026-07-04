use ltx_text_encoder::embeddings_connector::EmbeddingsConnector;
use tch::{Device, Kind, Tensor};

#[test]
fn test_concatenate_embeddings() {
    let conn = EmbeddingsConnector::new();
    let text = Tensor::randn([1, 5, 64], (Kind::Float, Device::Cpu));
    let vision = Tensor::randn([1, 10, 64], (Kind::Float, Device::Cpu));
    let combined = conn.concatenate(&text, &vision);
    // Concatenated along seq dim: 5 + 10 = 15
    assert_eq!(combined.size(), vec![1, 15, 64]);
}

#[test]
fn test_concatenate_preserves_batch() {
    let conn = EmbeddingsConnector::new();
    let text = Tensor::randn([3, 4, 32], (Kind::Float, Device::Cpu));
    let vision = Tensor::randn([3, 6, 32], (Kind::Float, Device::Cpu));
    let combined = conn.concatenate(&text, &vision);
    assert_eq!(combined.size()[0], 3);
}

#[test]
fn test_create_cross_mask_shape() {
    let conn = EmbeddingsConnector::new();
    let mask = conn.create_cross_mask(5, 10, Device::Cpu);
    assert_eq!(mask.size(), vec![15, 15]);
}

#[test]
fn test_create_cross_mask_blocked_regions() {
    let conn = EmbeddingsConnector::new();
    let text_len = 3;
    let vision_len = 4;
    let mask = conn.create_cross_mask(text_len, vision_len, Device::Cpu);

    // Top-right block (text→vision) should be True (masked)
    let tr = mask.narrow(0, 0, text_len).narrow(1, text_len, vision_len);
    assert_eq!(
        tr.sum(tch::Kind::Float).double_value(&[]),
        (text_len * vision_len) as f64
    );

    // Bottom-left block (vision→text) should be True (masked)
    let bl = mask.narrow(0, text_len, vision_len).narrow(1, 0, text_len);
    assert_eq!(
        bl.sum(tch::Kind::Float).double_value(&[]),
        (text_len * vision_len) as f64
    );

    // Top-left block (text→text) should be False (not masked)
    let tl = mask.narrow(0, 0, text_len).narrow(1, 0, text_len);
    assert_eq!(tl.sum(tch::Kind::Float).double_value(&[]), 0.0);

    // Bottom-right block (vision→vision) should be False (not masked)
    let br = mask
        .narrow(0, text_len, vision_len)
        .narrow(1, text_len, vision_len);
    assert_eq!(br.sum(tch::Kind::Float).double_value(&[]), 0.0);
}

#[test]
fn test_concatenate_default() {
    let conn = EmbeddingsConnector;
    let text = Tensor::randn([1, 2, 16], (Kind::Float, Device::Cpu));
    let vision = Tensor::randn([1, 3, 16], (Kind::Float, Device::Cpu));
    let combined = conn.concatenate(&text, &vision);
    assert_eq!(combined.size(), vec![1, 5, 16]);
}
