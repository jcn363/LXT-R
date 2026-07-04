use ltx_conditioning::{causal_mask, padding_mask, cross_attention_mask};
use tch::Device;

#[test]
fn test_causal_mask_upper_triangular() {
    let mask = causal_mask(4, Device::Cpu);
    // Diagonal and below should be false (not masked)
    assert_eq!(mask.get(0).get(0).double_value(&[]), 0.0);
    assert_eq!(mask.get(1).get(0).double_value(&[]), 0.0);
    assert_eq!(mask.get(1).get(1).double_value(&[]), 0.0);
    // Above diagonal should be true (masked)
    assert_eq!(mask.get(0).get(1).double_value(&[]), 1.0);
    assert_eq!(mask.get(0).get(3).double_value(&[]), 1.0);
    assert_eq!(mask.get(2).get(3).double_value(&[]), 1.0);
}

#[test]
fn test_causal_mask_single_element() {
    let mask = causal_mask(1, Device::Cpu);
    assert_eq!(mask.size(), vec![1, 1]);
    assert_eq!(mask.get(0).get(0).double_value(&[]), 0.0);
}

#[test]
fn test_padding_mask_all_valid() {
    let mask = padding_mask(&[5, 5, 5], Device::Cpu);
    assert_eq!(mask.size(), vec![3, 5]);
    // All positions should be valid
    for b in 0..3 {
        for p in 0..5 {
            assert_eq!(mask.get(b).get(p).double_value(&[]), 1.0);
        }
    }
}

#[test]
fn test_padding_mask_empty_lengths() {
    let mask = padding_mask(&[], Device::Cpu);
    assert_eq!(mask.size(), vec![0, 0]);
}

#[test]
fn test_cross_attention_mask_shape() {
    let mask = cross_attention_mask(&[3, 5], &[4, 6], Device::Cpu);
    assert_eq!(mask.size(), vec![2, 6, 5]);
}

#[test]
fn test_cross_attention_mask_valid_positions() {
    let mask = cross_attention_mask(&[2, 3], &[2, 3], Device::Cpu);
    // For batch 0: encoder has 2 valid positions, decoder has 2
    // Positions 0,1 in decoder can attend to encoder positions 0,1 (not masked)
    // Position 0,1 in decoder cannot attend to encoder position 2 (masked)
    assert_eq!(mask.get(0).get(0).get(0).double_value(&[]), 0.0);
    assert_eq!(mask.get(0).get(0).get(2).double_value(&[]), 1.0);
}

#[test]
#[should_panic]
fn test_cross_attention_mask_mismatched_batch() {
    let _ = cross_attention_mask(&[3], &[4, 5], Device::Cpu);
}
