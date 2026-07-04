use tch::Tensor;

/// Create a causal attention mask for autoregressive decoding.
///
/// Returns a upper-triangular boolean mask of shape `[seq_len, seq_len]`.
/// `true` positions are masked out (not attended to).
pub fn causal_mask(seq_len: i64, device: tch::Device) -> Tensor {
    let mask = Tensor::ones([seq_len, seq_len], (tch::Kind::Bool, device));
    mask.triu(1)
}

/// Create a padding mask for variable-length sequences.
///
/// `lengths` contains the valid length for each sequence in the batch.
/// Returns a boolean mask of shape `[batch, max_len]` where `true` = valid position.
pub fn padding_mask(lengths: &[i64], device: tch::Device) -> Tensor {
    let max_len = *lengths.iter().max().unwrap_or(&0);
    let range = Tensor::arange(max_len, (tch::Kind::Int64, device));
    let lengths_t = Tensor::from_slice(lengths).to_device(device).unsqueeze(1);
    // range < lengths_t using comparison method
    range.unsqueeze(0).lt_tensor(&lengths_t)
}

/// Create an attention mask combining causal and padding constraints.
///
/// Returns a boolean mask of shape `[batch, seq_len, seq_len]`.
/// `true` positions are masked out (not attended to).
pub fn causal_padding_mask(lengths: &[i64], device: tch::Device) -> Tensor {
    let batch = lengths.len() as i64;
    let max_len = *lengths.iter().max().unwrap_or(&0);
    let causal = causal_mask(max_len, device)
        .unsqueeze(0)
        .expand([batch, max_len, max_len], false);
    let pad = padding_mask(lengths, device);
    // pad[i][j] is true when position j is valid
    // We want to mask where j >= length_i OR where causal mask is set
    let pad_mask = pad
        .unsqueeze(2)
        .expand([batch, max_len, max_len], false)
        .to_kind(tch::Kind::Bool)
        .logical_not();
    causal.logical_or(&pad_mask)
}

/// Create a cross-attention mask for encoder-decoder attention.
///
/// `encoder_lengths` contains valid lengths for encoder sequences.
/// `decoder_lengths` contains valid lengths for decoder sequences.
/// Returns a boolean mask of shape `[batch, decoder_len, encoder_len]`.
pub fn cross_attention_mask(
    encoder_lengths: &[i64],
    decoder_lengths: &[i64],
    device: tch::Device,
) -> Tensor {
    assert_eq!(encoder_lengths.len(), decoder_lengths.len());
    let batch = encoder_lengths.len() as i64;
    let enc_max = *encoder_lengths.iter().max().unwrap_or(&0);
    let dec_max = *decoder_lengths.iter().max().unwrap_or(&0);

    // Encoder valid positions
    let enc_pad = padding_mask(encoder_lengths, device)
        .unsqueeze(1)
        .expand([batch, dec_max, enc_max], false);
    enc_pad.logical_not()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_mask_shape() {
        let mask = causal_mask(4, tch::Device::Cpu);
        assert_eq!(mask.size(), vec![4, 4]);
    }

    #[test]
    fn test_padding_mask() {
        let mask = padding_mask(&[3, 5, 2], tch::Device::Cpu);
        assert_eq!(mask.size(), vec![3, 5]);
        // First sequence has 3 valid positions
        assert_ne!(mask.get(0).get(2).double_value(&[]), 0.0);
        assert_eq!(mask.get(0).get(3).double_value(&[]), 0.0);
    }

    #[test]
    fn test_causal_padding_mask_shape() {
        let mask = causal_padding_mask(&[3, 4], tch::Device::Cpu);
        assert_eq!(mask.size(), vec![2, 4, 4]);
    }
}
