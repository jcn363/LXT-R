use tch::Tensor;

pub struct EmbeddingsConnector;

impl EmbeddingsConnector {
    pub fn new() -> Self {
        Self
    }

    /// Concatenate text and vision embeddings along the sequence dimension.
    pub fn concatenate(
        &self,
        text_embeds: &Tensor,
        vision_embeds: &Tensor,
    ) -> Tensor {
        Tensor::cat(&[text_embeds, vision_embeds], 1)
    }

    /// Create an attention mask that blocks cross-modal attention.
    pub fn create_cross_mask(
        &self,
        text_len: i64,
        vision_len: i64,
        device: tch::Device,
    ) -> Tensor {
        let total = text_len + vision_len;
        let mask = Tensor::zeros(&[total, total], (tch::Kind::Bool, device));

        // Block vision tokens from attending to text and vice versa
        let _ = mask.narrow(0, 0, text_len)
            .narrow(1, text_len, vision_len)
            .fill_(1i64);
        let _ = mask.narrow(0, text_len, vision_len)
            .narrow(1, 0, text_len)
            .fill_(1i64);

        mask
    }
}
