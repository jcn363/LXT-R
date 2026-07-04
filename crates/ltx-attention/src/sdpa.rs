use tch::{Kind, Tensor};

pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    is_causal: bool,
) -> Tensor {
    // tch 0.16 FFI: passing None for the mask optional sends nullptr through C,
    // which PyTorch interprets as a default-constructed Tensor (not std::nullopt).
    // Workaround: construct an all-True boolean mask when none is provided.
    let mask = match mask {
        Some(m) => Some(m.shallow_clone()),
        None => {
            let seq_q = q.size()[2];
            let seq_k = k.size()[2];
            Some(Tensor::ones([seq_q, seq_k], (Kind::Bool, q.device())))
        }
    };
    Tensor::scaled_dot_product_attention(q, k, v, mask, 0.0, is_causal, None)
}
