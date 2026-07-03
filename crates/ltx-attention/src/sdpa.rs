use tch::Tensor;

pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    is_causal: bool,
) -> Tensor {
    let mask_owned = mask.map(|m| m.shallow_clone());
    Tensor::scaled_dot_product_attention(q, k, v, mask_owned, 0.0, is_causal, None)
}
