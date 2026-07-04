use ltx_types::ROPE_FREQ_SCALE;
use tch::Tensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeType {
    Interleaved,
    Split,
}

pub fn apply_rotary_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    rope_type: RopeType,
) -> (Tensor, Tensor) {
    let q_rot = match rope_type {
        RopeType::Interleaved => apply_interleaved(q, cos, sin),
        RopeType::Split => apply_split(q, cos, sin),
    };
    let k_rot = match rope_type {
        RopeType::Interleaved => apply_interleaved(k, cos, sin),
        RopeType::Split => apply_split(k, cos, sin),
    };
    (q_rot, k_rot)
}

pub fn precompute_freqs_cis(
    dim: i64,
    max_seq_len: i64,
    theta: f64,
    rope_type: RopeType,
    device: tch::Device,
) -> (Tensor, Tensor) {
    let half_dim = dim / 2;
    let range = Tensor::arange_start(0, half_dim, (tch::Kind::Float, device));
    let theta_t = Tensor::from_slice(&[theta as f32]).to_device(device);
    let inv_freq = theta_t.pow(&(2.0 * range / dim as f64)).reciprocal();
    let t = Tensor::arange_start(0, max_seq_len, (tch::Kind::Float, device)).unsqueeze(1);
    let freqs = t * inv_freq.unsqueeze(0) * ROPE_FREQ_SCALE;
    let cos = freqs.cos();
    let sin = freqs.sin();
    match rope_type {
        RopeType::Interleaved => (
            cos.repeat_interleave_self_int(2, -1, None),
            sin.repeat_interleave_self_int(2, -1, None),
        ),
        RopeType::Split => (cos, sin),
    }
}

fn apply_interleaved(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
    let dims = x.size();
    let last_dim = dims
        .last()
        .expect("apply_interleaved: tensor must have at least one dimension");
    let half = last_dim / 2;
    // Reshape [..., dim] → [..., half, 2] to expose consecutive pairs
    let mut shape: Vec<i64> = x.size();
    *shape
        .last_mut()
        .expect("apply_interleaved: tensor must have at least one dimension") = half;
    shape.push(2);
    let x_pairs = x.reshape(&shape); // [..., half, 2]
    let x_even = x_pairs.narrow(-1, 0, 1); // [..., half, 1]
    let x_odd = x_pairs.narrow(-1, 1, 1); // [..., half, 1]
                                          // Pair-wise rotation: [even, odd] → [-odd, even] (90° rotation)
    let x_rot = Tensor::cat(&[&(-x_odd), &x_even], -1).reshape(x.size());
    x * cos + x_rot * sin
}

fn apply_split(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
    let d = x
        .size()
        .last()
        .expect("apply_split: tensor must have at least one dimension")
        / 2;
    let x1 = x.narrow(-1, 0, d); // first half
    let x2 = x.narrow(-1, d, d); // second half
    let out1 = x1.shallow_clone() * cos - x2.shallow_clone() * sin;
    let out2 = x2 * cos + x1 * sin;
    Tensor::cat(&[&out1, &out2], -1)
}
