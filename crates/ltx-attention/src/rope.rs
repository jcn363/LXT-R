use tch::Tensor;
use ltx_types::ROPE_FREQ_SCALE;

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
    let last_dim = x.size().last().copied().unwrap();
    let d = last_dim / 2;
    let shape = {
        let mut s = x.size();
        s.push(d);
        s.push(2);
        s
    };
    let t_dup = x.reshape(&shape);
    let t1 = t_dup.narrow(-1, 0, 1);
    let t2 = t_dup.narrow(-1, 1, 1);
    let t_rot = Tensor::stack(&[&(-t2), &t1], -1).reshape(&x.size());
    x * cos + t_rot * sin
}

fn apply_split(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
    let d = x.size().last().copied().unwrap() / 2;
    let x1 = x.narrow(-1, 0, d);
    let x2 = x.narrow(-1, d, d);
    let rotated = Tensor::cat(&[&(-x2), &x1], -1);
    x * cos + rotated * sin
}
