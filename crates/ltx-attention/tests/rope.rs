use ltx_attention::{apply_rotary_emb, precompute_freqs_cis, RopeType};
use tch::{Device, Kind, Tensor};

#[test]
fn test_precompute_interleaved_rope_shape() {
    let (cos, sin) = precompute_freqs_cis(64, 16, 10_000.0, RopeType::Interleaved, Device::Cpu);
    assert_eq!(cos.size(), vec![16, 64]);
    assert_eq!(sin.size(), vec![16, 64]);
}

#[test]
fn test_precompute_split_rope_shape() {
    let (cos, sin) = precompute_freqs_cis(64, 16, 10_000.0, RopeType::Split, Device::Cpu);
    assert_eq!(cos.size(), vec![16, 32]);
    assert_eq!(sin.size(), vec![16, 32]);
}

#[test]
fn test_apply_rotary_emb_preserves_shapes() {
    let batch = 2;
    let _heads = 4;
    let seq_len = 8;
    let head_dim = 64;
    // apply_rotary_emb operates on per-head tensors: [B, T, head_dim]
    let q = Tensor::randn([batch, seq_len, head_dim], (Kind::Float, Device::Cpu));
    let k = Tensor::randn([batch, seq_len, head_dim], (Kind::Float, Device::Cpu));
    let (cos, sin) = precompute_freqs_cis(head_dim, seq_len, 10_000.0, RopeType::Interleaved, Device::Cpu);
    let (q_rot, k_rot) = apply_rotary_emb(&q, &k, &cos, &sin, RopeType::Interleaved);
    assert_eq!(q_rot.size(), q.size());
    assert_eq!(k_rot.size(), k.size());
}

#[test]
fn test_apply_rotary_emb_split() {
    let batch = 1;
    let _heads = 2;
    let seq_len = 4;
    let head_dim = 32;
    let q = Tensor::randn([batch, seq_len, head_dim], (Kind::Float, Device::Cpu));
    let k = Tensor::randn([batch, seq_len, head_dim], (Kind::Float, Device::Cpu));
    let (cos, sin) = precompute_freqs_cis(head_dim, seq_len, 10_000.0, RopeType::Split, Device::Cpu);
    let (q_rot, k_rot) = apply_rotary_emb(&q, &k, &cos, &sin, RopeType::Split);
    assert_eq!(q_rot.size(), q.size());
    assert_eq!(k_rot.size(), k.size());
}

#[test]
fn test_rope_preserves_norm() {
    let seq_len = 8;
    let head_dim = 32;
    let q = Tensor::randn([1, seq_len, head_dim], (Kind::Float, Device::Cpu));
    let (cos, sin) = precompute_freqs_cis(head_dim, seq_len, 10_000.0, RopeType::Interleaved, Device::Cpu);
    let (q_rot, _) = apply_rotary_emb(&q, &q, &cos, &sin, RopeType::Interleaved);
    // RoPE is norm-preserving
    let orig_norm = q.norm();
    let rot_norm = q_rot.norm();
    assert!((orig_norm - rot_norm).abs().double_value(&[]) < 1e-4);
}
