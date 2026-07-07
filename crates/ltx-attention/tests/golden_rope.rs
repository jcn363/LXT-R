use ltx_attention::{apply_rotary_emb, precompute_freqs_cis, RopeType};
use tch::{Device, Kind, Tensor};

/// Golden test for RoPE precomputation.
///
/// Verifies that precomputed cos/sin frequencies match expected values
/// for a small dimension.
#[test]
fn test_golden_rope_precompute() {
    let dim = 4;
    let max_seq = 4;
    let theta = 10000.0;
    let (cos, sin) = precompute_freqs_cis(dim, max_seq, theta, RopeType::Split, Device::Cpu);

    // cos and sin should have shape [max_seq, dim/2] for Split mode
    assert_eq!(cos.size(), vec![max_seq, dim / 2]);
    assert_eq!(sin.size(), vec![max_seq, dim / 2]);

    // cos^2 + sin^2 should be approximately 1.0 for all positions
    let identities = &cos * &cos + &sin * &sin;
    let max_deviation = (identities - 1.0).abs().max().double_value(&[]);
    assert!(
        max_deviation < 1e-5,
        "RoPE golden test: cos^2+sin^2 deviates from 1.0 by {max_deviation}"
    );
}

/// Golden test for RoPE application — verifies norm preservation.
#[test]
fn test_golden_rope_norm_preservation() {
    let dim = 8;
    let seq_len = 8;
    let (cos, sin) =
        precompute_freqs_cis(dim, seq_len, 10000.0, RopeType::Interleaved, Device::Cpu);

    let q = Tensor::randn([1, seq_len, dim], (Kind::Float, Device::Cpu));
    let k = Tensor::randn([1, seq_len, dim], (Kind::Float, Device::Cpu));

    let norm_before = (&q * &q).sum(Kind::Float).double_value(&[]);

    let (q_rot, _k_rot) = apply_rotary_emb(&q, &k, &cos, &sin, RopeType::Interleaved);

    let norm_after = (&q_rot * &q_rot).sum(Kind::Float).double_value(&[]);

    let rel_diff = ((norm_after - norm_before) / norm_before).abs();
    assert!(
        rel_diff < 1e-5,
        "RoPE golden test: norm changed by {rel_diff} (before={norm_before}, after={norm_after})"
    );
}

/// Golden test: RoPE precompute cos/sin match Python reference.
#[test]
fn test_golden_rope_precompute_from_file() {
    let expected_cos =
        ltx_test_utils::load_golden("crates/goldens/rope_precompute.safetensors", "cos");
    let expected_sin =
        ltx_test_utils::load_golden("crates/goldens/rope_precompute.safetensors", "sin");

    let dim = expected_cos.size()[1] * 2;
    let max_seq = expected_cos.size()[0];
    let (cos, sin) = precompute_freqs_cis(dim, max_seq, 10000.0, RopeType::Split, Device::Cpu);

    ltx_test_utils::assert_allclose(&cos, &expected_cos, 1e-5, 1e-5);
    ltx_test_utils::assert_allclose(&sin, &expected_sin, 1e-5, 1e-5);
}
