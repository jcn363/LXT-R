use criterion::{criterion_group, criterion_main, Criterion};
use ltx_attention::{precompute_freqs_cis, apply_rotary_emb, RopeType, scaled_dot_product_attention};
use ltx_norm::RMSNorm;
use ltx_patchify::{patchify_5d, unpatchify_5d};
use ltx_timestep::get_timestep_embedding;
use tch::Tensor;

fn bench_patchify_5d(c: &mut Criterion) {
    let x = Tensor::randn([1, 4, 8, 32, 32], (tch::Kind::Float, tch::Device::Cpu));
    c.bench_function("patchify_5d 1x4x8x32x32", |b| {
        b.iter(|| patchify_5d(&x, 2, 4, 4))
    });
}

fn bench_unpatchify_5d(c: &mut Criterion) {
    let x = Tensor::randn([1, 256, 128], (tch::Kind::Float, tch::Device::Cpu));
    c.bench_function("unpatchify_5d 1x256x128", |b| {
        b.iter(|| unpatchify_5d(&x, 1, 4, 8, 32, 32, 2, 4, 4))
    });
}

fn bench_rms_norm(c: &mut Criterion) {
    let norm = RMSNorm::new(512, 1e-6, tch::Device::Cpu);
    let x = Tensor::randn([1, 128, 512], (tch::Kind::Float, tch::Device::Cpu));
    c.bench_function("rms_norm 512d", |b| b.iter(|| norm.forward(&x)));
}

fn bench_rope(c: &mut Criterion) {
    let mut group = c.benchmark_group("rope");
    let seq_len = 128i64;
    for &rope_type in &[RopeType::Interleaved, RopeType::Split] {
        let (cos_full, sin_full) = precompute_freqs_cis(128, seq_len, 10000.0, rope_type, tch::Device::Cpu);
        let q = Tensor::randn([1, 8, seq_len, 128], (tch::Kind::Float, tch::Device::Cpu));
        let k = Tensor::randn([1, 8, seq_len, 128], (tch::Kind::Float, tch::Device::Cpu));
        let label = format!("{:?}", rope_type);
        group.bench_function(&label, |b| {
            b.iter(|| apply_rotary_emb(&q, &k, &cos_full, &sin_full, rope_type))
        });
    }
    group.finish();
}

fn bench_sdpa(c: &mut Criterion) {
    let q = Tensor::randn([1, 8, 128, 128], (tch::Kind::Float, tch::Device::Cpu));
    let k = Tensor::randn([1, 8, 128, 128], (tch::Kind::Float, tch::Device::Cpu));
    let v = Tensor::randn([1, 8, 128, 128], (tch::Kind::Float, tch::Device::Cpu));
    c.bench_function("sdpa 8 heads x 128 seq", |b| {
        b.iter(|| scaled_dot_product_attention(&q, &k, &v, None, false))
    });
}

fn bench_timestep_embedding(c: &mut Criterion) {
    let timesteps = Tensor::from_slice(&[0.5f32]);
    c.bench_function("sinusoidal_embedding dim=256", |b| {
        b.iter(|| get_timestep_embedding(&timesteps, 256, 10000))
    });
}

criterion_group!(
    benches,
    bench_patchify_5d,
    bench_unpatchify_5d,
    bench_rms_norm,
    bench_rope,
    bench_sdpa,
    bench_timestep_embedding,
);
criterion_main!(benches);
