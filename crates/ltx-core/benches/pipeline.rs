use criterion::{criterion_group, criterion_main, Criterion};
use ltx_attention::{
    apply_rotary_emb, precompute_freqs_cis, scaled_dot_product_attention, RopeType,
};
use ltx_norm::RMSNorm;
use ltx_patchify::{patchify_4d, patchify_5d, unpatchify_4d, unpatchify_5d};
use ltx_timestep::get_timestep_embedding;
use tch::Tensor;

fn bench_patchify_5d(c: &mut Criterion) {
    let mut group = c.benchmark_group("patchify_5d");
    // Small: 1x4x4x16x16 (512 patches)
    let x = Tensor::randn([1, 4, 4, 16, 16], (tch::Kind::Float, tch::Device::Cpu));
    group.bench_function("small 4x16x16", |b| b.iter(|| patchify_5d(&x, 2, 4, 4)));
    // Medium: 1x4x8x32x32 (2048 patches)
    let x = Tensor::randn([1, 4, 8, 32, 32], (tch::Kind::Float, tch::Device::Cpu));
    group.bench_function("medium 8x32x32", |b| b.iter(|| patchify_5d(&x, 2, 4, 4)));
    group.finish();
}

fn bench_unpatchify_5d(c: &mut Criterion) {
    let mut group = c.benchmark_group("unpatchify_5d");
    let x = Tensor::randn([1, 512, 128], (tch::Kind::Float, tch::Device::Cpu));
    group.bench_function("small 512x128", |b| {
        b.iter(|| unpatchify_5d(&x, 1, 4, 4, 16, 16, 2, 4, 4))
    });
    let x = Tensor::randn([1, 2048, 128], (tch::Kind::Float, tch::Device::Cpu));
    group.bench_function("medium 2048x128", |b| {
        b.iter(|| unpatchify_5d(&x, 1, 4, 8, 32, 32, 2, 4, 4))
    });
    group.finish();
}

fn bench_patchify_4d(c: &mut Criterion) {
    let mut group = c.benchmark_group("patchify_4d");
    let x = Tensor::randn([1, 3, 64, 64], (tch::Kind::Float, tch::Device::Cpu));
    group.bench_function("3ch 64x64", |b| b.iter(|| patchify_4d(&x, 4)));
    let x = Tensor::randn([1, 16, 32, 32], (tch::Kind::Float, tch::Device::Cpu));
    group.bench_function("16ch 32x32", |b| b.iter(|| patchify_4d(&x, 2)));
    group.finish();
}

fn bench_unpatchify_4d(c: &mut Criterion) {
    let mut group = c.benchmark_group("unpatchify_4d");
    // patchify_4d([1,3,64,64], p=4) → [1, 48, 16, 16]
    let x = Tensor::randn([1, 48, 16, 16], (tch::Kind::Float, tch::Device::Cpu));
    group.bench_function("48ch 16x16", |b| {
        b.iter(|| unpatchify_4d(&x, 1, 3, 64, 64, 4))
    });
    group.finish();
}

fn bench_rms_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("rms_norm");
    for &dim in &[128i64, 512, 2048] {
        let norm = RMSNorm::new(dim, 1e-6, tch::Device::Cpu);
        let x = Tensor::randn([1, 128, dim], (tch::Kind::Float, tch::Device::Cpu));
        let label = format!("dim={dim}");
        group.bench_function(&label, |b| b.iter(|| norm.forward(&x)));
    }
    group.finish();
}

fn bench_rope(c: &mut Criterion) {
    let mut group = c.benchmark_group("rope");
    let seq_len = 128i64;
    for &rope_type in &[RopeType::Interleaved, RopeType::Split] {
        let (cos_full, sin_full) =
            precompute_freqs_cis(128, seq_len, 10000.0, rope_type, tch::Device::Cpu);
        let q = Tensor::randn([1, 8, seq_len, 128], (tch::Kind::Float, tch::Device::Cpu));
        let k = Tensor::randn([1, 8, seq_len, 128], (tch::Kind::Float, tch::Device::Cpu));
        let label = format!("{:?}", rope_type);
        group.bench_function(&label, |b| {
            b.iter(|| apply_rotary_emb(&q, &k, &cos_full, &sin_full, rope_type))
        });
    }
    // Long sequence: 512 tokens
    let (cos_full, sin_full) =
        precompute_freqs_cis(128, 512, 10000.0, RopeType::Interleaved, tch::Device::Cpu);
    let q = Tensor::randn([1, 8, 512, 128], (tch::Kind::Float, tch::Device::Cpu));
    let k = Tensor::randn([1, 8, 512, 128], (tch::Kind::Float, tch::Device::Cpu));
    group.bench_function("Interleaved 512seq", |b| {
        b.iter(|| apply_rotary_emb(&q, &k, &cos_full, &sin_full, RopeType::Interleaved))
    });
    group.finish();
}

fn bench_sdpa(c: &mut Criterion) {
    let mut group = c.benchmark_group("sdpa");
    let q = Tensor::randn([1, 8, 128, 128], (tch::Kind::Float, tch::Device::Cpu));
    let k = Tensor::randn([1, 8, 128, 128], (tch::Kind::Float, tch::Device::Cpu));
    let v = Tensor::randn([1, 8, 128, 128], (tch::Kind::Float, tch::Device::Cpu));
    group.bench_function("8 heads x 128 seq", |b| {
        b.iter(|| scaled_dot_product_attention(&q, &k, &v, None, false))
    });
    // Larger: 16 heads x 256 seq
    let q = Tensor::randn([1, 16, 256, 128], (tch::Kind::Float, tch::Device::Cpu));
    let k = Tensor::randn([1, 16, 256, 128], (tch::Kind::Float, tch::Device::Cpu));
    let v = Tensor::randn([1, 16, 256, 128], (tch::Kind::Float, tch::Device::Cpu));
    group.bench_function("16 heads x 256 seq", |b| {
        b.iter(|| scaled_dot_product_attention(&q, &k, &v, None, false))
    });
    // With causal mask
    let q = Tensor::randn([1, 8, 128, 128], (tch::Kind::Float, tch::Device::Cpu));
    let k = Tensor::randn([1, 8, 128, 128], (tch::Kind::Float, tch::Device::Cpu));
    let v = Tensor::randn([1, 8, 128, 128], (tch::Kind::Float, tch::Device::Cpu));
    let mask = Tensor::ones([128, 128], (tch::Kind::Bool, tch::Device::Cpu)).tril(0);
    group.bench_function("8 heads x 128 seq causal", |b| {
        b.iter(|| scaled_dot_product_attention(&q, &k, &v, Some(&mask), false))
    });
    group.finish();
}

fn bench_timestep_embedding(c: &mut Criterion) {
    let mut group = c.benchmark_group("sinusoidal_embedding");
    let timesteps = Tensor::from_slice(&[0.5f32]);
    for &dim in &[128i64, 256, 512] {
        let label = format!("dim={dim}");
        group.bench_function(&label, |b| {
            b.iter(|| get_timestep_embedding(&timesteps, dim, 10000))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_patchify_5d,
    bench_unpatchify_5d,
    bench_patchify_4d,
    bench_unpatchify_4d,
    bench_rms_norm,
    bench_rope,
    bench_sdpa,
    bench_timestep_embedding,
);
criterion_main!(benches);
