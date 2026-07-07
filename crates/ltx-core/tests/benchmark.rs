use ltx_attention::{precompute_freqs_cis, RopeType};
use ltx_norm::RMSNorm;
use ltx_patchify::{patchify_5d, unpatchify_5d};
use ltx_timestep::get_timestep_embedding;
use ltx_types::Scheduler;
use std::time::Instant;
use tch::{Device, Kind, Tensor};

fn time_fn<F: FnMut()>(name: &str, f: &mut F, iterations: u32, warmup: u32) {
    for _ in 0..warmup {
        f();
    }
    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    let elapsed = start.elapsed();
    let ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    let ops_per_sec = 1000.0 / ms;
    println!("{:<30} {:<15.4} {:<15.0}", name, ms, ops_per_sec);
}

#[test]
fn benchmark_rust_operations() {
    let iterations = 100;
    let warmup = 10;

    println!(
        "\nBenchmarking Rust LTX operations ({} iterations, {} warmup)\n",
        iterations, warmup
    );
    println!(
        "{:<30} {:<15} {:<15}",
        "Operation", "Time (ms)", "Throughput"
    );
    println!("{}", "-".repeat(60));

    // RMSNorm
    let dim = 64;
    let norm = RMSNorm::new(dim, 1e-6, Device::Cpu);
    let x = Tensor::randn([2, 128, dim], (Kind::Float, Device::Cpu));
    time_fn(
        "RMSNorm (2,128,64)",
        &mut || {
            let _ = norm.forward(&x);
        },
        iterations,
        warmup,
    );

    // Sinusoidal embedding
    let t = Tensor::from_slice(&[0.5f32]);
    time_fn(
        "Sinusoidal embed",
        &mut || {
            let _ = get_timestep_embedding(&t, 64, 10_000);
        },
        iterations,
        warmup,
    );

    // RoPE
    time_fn(
        "RoPE precompute",
        &mut || {
            let _ = precompute_freqs_cis(64, 128, 10000.0, RopeType::Split, Device::Cpu);
        },
        iterations,
        warmup,
    );

    // Patchify 5D roundtrip
    let x = Tensor::randn([1, 4, 8, 32, 32], (Kind::Float, Device::Cpu));
    time_fn(
        "Patchify 5D roundtrip",
        &mut || {
            let p = patchify_5d(&x, 2, 4, 4);
            let _ = unpatchify_5d(&p, 1, 4, 8, 32, 32, 2, 4, 4);
        },
        iterations,
        warmup,
    );

    // FP8 quantize
    let w = Tensor::randn([256, 512], (Kind::Float, Device::Cpu));
    time_fn(
        "FP8 quantize (256x512)",
        &mut || {
            let _ = ltx_fp8::quantize_weight_to_fp8_per_tensor(&w);
        },
        iterations,
        warmup,
    );

    // Scheduler
    time_fn(
        "Scheduler sigmas (n=50)",
        &mut || {
            let sched = ltx_components::Ltx2Scheduler::default();
            let _ = sched.sigmas(50);
        },
        iterations,
        warmup,
    );

    println!();
}
