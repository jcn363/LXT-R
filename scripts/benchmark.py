#!/usr/bin/env python3
"""Benchmark Python implementation of LTX-2.3 operations.

Measures execution time for key operations to compare against Rust.
Uses warmup + multiple iterations for stable measurements.

Usage:
    python3 scripts/benchmark.py [--iterations 100] [--warmup 10]
"""

import argparse
import time
import torch
import math


def time_fn(fn, iterations, warmup):
    """Time a function with warmup and multiple iterations."""
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    elapsed = time.perf_counter() - start
    return elapsed / iterations * 1000  # ms per iteration


# ── RMSNorm ──────────────────────────────────────────────────────────────

class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = torch.ones(dim)
        self.eps = eps

    def forward(self, x):
        x_f32 = x.float()
        rms = x_f32.pow(2).mean(dim=-1, keepdim=True)
        return (x_f32 / (rms + self.eps).sqrt()).to(x.dtype) * self.weight


def bench_rms_norm(iterations, warmup):
    dim = 64
    norm = RMSNorm(dim).to(torch.float32)
    x = torch.randn(2, 128, dim)
    return time_fn(lambda: norm(x), iterations, warmup)


# ── Sinusoidal Embedding ─────────────────────────────────────────────────

def get_timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.arange(0, half, dtype=torch.float32)
    freqs = -torch.tensor(float(max_period)).log() * freqs / (half - 1)
    freqs = freqs.exp()
    args = timesteps.unsqueeze(1).float() * freqs.unsqueeze(0)
    return torch.cat([args.sin(), args.cos()], dim=1)


def bench_sinusoidal(iterations, warmup):
    dim = 64
    t = torch.tensor([0.5])
    return time_fn(lambda: get_timestep_embedding(t, dim), iterations, warmup)


# ── RoPE ─────────────────────────────────────────────────────────────────

def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).unsqueeze(1).float()
    freqs = t * freqs.unsqueeze(0) * (math.pi / 2)
    return torch.cos(freqs), torch.sin(freqs)


def bench_rope(iterations, warmup):
    dim = 64
    max_seq = 128
    cos, sin = precompute_freqs_cis(dim, max_seq)
    # cos/sin have shape [max_seq, dim//2] for Split mode
    # Need to repeat to match dim
    cos_full = cos.repeat_interleave(2, dim=1)
    sin_full = sin.repeat_interleave(2, dim=1)
    q = torch.randn(1, max_seq, dim)
    return time_fn(lambda: (q * cos_full) + (q * sin_full), iterations, warmup)


# ── Patchify 5D ──────────────────────────────────────────────────────────

def patchify_5d(x, p1, p2, p3):
    b, c, f, h, w = x.shape
    return x.reshape(b, c, f // p1, p1, h // p2, p2, w // p3, p3) \
             .permute(0, 2, 4, 6, 1, 3, 5, 7) \
             .reshape(b, (f // p1) * (h // p2) * (w // p3), c * p1 * p2 * p3)


def unpatchify_5d(x, b, c, f, h, w, p1, p2, p3):
    fp, hp, wp = f // p1, h // p2, w // p3
    return x.reshape(b, fp, hp, wp, c, p1, p2, p3) \
             .permute(0, 4, 1, 5, 2, 6, 3, 7) \
             .reshape(b, c, f, h, w)


def bench_patchify(iterations, warmup):
    x = torch.randn(1, 4, 8, 32, 32)
    p1, p2, p3 = 2, 4, 4
    return time_fn(lambda: unpatchify_5d(patchify_5d(x, p1, p2, p3), 1, 4, 8, 32, 32, p1, p2, p3), iterations, warmup)


# ── FP8 Quantize ─────────────────────────────────────────────────────────

FP8_MAX = 448.0

def quantize_weight_to_fp8_per_tensor(weight):
    f32 = weight.float()
    max_abs = f32.abs().amax().clamp(min=1e-8)
    scale = torch.tensor(FP8_MAX) / max_abs
    q = (f32 * scale).clamp(-FP8_MAX, FP8_MAX)
    return q, (1.0 / scale)


def bench_fp8_quantize(iterations, warmup):
    w = torch.randn(256, 512)
    return time_fn(lambda: quantize_weight_to_fp8_per_tensor(w), iterations, warmup)


# ── Scheduler ────────────────────────────────────────────────────────────

def bench_scheduler(iterations, warmup):
    def ltx2_sigmas(n_steps, max_shift=2.05, base_shift=0.95, terminal=0.1):
        if n_steps == 0:
            return [1.0]
        n = float(n_steps)
        sigmas = []
        for i in range(n_steps + 1):
            t = i / n
            shifted = t * max_shift + base_shift
            sigma = terminal + (1.0 - terminal) / (1.0 + math.exp(shifted - base_shift - max_shift / 2.0))
            sigmas.append(max(0.0, min(1.0, sigma)))
        return sigmas
    return time_fn(lambda: ltx2_sigmas(50), iterations, warmup)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark Python LTX operations")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    args = parser.parse_args()

    torch.manual_seed(42)

    print(f"Benchmarking with {args.iterations} iterations, {args.warmup} warmup\n")
    print(f"{'Operation':<30} {'Time (ms)':<15} {'Throughput':<15}")
    print("-" * 60)

    benchmarks = [
        ("RMSNorm (2,128,64)", lambda: bench_rms_norm(args.iterations, args.warmup)),
        ("Sinusoidal embed", lambda: bench_sinusoidal(args.iterations, args.warmup)),
        ("RoPE precompute", lambda: bench_rope(args.iterations, args.warmup)),
        ("Patchify 5D roundtrip", lambda: bench_patchify(args.iterations, args.warmup)),
        ("FP8 quantize (256x512)", lambda: bench_fp8_quantize(args.iterations, args.warmup)),
        ("Scheduler sigmas (n=50)", lambda: bench_scheduler(args.iterations, args.warmup)),
    ]

    for name, fn in benchmarks:
        ms = fn()
        ops_per_sec = 1000.0 / ms if ms > 0 else float('inf')
        print(f"{name:<30} {ms:<15.4f} {ops_per_sec:<15.0f}")

    print()


if __name__ == "__main__":
    main()
