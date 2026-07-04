#!/usr/bin/env python3
"""Generate golden reference data for LTX-R numerical regression tests.

This script produces .safetensors files containing input/output tensor pairs
for key modules. These files are loaded by Rust tests via ltx_test_utils::load_golden
and compared against the Rust implementation outputs.

Usage:
    python scripts/generate_goldens.py [--output-dir crates/goldens]

Requirements:
    pip install torch safetensors numpy
"""

import argparse
import os
import torch
from safetensors.torch import save_file


def make_output_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


# ── RMSNorm ──────────────────────────────────────────────────────────────

def generate_rms_norm(out_dir: str):
    """Generate golden data for RMSNorm."""
    dim = 64
    eps = 1e-6

    # Simple RMSNorm implementation matching the Rust code
    class RMSNorm(torch.nn.Module):
        def __init__(self, dim, eps):
            super().__init__()
            self.weight = torch.ones(dim)
            self.eps = eps

        def forward(self, x):
            x_f32 = x.float()
            rms = x_f32.pow(2).mean(dim=-1, keepdim=True)
            return (x_f32 / (rms + self.eps).sqrt()).to(x.dtype) * self.weight

    norm = RMSNorm(dim, eps)

    # Test 1: All ones
    x1 = torch.ones(1, 8, dim)
    out1 = norm(x1)
    save_file({"input": x1, "output": out1}, os.path.join(out_dir, "rms_norm_ones.safetensors"))

    # Test 2: Non-trivial input
    x2 = torch.randn(1, 8, dim)
    out2 = norm(x2)
    save_file({"input": x2, "output": out2}, os.path.join(out_dir, "rms_norm_nontrivial.safetensors"))

    print(f"  Generated rms_norm_ones.safetensors, rms_norm_nontrivial.safetensors")


# ── Sinusoidal Embedding ─────────────────────────────────────────────────

def generate_sinusoidal(out_dir: str):
    """Generate golden data for sinusoidal timestep embedding."""
    def get_timestep_embedding(timesteps, dim, max_period=10000):
        half = dim // 2
        freqs = torch.arange(0, half, dtype=torch.float32)
        freqs = -torch.tensor(float(max_period)).log() * freqs / (half - 1)
        freqs = freqs.exp()
        args = timesteps.unsqueeze(1).float() * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=1)

    dim = 64

    # Test 1: Single timestep
    t1 = torch.tensor([0.0])
    out1 = get_timestep_embedding(t1, dim)
    save_file({"input": t1, "output": out1}, os.path.join(out_dir, "sinusoidal_single.safetensors"))

    # Test 2: Batch of timesteps
    t2 = torch.tensor([0.0, 1.0, 10.0, 100.0, 999.0])
    out2 = get_timestep_embedding(t2, dim)
    save_file({"input": t2, "output": out2}, os.path.join(out_dir, "sinusoidal_batch.safetensors"))

    # Test 3: Verify no NaN
    for t_val in [0.0, 1.0, 10.0, 100.0, 999.0]:
        t = torch.tensor([t_val])
        emb = get_timestep_embedding(t, dim)
        assert not torch.isnan(emb).any(), f"NaN at timestep={t_val}"
        assert not torch.isinf(emb).any(), f"Inf at timestep={t_val}"

    print(f"  Generated sinusoidal_single.safetensors, sinusoidal_batch.safetensors")


# ── RoPE ─────────────────────────────────────────────────────────────────

def generate_rope(out_dir: str):
    """Generate golden data for Rotary Position Embeddings."""
    import math

    def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).unsqueeze(1).float()
        # Match Rust: multiply by ROPE_FREQ_SCALE (pi/2)
        freqs = t * freqs.unsqueeze(0) * (math.pi / 2)
        return torch.cos(freqs), torch.sin(freqs)

    dim = 8
    max_seq = 8

    cos, sin = precompute_freqs_cis(dim, max_seq)

    # Verify cos^2 + sin^2 = 1
    identities = cos ** 2 + sin ** 2
    assert torch.allclose(identities, torch.ones_like(identities), atol=1e-5)

    save_file(
        {"cos": cos, "sin": sin},
        os.path.join(out_dir, "rope_precompute.safetensors")
    )

    print(f"  Generated rope_precompute.safetensors")


# ── Patchify/Unpatchify ──────────────────────────────────────────────────

def generate_patchify(out_dir: str):
    """Generate golden data for patchify/unpatchify operations."""
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

    # Test: patchify then unpatchify should recover original
    x = torch.randn(1, 4, 4, 16, 16)
    p1, p2, p3 = 2, 4, 4
    patched = patchify_5d(x, p1, p2, p3)
    recovered = unpatchify_5d(patched, 1, 4, 4, 16, 16, p1, p2, p3)

    save_file(
        {"input": x, "patched": patched, "recovered": recovered},
        os.path.join(out_dir, "patchify_5d.safetensors")
    )

    # Verify roundtrip
    assert torch.allclose(x, recovered, atol=1e-6)

    print(f"  Generated patchify_5d.safetensors")


# ── FP8 Quantize ─────────────────────────────────────────────────────────

def generate_fp8(out_dir: str):
    """Generate golden data for FP8 quantization."""
    FP8_MAX = 448.0

    def quantize_weight_to_fp8_per_tensor(weight):
        f32 = weight.float()
        max_abs = f32.abs().amax().clamp(min=1e-8)
        scale = torch.tensor(FP8_MAX) / max_abs
        q = (f32 * scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        return q, (1.0 / scale)

    # Test 1: Normal weights
    w1 = torch.randn(16, 32)
    q1, inv_scale1 = quantize_weight_to_fp8_per_tensor(w1)
    q1_f32 = q1.to(torch.float32)
    save_file(
        {"weight": w1, "quantized": q1_f32, "inv_scale": inv_scale1.unsqueeze(0)},
        os.path.join(out_dir, "fp8_quantize_normal.safetensors")
    )

    # Test 2: Extreme values
    w2 = torch.tensor([1e6, -1e6, 1e-6, -1e-6])
    q2, inv_scale2 = quantize_weight_to_fp8_per_tensor(w2)
    q2_f32 = q2.to(torch.float32)
    save_file(
        {"weight": w2, "quantized": q2_f32, "inv_scale": inv_scale2.unsqueeze(0)},
        os.path.join(out_dir, "fp8_quantize_extreme.safetensors")
    )

    # Verify quantization error
    recovered = q1_f32 * inv_scale1
    max_err = (w1 - recovered).abs().max().item()
    assert max_err < 1.0, f"FP8 quantization error too large: {max_err}"

    print(f"  Generated fp8_quantize_normal.safetensors, fp8_quantize_extreme.safetensors")


# ── Scheduler ────────────────────────────────────────────────────────────

def generate_scheduler(out_dir: str):
    """Generate golden data for scheduler sigma schedules."""
    import math

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

    # Test: various n_steps
    for n in [0, 1, 5, 10, 50]:
        sigmas = ltx2_sigmas(n)
        t = torch.tensor(sigmas)
        save_file(
            {"sigmas": t},
            os.path.join(out_dir, f"scheduler_n{n}.safetensors")
        )

    print(f"  Generated scheduler_n*.safetensors")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate golden reference data")
    parser.add_argument("--output-dir", default="crates/goldens",
                        help="Output directory for .safetensors files")
    args = parser.parse_args()

    out_dir = make_output_dir(args.output_dir)
    print(f"Generating golden data in {out_dir}/")

    torch.manual_seed(42)  # Reproducible

    generate_rms_norm(out_dir)
    generate_sinusoidal(out_dir)
    generate_rope(out_dir)
    generate_patchify(out_dir)
    generate_fp8(out_dir)
    generate_scheduler(out_dir)

    print(f"\nDone! Generated golden files in {out_dir}/")


if __name__ == "__main__":
    main()
