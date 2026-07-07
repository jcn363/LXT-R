# LTX-2.3 Benchmark Results

Benchmark date: 2026-07-07
Binary: `ltx-benchmark` (release mode, libtorch 2.3.0+cu121)
Supported backends: CUDA (NVIDIA), ROCm (AMD), MPS (Apple Metal), CPU

## Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA GeForce GTX 1050 |
| VRAM | 2048 MB (2 GB) |
| CUDA | 12.1 (driver 535.309.01) |
| CPU | (varies by run) |

## Model Configurations

| Config | Dim | Layers | Params | Weight Size (FP16) |
|--------|-----|--------|--------|---------------------|
| Synthetic (benchmark default) | 128 | 2 | ~0.5M | 3.4 MB |
| Real LTX-2.3 | 2048 | 28 | ~1.4B | 2820 MB |

Patchification: p1=2, p2=4, p3=4 (temporal×height×width)

---

## 1. Resolution Scaling

4 frames, 20 steps, 5 iterations (2 warmup).

### Synthetic Model (128d, 2 layers)

| Resolution | Latent Size | CPU FPS | GPU FPS | GPU Speedup |
|------------|-------------|---------|---------|-------------|
| 512×512 | 16×16 | 10.4 | 268.8 | 25.8× |
| 768×768 | 24×24 | 10.6 | 256.0 | 24.1× |
| 1024×1024 | 32×32 | 10.9 | 255.9 | 23.5× |

Resolution scaling is nearly flat at this model size — transformer ops don't dominate with only 2 layers.

### GPU Per-Step Latency

| Resolution | Per-Step |
|------------|----------|
| 512×512 | 14.9 ms |
| 768×768 | 15.6 ms |
| 1024×1024 | 15.6 ms |

---

## 2. Frame Scaling

16×16 latent, 10 steps, 5 iterations (2 warmup).

| Frames | CPU FPS | GPU FPS | GPU Speedup |
|--------|---------|---------|-------------|
| 2 | 41.7 | 136.8 | 3.3× |
| 4 | 71.9 | 257.5 | 3.6× |
| 8 | 73.0 | 487.3 | 6.7× |
| 16 | 210.9 | 1046.5 | 5.0× |

GPU parallelism scales well with frame count — 16 frames achieves 1,046 FPS throughput.

---

## 3. Step Method Comparison

16×16 latent, 4 frames, 10 steps, 8 iterations (3 warmup).

| Method | CPU FPS | GPU FPS |
|--------|---------|---------|
| Euler | 12.0 | 258.9 |
| Res2s | 12.4 | 253.6 |

~3% difference — step method choice is negligible for performance.

---

## 4. Guider Comparison

16×16 latent, 4 frames, 10 steps, 8 iterations (3 warmup).

| Guider | CPU FPS | GPU FPS | Notes |
|--------|---------|---------|-------|
| CFG | 148.1 | 250.9 | 2 model forward passes |
| APG | 41.3 | 265.6 | 2 forward passes + projection math |
| STG | 131.4 | 286.8 | 2 forward passes, simple blend |

APG is 3.5× slower on CPU due to `projection_coef` dot product. On GPU the gap narrows because tensor ops are parallelized.

---

## 5. VRAM Analysis

### Synthetic Model (2 layers)

| Resolution | VRAM Used | VRAM Total | Utilization |
|------------|-----------|------------|-------------|
| 512×512 | 526 MB | 2048 MB | 26% |
| 768×768 | 582 MB | 2048 MB | 28% |
| 1024×1024 | 660 MB | 2048 MB | 32% |

### Real 28-Layer Model — Quantization Comparison

1.41B parameters. GTX 1050 has 2048 MB VRAM.

#### 512×512 (16×16 latent, 4 frames)

| Dtype | Weights | Activations | Total VRAM | Fits? |
|-------|---------|-------------|------------|-------|
| FP32 | 5640 MB | 162 MB | 5802 MB | **No** |
| FP16 | 2820 MB | 162 MB | 2982 MB | **No** |
| INT8 | 1410 MB | 162 MB | 1572 MB | **Yes** (77%) |
| INT4 | 705 MB | 162 MB | 867 MB | **Yes** (42%) |

#### 768×768 (24×24 latent, 4 frames)

| Dtype | Weights | Activations | Total VRAM | Fits? |
|-------|---------|-------------|------------|-------|
| FP32 | 5640 MB | 363 MB | 6004 MB | **No** |
| FP16 | 2820 MB | 363 MB | 3183 MB | **No** |
| INT8 | 1410 MB | 363 MB | 1773 MB | **Yes** (87%) |
| INT4 | 705 MB | 363 MB | 1068 MB | **Yes** (52%) |

#### 1024×1024 (32×32 latent, 4 frames)

| Dtype | Weights | Activations | Total VRAM | Fits? |
|-------|---------|-------------|------------|-------|
| FP32 | 5640 MB | 646 MB | 6286 MB | **No** |
| FP16 | 2820 MB | 646 MB | 3466 MB | **No** |
| INT8 | 1410 MB | 646 MB | 2056 MB | **No** (100.4%) |
| INT4 | 705 MB | 646 MB | 1351 MB | **Yes** (66%) |

### Verdict: Can It Fit on GTX 1050?

| Resolution | INT8 | INT4 | FP16 | FP32 |
|------------|------|------|------|------|
| 512×512 | **Yes** (77%) | **Yes** (42%) | No | No |
| 768×768 | **Yes** (87%) | **Yes** (52%) | No | No |
| 1024×1024 | No (100.4%) | **Yes** (66%) | No | No |

**INT8 fits at 512×512 and 768×768.** Misses by 8 MB at 1024×1024.
**INT4 fits at all resolutions** with 34–66% headroom.

---

## 6. Quantized Inference Benchmarks

Weight-only quantization: weights stored in INT8/INT4, dequantized to FP32 before each matmul.
Implemented in `ltx-quantization` crate: `int8_mm.rs` (per-tensor symmetric) and `int4_mm.rs` (per-group, group_size=128).

### Throughput by Dtype (GPU, 4 frames, 20 steps)

| Resolution | FP32 FPS | FP16 FPS | INT8 FPS | INT4 FPS | INT8 Overhead | INT4 Overhead |
|------------|----------|----------|----------|----------|---------------|---------------|
| 512×512 | 248.1 | 240.6 | 228.2 | 234.9 | 8.0% | 5.3% |
| 768×768 | 240.0 | 225.0 | 238.3 | 233.1 | 0.7% | 2.9% |
| 1024×1024 | 245.5 | 243.1 | 232.2 | 233.5 | 5.4% | 4.9% |

### Per-Step Latency

| Resolution | FP32 | FP16 | INT8 | INT4 |
|------------|------|------|------|------|
| 512×512 | 16.1ms | 16.6ms | 17.5ms | 17.0ms |
| 768×768 | 16.7ms | 17.8ms | 16.8ms | 17.2ms |
| 1024×1024 | 16.3ms | 16.5ms | 17.2ms | 17.1ms |

### Key Findings

- **Quantization overhead is minimal**: INT8 adds 1–8%, INT4 adds 3–5% vs FP32
- **VRAM is identical** across dtypes on the synthetic model (3.4 MB weights are negligible)
- **Real savings appear at scale**: For the 28-layer model, INT8 saves 1410 MB, INT4 saves 2115 MB
- **GTX 1050 lacks INT8 tensor cores** (compute capability 6.1) — dequant runs as FP32 with scalar ops
- **INT4 dequant is slightly faster than INT8** at some resolutions due to simpler unpacking

### Real Model Projection

| Dtype | Weight Size | VRAM @ 512×512 | VRAM @ 1024×1024 | Fits GTX 1050? |
|-------|-------------|-----------------|-------------------|----------------|
| FP32 | 5640 MB | 5802 MB | 6286 MB | No |
| FP16 | 2820 MB | 2982 MB | 3466 MB | No |
| INT8 | 1410 MB | 1572 MB | 2056 MB | Yes (≤768×768) |
| INT4 | 705 MB | 867 MB | 1351 MB | Yes (all) |

### Recommended GPUs by Use Case

| Use Case | Min GPU | Notes |
|----------|---------|-------|
| INT4 inference | GTX 1050 (2 GB) | Fits, ~5% overhead |
| INT8 inference | GTX 1050 (2 GB) | Fits at ≤768×768, ~6% overhead |
| FP16 inference | RTX 3060 (12 GB) | Best quality/speed ratio |
| FP16 training | RTX 3090 (24 GB) | Gradient + optimizer states |
| Full training | A100 (80 GB) | Mixed precision recommended |

---

## Summary

- GPU delivers **24–26× speedup** over CPU on the synthetic model
- Frame parallelism is where GPUs shine: **1,046 FPS** at 16 frames
- Resolution barely matters at small model sizes
- **INT8 fits on GTX 1050 at 512×512 and 768×768** (1572–1773 MB vs 2048 MB)
- **INT4 fits at all resolutions** (867–1351 MB)
- FP16 does not fit (2820 MB weights alone exceed 2 GB)
- **Quantized inference adds only 1–8% overhead** vs FP32 (dequantize-on-the-fly)
- Euler and Res2s step methods perform identically
- CFG/APG/STG guiders have similar GPU performance (APG is CPU-bound due to projection math)
