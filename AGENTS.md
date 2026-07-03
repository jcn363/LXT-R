# AGENTS.md

Planning repo for the Rust rewrite of LTX-2.3 core. `PLAN.md` is the single source of truth for architecture, crate layout, and implementation order.

## Core Principle: SSOT

Every constant, type, function, and algorithm has exactly ONE definition. Violations are compile errors. This is non-negotiable.

- Constants: only in `ltx_types::constants` (e.g. `NORM_EPS`, `FP8_MAX`, `ROPE_THETA`)
- Utilities: only in `ltx_types::utils` (e.g. `to_velocity`, `to_denoised`, `projection_coef`)
- Normalization: only in `ltx_norm` (RMSNorm, PixelNorm, GroupNorm, factory)
- Each shared primitive (attention, conv, resblock, timestep, patchify, fp8) lives in its own crate
- Model crates are thin wrappers — they compose shared primitives, never reimplement them

## Rust-Only Boundary

All model logic is pure Rust. External FFI is isolated behind safe Rust APIs:
- `tch` (PyTorch bindings) — tensor compute backend, used throughout
- `ltx-fp8`, `ltx-loader` — contain CUDA C FFI (`kernels.cu`, `cublas.rs`), require CUDA toolchain
- `safetensors`, `tokenizers` — Hugging Face crates for checkpoint loading and tokenization
- Rule: Rust owns all algorithms; externals provide I/O and compute backends only

## Crate Hierarchy

```
ltx-core (facade)
├── ltx-types          ← shapes, protocols, enums, constants, utils (EVERYTHING depends on this)
├── Shared primitives:
│   ├── ltx-norm       ← RMSNorm, PixelNorm, GroupNorm, factory
│   ├── ltx-attention   ← RoPE, SDPA, TransformerAttention, SimpleAttnBlock, factory
│   ├── ltx-conv       ← CausalConv2d/3d, DualConv3d, factory
│   ├── ltx-resblock   ← ResnetBlock2D/3D, UNetMidBlock3D, factory
│   ├── ltx-timestep   ← sinusoidal, MLP, combined, AdaLN
│   ├── ltx-patchify   ← patchify/unpatchify ops (5d, 4d, audio), tiling, coords
│   └── ltx-fp8        ← quantize, dequantize, cast, cuBLAS FFI
├── Model crates (compose primitives):
│   ├── ltx-transformer, ltx-video-vae, ltx-audio-vae, ltx-upsampler
│   └── ltx-text-encoder (Gemma3 + SigLIP)
└── Infra: ltx-loader, ltx-quantization, ltx-components, ltx-conditioning, ltx-guidance
```

## Implementation Order

Phases P0→P10 in PLAN.md §9. Shared primitives (P1) come before model crates (P2–P8). Do not skip ahead — model crates depend on primitives being complete.

## SSOT Verification Commands

Run before merging anything:

```bash
# No hardcoded constants outside constants.rs
rg "1e-6|1e-8|448\.0|10000\.0" --include="*.rs" --glob="!**/constants.rs" --glob="!**/tests/**"

# No duplicate function definitions
rg "pub fn to_velocity|pub fn to_denoised|pub fn patchify|pub fn unpatchify" --include="*.rs"

# No duplicate type definitions
rg "pub struct RMSNorm|pub struct PixelNorm|pub struct CausalConv3d|pub struct ResnetBlock3D" --include="*.rs"

# All imports use ltx_* paths, never crate:: for shared primitives
rg "use crate::(norm|attention|conv|resblock|patchify|fp8)::" --include="*.rs" --glob="!**/lib.rs"

# Clippy strict
cargo clippy --all-targets -- -D warnings -D clippy::all
```

## Key Conventions

- Import from `ltx_*` crate root, never internal submodules (except explicit re-exports)
- Each shared crate has a `factory.rs` — use it to instantiate modules, never construct directly
- Tests go in each crate's `tests/` dir with golden `.npz` files from Python for numerical comparison
- 19 crates, 106 files, ~7,570 LOC
- Workspace root: `Cargo.toml` at repo root, all crates under `crates/`

## Gotchas

- `ltx-fp8` has CUDA C FFI (`kernels.cu`) — requires CUDA toolchain
- `ltx-loader` also has CUDA kernels for fused ops
- `ltx-text-encoder` is the largest crate (12 files) — Gemma3 has 48 transformer layers, SigLIP has 27
- `ltx-patchify::ops` reimplements `einops.rearrange` patterns in pure Rust — do not add an einops dependency
- RoPE has two variants (`Interleaved`, `Split`) — make sure you use the right one per model
- `ltx-components` bundles 4 concerns (scheduler, guider, noiser, step) — split if any grows beyond ~200 LOC
- tch 0.16 API: `Tensor::to_dtype` takes 3 args (kind, non_blocking, copy), not 1
- tch 0.16 API: `Tensor::sum_dim_intlist` takes 3 args (dims, keepdim, dtype)
- tch 0.16 API: Use `Tensor::arange_start(start, end, options)` for custom ranges
