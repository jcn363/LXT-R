# LXT-R

Rust rewrite of [LTX-2.3](https://github.com/LightricksResearch/LTX-Video) core — a modular, DRY, SSOT-enforced workspace for video/audio generative models.

## Architecture

19 crates, ~7,500 LOC. All model logic is pure Rust; external FFI (`tch`, CUDA) is isolated behind safe APIs.

```
ltx-core (facade)
├── ltx-types          ← constants, shapes, protocols, enums, utils
├── Shared primitives (SSOT — one implementation per primitive):
│   ├── ltx-norm       ← RMSNorm, PixelNorm, GroupNorm
│   ├── ltx-attention   ← RoPE, SDPA, TransformerAttention
│   ├── ltx-conv       ← CausalConv2d/3d, DualConv3d
│   ├── ltx-resblock   ← ResnetBlock2D/3D, UNetMidBlock3D
│   ├── ltx-timestep   ← sinusoidal, MLP, AdaLN
│   ├── ltx-patchify   ← patchify/unpatchify ops
│   └── ltx-fp8        ← FP8 quantize/dequantize
├── Model crates:
│   ├── ltx-transformer, ltx-video-vae, ltx-audio-vae, ltx-upsampler
│   └── ltx-text-encoder (Gemma3 + SigLIP)
└── Infrastructure:
    ├── ltx-loader, ltx-quantization
    ├── ltx-components, ltx-conditioning, ltx-guidance
    └── ltx-core (public API)
```

## Prerequisites

- Rust 1.75+
- CUDA toolkit (for `ltx-fp8` and `ltx-loader` FFI)
- Python 3.10+ (for generating golden test data)

## Build

```bash
# Full workspace
cargo build

# Single crate
cargo build -p ltx-transformer

# With CUDA support (requires CUDA toolkit)
cargo build --features cuda
```

## Test

```bash
# All tests
cargo test

# Single crate
cargo test -p ltx-norm

# With output
cargo test -- --nocapture
```

## Lint

```bash
cargo clippy --all-targets
```

## SSOT Enforcement

Every constant, type, and function has exactly ONE definition. Violations are caught at compile time.

```bash
# No hardcoded constants outside constants.rs
rg "1e-6|1e-8|448\.0|10000\.0" --include="*.rs" --glob="!**/constants.rs" --glob="!**/tests/**"

# No duplicate function definitions
rg "pub fn to_velocity|pub fn to_denoised|pub fn patchify|pub fn unpatchify" --include="*.rs"

# No duplicate type definitions
rg "pub struct RMSNorm|pub struct PixelNorm|pub struct CausalConv3d|pub struct ResnetBlock3D" --include="*.rs"

# All imports use ltx_* paths
rg "use crate::(norm|attention|conv|resblock|patchify|fp8)::" --include="*.rs" --glob="!**/lib.rs"
```

## Project Structure

```
Cargo.toml              # workspace root
PLAN.md                 # full architecture spec
AGENTS.md               # agent instructions
crates/
├── ltx-types/          # Foundation: constants, shapes, protocols
├── ltx-norm/           # Normalization (SSOT)
├── ltx-attention/      # Attention (SSOT)
├── ltx-conv/           # Convolution (SSOT)
├── ltx-resblock/       # Residual blocks (SSOT)
├── ltx-timestep/       # Timestep embeddings (SSOT)
├── ltx-patchify/       # Patchification (SSOT)
├── ltx-fp8/            # FP8 operations (SSOT)
├── ltx-components/     # Diffusion pipeline components
├── ltx-conditioning/   # Conditioning items and masks
├── ltx-guidance/       # Perturbation configs
├── ltx-transformer/    # DiT transformer model
├── ltx-video-vae/      # Video VAE
├── ltx-audio-vae/      # Audio VAE
├── ltx-upsampler/      # Latent upsampling
├── ltx-text-encoder/   # Gemma3 + SigLIP
├── ltx-loader/         # Checkpoint loading
├── ltx-quantization/   # FP8 quantization policy
└── ltx-core/           # Public API facade
```

## License

MIT
