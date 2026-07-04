# LTX-R

Rust rewrite of [LTX-2.3](https://github.com/LightricksResearch/LTX-Video) core — a modular, DRY, SSOT-enforced workspace for video/audio generative models.

## Architecture

20 crates, ~11,700 LOC (110 source files + 49 test files). All model logic is pure Rust; external FFI (`tch`, CUDA) is isolated behind safe APIs.

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
├── Infrastructure:
│   ├── ltx-loader, ltx-quantization
│   ├── ltx-components, ltx-conditioning, ltx-guidance
│   └── ltx-core (public API)
└── Testing:
    └── ltx-test-utils (golden file loading, assertions, fixtures)
```

## Quick Start

### Build

```bash
cargo build --workspace
```

### Run Inference Demo

```bash
# With random weights (demo mode)
cargo run --bin ltx-inference -- --steps 4

# Run with real weights (16x16, 20 steps)
cargo run --release --bin ltx-inference -- --weights weights_rust_no_te.safetensors --steps 20

# Run with custom resolution
cargo run --release --bin ltx-inference -- --weights weights_rust_no_te.safetensors --steps 8 --height 32 --width 32

# Run with prompt
cargo run --release --bin ltx-inference -- --weights weights_rust_no_te.safetensors --prompt "a sunset over mountains"
```

### Run Tests

```bash
cargo test --workspace  # 384 tests
```

## Weight Conversion

Weight conversion from HuggingFace LTX-Video checkpoints to Rust-compatible safetensors format:

```bash
# Download from HuggingFace
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Lightricks/LTX-Video', 'ltxv-2b-0.9.8-distilled.safetensors', local_dir='weights')
"

# Convert to Rust format (handles key remapping, adaln duplication)
python3 scripts/convert_ltx_weights.py --input weights/ltxv-2b-0.9.8-distilled.safetensors --output weights_rust.safetensors
```

See `scripts/convert_ltx_weights.py` for key remapping details.

## Prerequisites

- Rust 1.75+
- CUDA toolkit (for `ltx-fp8` and `ltx-loader` FFI)
- Python 3.10+ (for golden test data generation and weight conversion)

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
scripts/
├── convert_ltx_weights.py  # Convert LTX-Video weights to Rust format
├── convert_weights.py      # Generic PyTorch weight conversion
├── generate_goldens.py     # Generate golden reference data
└── benchmark.py            # Python benchmarks for comparison
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
├── ltx-test-utils/     # Golden file loading, assertions
├── ltx-core/           # Public API facade + inference binary
└── goldens/            # Golden reference data (.safetensors)
```

## Remaining Work

### Completed
- All 20 crates implemented (~11,700 LOC)
- 384 tests passing
- Inference pipeline loads 929/929 tensors from LTX-Video 2B model
- 28-layer transformer runs with real weights
- Video frames generated (16x16 to 32x32 RGB)
- SentencePiece tokenizer integrated
- Configurable resolution and denoising steps via CLI

### Known Limitations
- **VAE decoder** — Decoder architecture mismatch with Python model (7 up_blocks vs 4 in Rust). Needs architecture alignment.
- **Text encoder** — Gemma3 tokenizer infrastructure complete. Full encoder not yet wired due to memory constraints.
- **Resolution** — 32x32 works with 8 steps on 32GB RAM. Higher resolutions require GPU or model sharding.
- **Denoising steps** — 20 steps at 16x16, 8 steps at 32x32 on 32GB RAM. More steps possible on GPU.

### Not Yet Implemented
1. **VAE decoder alignment** — Align Rust decoder architecture with Python model's 7-block structure
2. **Text encoder integration** — Wire Gemma3 encoder for prompt conditioning
3. **GPU support** — CUDA acceleration for higher resolution and more steps
4. **Model sharding** — Split model across CPU/GPU for larger resolutions
5. **Audio pipeline** — Audio VAE + transformer for audio generation

## License

MIT
