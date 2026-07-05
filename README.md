# LTX-R

Rust rewrite of [LTX-2.3](https://github.com/LightricksResearch/LTX-Video) core — a modular, DRY, SSOT-enforced workspace for video/audio generative models.

## Architecture

22 crates, ~13,600 LOC (119 source files + 52 test files). All model logic is pure Rust; external FFI (`tch`, CUDA) is isolated behind safe APIs.

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
├── Applications:
│   └── ltx-app (eframe GUI + CLI inference)
└── Testing:
    └── ltx-test-utils (golden file loading, assertions, fixtures)
```

## Quick Start

### Build

```bash
cargo build --workspace
```

### Run Inference (CLI)

```bash
# Random weights (demo)
cargo run --bin ltx-inference -- --steps 4

# Real weights with text encoder (T5 prompt conditioning)
cargo run --release --bin ltx-inference -- \
  --weights weights/ltx-video-2b-v0.9.1-rust.safetensors \
  --tokenizer weights/tokenizer/spiece.model \
  --text-weights weights/text_encoder.safetensors \
  --prompt "a sunset over mountains" \
  --steps 20

# Real weights on GPU
cargo run --release --bin ltx-inference -- \
  --weights weights/ltx-video-2b-v0.9.1-rust.safetensors \
  --device cuda \
  --steps 20

# Custom resolution
cargo run --release --bin ltx-inference -- \
  --weights weights/ltx-video-2b-v0.9.1-rust.safetensors \
  --steps 8 --height 32 --width 32 --frames 8

# Batch processing — multiple prompts from file
printf "a sunset over mountains\na cat walking on grass\na dog running\n" > prompts.txt
cargo run --release --bin ltx-inference -- \
  --weights weights/ltx-video-2b-v0.9.1-rust.safetensors \
  --tokenizer weights/tokenizer/spiece.model \
  --text-weights weights/text_encoder.safetensors \
  --prompts-file prompts.txt \
  --output-dir batch_output \
  --steps 10

# Output structure:
# batch_output/
# ├── 0001/  (frames + gif for prompt 1)
# ├── 0002/  (frames + gif for prompt 2)
# └── 0003/  (frames + gif for prompt 3)
```

### Generate GIF

```bash
# Simple
./generate_gif.sh "a sunset over mountains"

# With options
./generate_gif.sh "a cat walking" --steps 20 --fps 12 --scale 512
./generate_gif.sh "a dog running" --pixel  # pixelated look
```

### Run GUI

```bash
cargo run -p ltx-app
```

The GUI provides:
- Prompt input and model weights picker
- Text encoder weights picker (T5 or Gemma3)
- Resolution, steps, CFG, scheduler, device controls
- Live preview with play/pause and frame scrubber
- Export: PNGs, MP4, or GIF (via toolbar buttons)
- Help tooltips on all controls (hover for descriptions)

### Run Tests

```bash
cargo test --workspace  # 485 tests
```

## Weight Conversion

Download and convert LTX-Video weights:

```bash
# Download from HuggingFace
python3 -c "
from huggingface_hub import hf_hub_download, list_repo_files
import os
os.makedirs('weights', exist_ok=True)
hf_hub_download('Lightricks/LTX-Video', 'ltx-video-2b-v0.9.1.safetensors', local_dir='weights')
hf_hub_download('Lightricks/LTX-Video', 'tokenizer/spiece.model', local_dir='weights')
# Download T5 text encoder (split into 4 shards)
for i in range(1, 5):
    hf_hub_download('Lightricks/LTX-Video', f'text_encoder/model-0000{i}-of-00004.safetensors', local_dir='weights')
"

# Convert transformer weights to Rust format
python3 scripts/convert_ltx_weights.py \
  --input weights/ltx-video-2b-v0.9.1.safetensors \
  --output weights/ltx-video-2b-v0.9.1-rust.safetensors
```

## CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--weights` | none | Transformer .safetensors (omit for random init) |
| `--tokenizer` | none | SentencePiece tokenizer model |
| `--text-weights` | none | Text encoder .safetensors (T5 or Gemma3) |
| `--device` | `cpu` | Inference device (`cpu`, `cuda`, `cuda:N`) |
| `--steps` | `20` | Denoising steps |
| `--prompt` | `"a colorful abstract pattern"` | Text prompt |
| `--prompts-file` | none | Text file with prompts (one per line) for batch mode |
| `--output-dir` | `batch_output` | Output directory for batch results |
| `--height` | `16` | Latent height |
| `--width` | `16` | Latent width |
| `--frames` | `4` | Number of frames |
| `--cfg` | `7.5` | Classifier-free guidance scale |

## Prerequisites

- Rust 1.75+
- CUDA toolkit (optional, for GPU inference)
- Python 3.10+ (for weight conversion and HuggingFace downloads)
- ffmpeg (for MP4/GIF export)

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
├── ltx-text-encoder/   # T5 + Gemma3 + SigLIP text encoders
├── ltx-loader/         # Checkpoint loading
├── ltx-quantization/   # FP8 quantization policy
├── ltx-test-utils/     # Golden file loading, assertions
├── ltx-app/            # eframe GUI application
├── ltx-core/           # Public API facade + inference binary
└── goldens/            # Golden reference data (.safetensors)
```

## Remaining Work

### Completed
- All 22 crates implemented (~13,600 LOC)
- 485 tests passing
- Inference pipeline loads 929/929 tensors from LTX-Video 2B model
- 28-layer transformer runs with real weights
- T5 text encoder with mmap + FP16 loading (memory-efficient)
- Prompt conditioning via T5 cross-attention (4096-dim context)
- GPU support via `--device cuda` flag
- eframe GUI with video playback toolbar and export (PNG/MP4/GIF)
- CLI with full argument support

### Known Limitations
- **VAE decoder** — Decoder architecture mismatch with Python model (7 up_blocks vs 4 in Rust). Needs architecture alignment.
- **Resolution** — 32x32 works with 8 steps on 32GB RAM. Higher resolutions require GPU or model sharding.
- **56 skipped weights** — Cross-attention K/V projections use context_dim=4096 (T5) but were trained with 2048 (Gemma3).

### Not Yet Implemented
1. **VAE decoder alignment** — Align Rust decoder architecture with Python model's 7-block structure
2. **Model sharding** — Split model across CPU/GPU for larger resolutions
3. **Audio pipeline** — Audio VAE + transformer for audio generation
4. **Full GPU inference** — Move transformer to GPU for faster denoising

## License

MIT
