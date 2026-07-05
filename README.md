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

# Real weights on GPU (auto-detects CUDA/MPS/CPU)
cargo run --release --bin ltx-inference -- \
  --weights weights/ltx-video-2b-v0.9.1-rust.safetensors \
  --device auto \
  --steps 20

# Custom resolution
cargo run --release --bin ltx-inference -- \
  --weights weights/ltx-video-2b-v0.9.1-rust.safetensors \
  --steps 8 --height 32 --width 32 --frames 8

# img2img — transform an existing image with a text prompt
cargo run --release --bin ltx-inference -- \
  --weights weights/ltx-video-2b-v0.9.1-rust.safetensors \
  --tokenizer weights/tokenizer/spiece.model \
  --text-weights weights/text_encoder.safetensors \
  --init-image path/to/image.png \
  --prompt "a sunset over mountains" \
  --strength 0.5 \
  --steps 20

# img2img with batch prompts (same image, multiple prompts)
cargo run --release --bin ltx-inference -- \
  --weights weights/ltx-video-2b-v0.9.1-rust.safetensors \
  --tokenizer weights/tokenizer/spiece.model \
  --text-weights weights/text_encoder.safetensors \
  --init-image path/to/image.png \
  --prompts-file prompts.txt \
  --strength 0.6 \
  --steps 15

# Batch processing — multiple prompts from file
printf "a sunset over mountains\na cat walking on grass\na dog running\n" > prompts.txt
cargo run --release --bin ltx-inference -- \
  --weights weights/ltx-video-2b-v0.9.1-rust.safetensors \
  --tokenizer weights/tokenizer/spiece.model \
  --text-weights weights/text_encoder.safetensors \
  --prompts-file prompts.txt \
  --output-dir batch_output \
  --steps 10

# Resume interrupted batch (skip completed prompts)
cargo run --release --bin ltx-inference -- \
  --weights weights/ltx-video-2b-v0.9.1-rust.safetensors \
  --tokenizer weights/tokenizer/spiece.model \
  --text-weights weights/text_encoder.safetensors \
  --prompts-file prompts.txt \
  --output-dir batch_output \
  --steps 10 \
  --resume

# With custom seed for reproducibility
cargo run --release --bin ltx-inference -- \
  --prompts-file prompts.txt \
  --seed 123 \
  --steps 10

# Output structure:
# batch_output/
# ├── manifest.json    (results summary: prompts, timings, settings)
# ├── 0001/            (frames + gif for prompt 1)
# ├── 0001.gif
# ├── 0002/            (frames + gif for prompt 2)
# ├── 0002.gif
# └── 0003/            (frames + gif for prompt 3)
# └── 0003.gif
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
| `--device` | `auto` | Inference device (`auto`, `cpu`, `cuda`, `cuda:N`, `mps`) |
| `--steps` | `20` | Denoising steps |
| `--prompt` | `"a colorful abstract pattern"` | Text prompt |
| `--prompts-file` | none | Text file with prompts (one per line) for batch mode |
| `--output-dir` | `batch_output` | Output directory for batch results |
| `--seed` | `42` | Random seed for reproducibility |
| `--resume` | off | Skip prompts whose output directory already exists |
| `--init-image` | none | Input image for img2img mode (any format: PNG, JPG, etc.) |
| `--strength` | `0.75` | Denoising strength for img2img (0.0=keep original, 1.0=full txt2img) |
| `--height` | `16` | Latent height |
| `--width` | `16` | Latent width |
| `--frames` | `4` | Number of frames |
| `--cfg` | `7.5` | Classifier-free guidance scale |

## Prerequisites

- Rust 1.75+
- CUDA toolkit (optional, for GPU inference)
- Python 3.10+ (for weight conversion and HuggingFace downloads)
- ffmpeg (for MP4/GIF export)

## GPU Inference

The transformer can run on GPU for faster denoising. The `--device` flag controls which backend is used, and defaults to `auto` for hands-free detection.

### Supported Backends

| Backend | `--device` value | Status | Requirements |
|---------|------------------|--------|--------------|
| NVIDIA CUDA | `cuda` / `cuda:N` | **Fully supported** | CUDA 12.1+ toolkit, NVIDIA GPU |
| Apple Metal (MPS) | `mps` | **Fully supported** | macOS 13+, Apple Silicon or AMD GPU |
| CPU fallback | `cpu` | Always available | — |

### Auto-Detection (`--device auto`)

When `--device auto` is specified (the default), the runtime probes backends in priority order:

1. **CUDA** — checks `tch::Cuda::is_available()`
2. **MPS** — checks `tch::utils::has_mps()` (macOS only)
3. **CPU** — used when no GPU accelerator is found

The chosen backend is printed at startup:
```
auto-detected: NVIDIA CUDA (gpu 0)
auto-detected: Apple Metal Performance Shaders
no GPU accelerator detected, using CPU
```

### CUDA Setup

**Option 1: Download CUDA libtorch (recommended)**

```bash
# Download CUDA 12.1 libtorch
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.3.0%2Bcu121.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.3.0+cu121.zip -d /opt/libtorch

# Set environment variables
export LIBTORCH=/opt/libtorch
export LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH
```

**Option 2: Use system CUDA toolkit**

```bash
export LIBTORCH=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Verify CUDA detection:**
```bash
nvidia-smi          # confirm GPU is visible
nvcc --version      # confirm CUDA toolkit
```

### macOS Metal Setup

No extra setup is required — `tch` ships with Metal support when built on macOS. Just run:

```bash
cargo run --release --bin ltx-inference -- --device mps --steps 20
```

### Multi-GPU Selection

```bash
# Use GPU 0 (default)
--device cuda

# Use GPU 1
--device cuda:1

# Auto-detect picks the first available GPU
--device auto
```

### GPU Requirements

| Model | RAM (CPU) | VRAM (GPU) |
|-------|-----------|------------|
| Transformer (2B) | ~6 GB | ~6 GB |
| T5 text encoder (XXL) | ~9 GB (mmap) | ~9 GB |
| Both combined | ~15 GB | ~15 GB |

### Future Backends

The following backends are **not yet supported** by the `tch` (libtorch) bindings used by this project. Integrating them would require switching to or adding alternative compute backends:

| Backend | Notes |
|---------|-------|
| **ROCm** (AMD) | Supported by libtorch on Linux; requires ROCm-enabled libtorch build. Set `LIBTORCH` to a ROCm build to enable. |
| **Vulkan** | Not supported by `tch`. Would require a Vulkan compute crate (e.g., `ash`, `gpu-allocator`). |
| **WebGPU** | Not supported by `tch`. Would require targeting WASM with `wgpu` or similar. |
| **Intel oneAPI** | Not supported by `tch`. Would require oneAPI SYCL runtime integration. |

Contributions to add these backends are welcome. See `PLAN.md` for architectural guidance.

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
- GPU support via `--device auto` (CUDA, MPS auto-detection) or explicit `cuda`/`mps`/`cpu`
- Batch processing: `--prompts-file` with upfront encoding, `--resume`, `--seed`, manifest.json
- img2img mode: `--init-image` + `--strength` for image-guided generation
- eframe GUI with video playback toolbar and export (PNG/MP4/GIF)
- CLI with full argument support

### Known Limitations
- **VAE decoder** — Decoder architecture mismatch with Python model (7 up_blocks vs 4 in Rust). Needs architecture alignment.
- **VAE encoder** — Encoder architecture mismatch with Python model (timestep-conditioned, different channel dims). img2img uses direct image-to-latent conversion instead.
- **Resolution** — 32x32 works with 8 steps on 32GB RAM. Higher resolutions require GPU or model sharding.
- **56 skipped weights** — Cross-attention K/V projections use context_dim=4096 (T5) but were trained with 2048 (Gemma3).

### Not Yet Implemented
1. **VAE decoder alignment** — Align Rust decoder architecture with Python model's 7-block structure
2. **Model sharding** — Split model across CPU/GPU for larger resolutions
3. **Audio pipeline** — Audio VAE + transformer for audio generation
4. **Full GPU inference** — Move transformer to GPU for faster denoising

## License

MIT
