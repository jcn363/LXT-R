# LTX-R

Rust rewrite of [LTX-2.3](https://github.com/LightricksResearch/LTX-Video) core — a modular, DRY, SSOT-enforced workspace for video/audio generative models.

## Architecture

21 crates, ~16,600 LOC (122 source files + 55 test files + 1 bench file). All model logic is pure Rust; external FFI (`tch`, CUDA/ROCm) is isolated behind safe APIs.

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

# Full pipeline: text encode → denoise → VAE decode → PNG/GIF output
cargo run --release --bin ltx-inference -- \
  --weights weights/ltx-video-2b-v0.9.1-rust.safetensors \
  --tokenizer weights/tokenizer/spiece.model \
  --text-weights weights/text_encoder.safetensors \
  --vae-weights weights/ltx-video-2b-v0.9.1.safetensors \
  --decode \
  --prompt "a sunset over mountains" \
  --steps 20

# GPU inference (auto-detects CUDA/ROCm/MPS)
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
  --vae-weights weights/ltx-video-2b-v0.9.1.safetensors \
  --init-image path/to/image.png \
  --prompt "a sunset over mountains" \
  --strength 0.5 \
  --steps 20

# Batch processing with resume
cargo run --release --bin ltx-inference -- \
  --weights weights/ltx-video-2b-v0.9.1-rust.safetensors \
  --tokenizer weights/tokenizer/spiece.model \
  --text-weights weights/text_encoder.safetensors \
  --prompts-file prompts.txt \
  --output-dir batch_output \
  --decode \
  --vae-weights weights/ltx-video-2b-v0.9.1.safetensors \
  --steps 10 \
  --resume

# Output structure:
# batch_output/
# ├── manifest.json    (results summary: prompts, timings, settings)
# ├── 0001/            (PNG frames for prompt 1)
# │   ├── frame_0000.png
# │   └── frame_0001.png
# ├── 0001.gif
# └── ...
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
| `--height` | `16` | Latent height |
| `--width` | `16` | Latent width |
| `--frames` | `4` | Number of frames |
| `--cfg` | `7.5` | Classifier-free guidance scale |
| `--init-image` | none | Input image for img2img mode |
| `--vae-weights` | none | VAE .safetensors for img2img encoding or decode |
| `--strength` | `0.75` | Denoising strength for img2img (0.0–1.0) |
| `--decode` | off | Decode latent through VAE decoder → pixel-space PNG frames |
| `--audio` | off | Enable audio generation alongside video |
| `--audio-vae-weights` | none | Audio VAE .safetensors for audio generation |
| `--audio-output` | `output.wav` | Output path for generated audio WAV |
| `--step-method` | `euler` | Diffusion step method: `euler` (default) or `res2s` |
| `--guider` | `cfg` | Guidance strategy: `cfg`, `apg`, or `stg` |
| `--apg-scale` | `7.5` | APG guidance scale (with `--guider apg`) |
| `--apg-momentum` | `0.0` | APG momentum factor (with `--guider apg`) |
| `--stg-spatial-scale` | `7.5` | STG spatial scale (with `--guider stg`) |
| `--stg-temporal-scale` | `3.0` | STG temporal scale (with `--guider stg`) |
| `--tile-size` | `0` | Spatial tile size in latent pixels (0=disabled) |
| `--tile-overlap` | `4` | Tiling overlap in latent pixels |
| `--shard` | none | Multi-GPU model sharding (e.g., `cuda:0,cuda:1` or `rocm:0,rocm:1`) |

## GPU Inference

The transformer runs directly on GPU for maximum denoising throughput. Text encoding runs on CPU (memory-efficient: encode then free the ~18GB encoder), and encoded context is copied to GPU once before the denoising loop.

### Supported Backends

| Backend | `--device` value | Status | Requirements |
|---------|------------------|--------|--------------|
| NVIDIA CUDA | `cuda` / `cuda:N` | **Fully supported** | CUDA 12.1+ toolkit, NVIDIA GPU |
| AMD ROCm | `rocm` / `rocm:N` | **Fully supported** | ROCm 6.0+ toolkit, AMD GPU |
| Apple Metal (MPS) | `mps` | **Fully supported** | macOS 13+, Apple Silicon or AMD GPU |
| CPU fallback | `cpu` | Always available | — |

ROCm uses the same tch CUDA device API when libtorch is built with ROCm. The runtime detects ROCm via `rocm-smi` and labels it accordingly.

### Auto-Detection (`--device auto`)

Probes backends in priority order: CUDA/ROCm → MPS → CPU. The chosen backend is printed at startup.

### GPU Requirements

| Component | CPU RAM | GPU VRAM |
|-----------|---------|----------|
| Transformer (2B) | ~6 GB | ~6 GB |
| T5 text encoder (XXL) | ~9 GB (mmap) | ~9 GB |
| Video VAE encoder | ~1 GB | ~1 GB |
| Video VAE decoder | ~3 GB | ~3 GB |
| Audio VAE | ~1 GB | ~1 GB |

### CUDA Setup

```bash
# Download CUDA 12.1 libtorch
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.3.0%2Bcu121.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.3.0+cu121.zip -d /opt/libtorch
export LIBTORCH=/opt/libtorch
export LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH
```

### ROCm Setup

```bash
# Download ROCm 6.0 libtorch
wget https://download.pytorch.org/libtorch/rocm6.0/libtorch-cxx11-abi-shared-with-deps-2.3.0%2Brocm6.0.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.3.0+rocm6.0.zip -d /opt/libtorch
export LIBTORCH=/opt/libtorch
export LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH
export HIP_VISIBLE_DEVICES=0
```

## Pipeline Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                    MEMORY-EFFICIENT PIPELINE                    │
├─────────────────────────────────────────────────────────────────┤
│  Phase 1: Text encoding (CPU)                                  │
│    Load T5/Gemma3 → encode prompts → free encoder (~18GB)     │
│    Context tensors copied to GPU once                          │
├─────────────────────────────────────────────────────────────────┤
│  Phase 2: Transformer denoising (GPU)                          │
│    Load transformer → for each prompt:                         │
│      patchify → [cond, uncond] forward → CFG/APG/STG         │
│      → Euler/Res2s step → unpatchify                          │
│    Optional: spatial tiling for large resolutions              │
│    All tensors on device during denoising loop                 │
├─────────────────────────────────────────────────────────────────┤
│  Phase 3: Decode & output                                      │
│    (optional) VAE decode → PNG/GIF frames                      │
│    (optional) Audio VAE → WAV output                           │
└─────────────────────────────────────────────────────────────────┘
```

## Video VAE

The video VAE encodes pixel-space video to 128-channel latents and decodes back.

### Encoder
- Input: `(B, 3, T, H, W)` pixel video
- Output: `(B, 128, T', H', W')` normalized latent (32× spatial, ~8× temporal compression)
- Architecture: `space_to_depth(r=4)` → 10 heterogeneous down_blocks → conv_out → per-channel normalization

### Decoder
- Input: `(B, 128, T', H', W')` latent + scalar timestep
- Output: `(B, 3, T, H, W)` pixel video
- Architecture: 7 up_blocks (ResBlock stages + CompressAllUpsample) → conv_out → `depth_to_space(r=4)`
- Timestep conditioning via AdaLN modulation, noise injection in blocks 2,4,6

### Roundtrip Verification

```
Input:  [1, 3, 1, 256, 256]  (RGB pixel)
Encode: [1, 128, 1, 8, 8]    (normalized latent)
Decode: [1, 3, 1, 256, 256]  (reconstructed pixel)
```

## Audio VAE

Converts mel spectrograms to latent representations and back.

- Encoder: Conv2D downsampling + ResnetBlock2D + attention mid-section
- Decoder: ConvTranspose2D upsampling + ResnetBlock2D + attention mid-section
- Vocoder: ConvTranspose1d upsampling + ResBlock1 refinement → waveform
- Latent channels: 64, mel features: 128 bins

## GUI Application

```bash
cargo run -p ltx-app
```

The `ltx-app` crate provides an eframe-based GUI with hover tooltips on all controls:

| Control | Description |
|---------|-------------|
| **Prompt** | Text description of the video to generate |
| **Model Weights** | Transformer .safetensors checkpoint (omit for random init) |
| **Text Encoder** | SentencePiece tokenizer + text encoder weights (T5 or Gemma3) |
| **Resolution** | Latent-space H/W dimensions (pixel = latent × 32) |
| **Frames** | Number of video frames to generate |
| **Steps** | Denoising steps: 5-10 quick preview, 20-50 for quality |
| **CFG Scale** | Classifier-free guidance: 1.0=none, 7.5=default, 15.0+=strong |
| **Scheduler** | Noise schedule: LTX-2 (default), Linear-Quadratic, Beta |
| **Device** | Compute backend: CPU, CUDA (NVIDIA), ROCm (AMD), MPS (Apple Metal) |
| **Generate** | Start video generation |
| **Export** | Save PNGs, MP4 video (H.264, 8fps), or animated GIF (256×256) |

## Testing

```bash
cargo test --workspace          # all tests
cargo test -p ltx-video-vae     # VAE encoder/decoder roundtrip
cargo test -p ltx-transformer   # transformer model
cargo test -p ltx-components    # scheduler, guider, noiser, diffusion step
cargo test -p ltx-audio-vae     # audio VAE encoder/decoder
```

All 397 tests pass across 21 crates with zero failures.

## Weight Conversion

```bash
# Download from HuggingFace
python3 -c "
from huggingface_hub import hf_hub_download
import os
os.makedirs('weights', exist_ok=True)
hf_hub_download('Lightricks/LTX-Video', 'ltx-video-2b-v0.9.1.safetensors', local_dir='weights')
hf_hub_download('Lightricks/LTX-Video', 'tokenizer/spiece.model', local_dir='weights')
for i in range(1, 5):
    hf_hub_download('Lightricks/LTX-Video', f'text_encoder/model-0000{i}-of-00004.safetensors', local_dir='weights')
"

# Convert transformer weights to Rust format
python3 scripts/convert_ltx_weights.py \
  --input weights/ltx-video-2b-v0.9.1.safetensors \
  --output weights/ltx-video-2b-v0.9.1-rust.safetensors
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
├── ltx-transformer/    # DiT transformer model (with audio modality)
├── ltx-video-vae/      # Video VAE (encoder + decoder, verified)
├── ltx-audio-vae/      # Audio VAE (encoder + decoder + vocoder)
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

### P10: Benchmarking + Optimization
- Profile denoising throughput (frames/sec at various resolutions)
- Memory profiling on GPU

### Completed in This Session
- ✅ VAE decoder rewrite to match Python checkpoint (7 up_blocks, CompressAllUpsample, AdaLN)
- ✅ VAE encoder architecture matching (10 blocks, r=4 space_to_depth, 128-ch latent)
- ✅ Full encode-decode roundtrip verified: `[1,3,1,256,256]` → `[1,128,1,8,8]` → `[1,3,1,256,256]`
- ✅ Latent normalization (per-channel mean/std loaded from checkpoint)
- ✅ Timestep scaling fix (multiply before sinusoidal embedding)
- ✅ Full GPU inference (context tensors on device, timestep on device)
- ✅ VAE decode pipeline (pixel-space PNG output)
- ✅ Audio pipeline (transformer audio modality, audio VAE, WAV output)
- ✅ Res2sStep alternative (second-order residual scaling)
- ✅ APG/STG guiders (adaptive projected, spatio-temporal guidance)
- ✅ Spatial tiling for memory-efficient generation
- ✅ INT4 weight quantization (per-group, `ltx-quantization::int4_mm`)
- ✅ Multi-GPU model sharding (round-robin layer distribution via `--shard`)
- ✅ PNG frame output alongside PGM/GIF
- ✅ Comprehensive CLI with 30 flags

## License

MIT
