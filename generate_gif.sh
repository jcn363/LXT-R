#!/usr/bin/env bash
# Generate a GIF from LTX-Video inference.
# Usage: ./generate_gif.sh "a sunset over mountains"
#        ./generate_gif.sh "a cat walking" --steps 20 --fps 12 --scale 512

set -euo pipefail

# Defaults
PROMPT="${1:-a colorful abstract pattern}"
STEPS=5
HEIGHT=16
WIDTH=16
FRAMES=4
FPS=8
SCALE=256
SCALING="lanczos"
OUTPUT="output.gif"

# Parse optional flags
shift || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps)   STEPS="$2"; shift 2 ;;
        --height)  HEIGHT="$2"; shift 2 ;;
        --width)   WIDTH="$2"; shift 2 ;;
        --frames)  FRAMES="$2"; shift 2 ;;
        --fps)     FPS="$2"; shift 2 ;;
        --scale)   SCALE="$2"; shift 2 ;;
        --output)  OUTPUT="$2"; shift 2 ;;
        --pixel)   SCALING="neighbor"; shift ;;
        *)         echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Paths
WEIGHTS="weights/ltx-video-2b-v0.9.1-rust.safetensors"
TOKENIZER="weights/tokenizer/spiece.model"
TEXT_WEIGHTS="weights/text_encoder.safetensors"

# Run inference
echo "Running inference: \"$PROMPT\" (${HEIGHT}x${WIDTH}x${FRAMES}, ${STEPS} steps)"
cargo run --release --bin ltx-inference -- \
    --weights "$WEIGHTS" \
    --tokenizer "$TOKENIZER" \
    --text-weights "$TEXT_WEIGHTS" \
    --prompt "$PROMPT" \
    --steps "$STEPS" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --frames "$FRAMES"

# Convert PGM frames to GIF
echo "Creating GIF: ${SCALE}x${SCALE}, ${FPS}fps, ${SCALING} scaling"
ffmpeg -y \
    -framerate "$FPS" \
    -i output_frames/frame_%04d.pgm \
    -vf "scale=${SCALE}:${SCALE}:flags=${SCALING},split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
    -loop 0 \
    "$OUTPUT"

echo "Done: $OUTPUT"
