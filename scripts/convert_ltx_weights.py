#!/usr/bin/env python3
"""Convert LTX-Video model weights to Rust-compatible format.

Usage:
    python3 scripts/convert_ltx_weights.py \
        --input /tmp/ltx-model/ltxv-2b-0.9.8-distilled.safetensors \
        --output weights_rust.safetensors

Key mappings:
    model.diffusion_model.transformer_blocks.N.attn1 → blocks.N.self_attn
    model.diffusion_model.transformer_blocks.N.attn2 → blocks.N.cross_attn
    model.diffusion_model.transformer_blocks.N.ff → blocks.N.ff
    model.diffusion_model.adaln_single → blocks.N.adaln (duplicated to all blocks)
    model.diffusion_model.proj_out → proj_out
"""

import argparse
import os
import re
from safetensors import safe_open
from safetensors.torch import save_file
import torch


def map_key(key: str) -> str:
    """Map a Python key to the Rust VarStore key format."""
    # Remove model.diffusion_model prefix
    if key.startswith("model.diffusion_model."):
        key = key[len("model.diffusion_model."):]
    elif key.startswith("model."):
        key = key[len("model."):]

    # Map transformer block attention keys
    key = re.sub(r'transformer_blocks\.(\d+)\.attn1\.', r'blocks.\1.self_attn.', key)
    key = re.sub(r'transformer_blocks\.(\d+)\.attn2\.', r'blocks.\1.cross_attn.', key)
    key = re.sub(r'transformer_blocks\.(\d+)\.ff\.', r'blocks.\1.ff.', key)

    # Map adaln_single to block 0 adaln (will be duplicated to all blocks)
    key = re.sub(r'adaln_single\.emb\.timestep_embedder\.linear_1', r'blocks.0.adaln.emb.embedder.linear_1', key)
    key = re.sub(r'adaln_single\.emb\.timestep_embedder\.linear_2', r'blocks.0.adaln.emb.embedder.linear_2', key)
    key = re.sub(r'adaln_single\.linear', r'blocks.0.adaln.linear', key)

    # Map scale_shift_table to adaln.linear (same function)
    # In Python: scale_shift_table [6, dim] → adaln.linear output [6*dim]
    # These are the same weights, just different shapes
    key = re.sub(r'transformer_blocks\.(\d+)\.scale_shift_table', r'blocks.\1.adaln.linear', key)
    key = re.sub(r'^scale_shift_table$', r'blocks.0.adaln.linear', key)

    # Map feed forward keys
    key = key.replace('.ff.net.0.proj.', '.ff.net_0.')
    key = key.replace('.ff.net.2.', '.ff.net_2.')

    # Map attention output keys
    key = key.replace('.to_out.0.', '.to_out.')

    return key


def should_skip(key: str) -> bool:
    """Check if a key should be skipped (not in our VarStore)."""
    skip_patterns = [
        'norm1.weight', 'norm1.bias',
        'norm_cross.weight', 'norm_cross.bias',
        'norm2.weight', 'norm2.bias',
        'caption_projection',
        'vae.',
    ]
    for pattern in skip_patterns:
        if pattern in key:
            return True
    return False


def convert_weights(input_path: str, output_path: str):
    """Convert weights from Python format to Rust format."""
    print(f"Loading: {input_path}")

    state_dict = {}
    skipped_keys = []

    with safe_open(input_path, framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            new_key = map_key(key)

            if should_skip(new_key):
                skipped_keys.append(key)
                continue

            if key != new_key:
                print(f"  {key}")
                print(f"    → {new_key}")

            state_dict[new_key] = tensor.clone()  # Clone to avoid shared memory

    print(f"\nSkipped {len(skipped_keys)} keys (not in Rust VarStore)")

    # Duplicate adaln weights to all transformer blocks
    adaln_keys = [k for k in state_dict.keys() if k.startswith("blocks.0.adaln.")]
    if adaln_keys:
        print(f"\nDuplicating adaln weights to all blocks...")
        num_blocks = 28  # 2B model has 28 blocks
        for block_idx in range(1, num_blocks):
            for key in adaln_keys:
                new_key = key.replace("blocks.0.adaln.", f"blocks.{block_idx}.adaln.")
                state_dict[new_key] = state_dict[key].clone()
                print(f"  {key} → {new_key}")

    print(f"\nSaving {len(state_dict)} tensors to: {output_path}")
    save_file(state_dict, output_path)

    # Verify
    with safe_open(output_path, framework="pt") as f:
        loaded_keys = list(f.keys())
        print(f"Verified: {len(loaded_keys)} tensors loaded")

    return state_dict


def main():
    parser = argparse.ArgumentParser(description="Convert LTX-Video weights to Rust format")
    parser.add_argument("--input", required=True, help="Input safetensors path")
    parser.add_argument("--output", default="weights_rust.safetensors", help="Output path")
    args = parser.parse_args()

    convert_weights(args.input, args.output)
    print("\nDone! Use with: cargo run --bin ltx-inference -- --weights weights_rust.safetensors")


if __name__ == "__main__":
    main()
