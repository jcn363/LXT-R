#!/usr/bin/env python3
"""Convert PyTorch model weights to safetensors format for Rust inference.

This script converts checkpoint files (.pt, .pth, .ckpt) to safetensors
format with the correct key naming for the Rust implementation.

Usage:
    python3 scripts/convert_weights.py --input model.pt --output model.safetensors
    python3 scripts/convert_weights.py --input checkpoint.ckpt --output model.safetensors
    python3 scripts/convert_weights.py --input model.safetensors --output model.safetensors  # re-key

The script will:
1. Load the checkpoint (supports .pt, .pth, .ckpt, .safetensors)
2. Extract the state dict
3. Apply key remapping to match Rust module paths
4. Save as safetensors
"""

import argparse
import os
import re
import torch
from safetensors.torch import save_file, load_file


# Key mapping from Python LTX-2.3 to Rust implementation
# Python uses "model.diffusion_model." prefix; Rust uses flat paths
KEY_REMAPPING = {
    # Remove common prefixes
    "model.diffusion_model.": "",
    "model.": "",

    # Transformer blocks
    "blocks.": "blocks.",

    # Attention
    "attn1.": "self_attn.",
    "attn2.": "cross_attn.",
    "norm1.": "norm1.",
    "norm_cross.": "norm_cross.",
    "norm2.": "norm2.",

    # Timestep
    "time_embed.": "adaln/emb/",
    "time_proj.": "adaln/emb/time_proj/",
    "time_mlp.": "adaln/emb/embedder/",
    "adaln.": "adaln/",
    "adaln_linear.": "adaln/linear",

    # Feed forward
    "ff.": "ff/",
    "net.0.": "net_0",
    "net.2.": "net_2",

    # VAE encoder
    "encoder.conv_in.": "encoder/conv_in/",
    "encoder.down.": "encoder/down_",
    "encoder.mid.": "encoder/mid/",
    "encoder.conv_norm_out.": "encoder/conv_norm_out/",
    "encoder.conv_out.": "encoder/conv_out/",

    # VAE decoder
    "decoder.conv_in.": "decoder/conv_in/",
    "decoder.up.": "decoder/up_",
    "decoder.mid.": "decoder/mid/",
    "decoder.conv_norm_out.": "decoder/conv_norm_out/",
    "decoder.conv_out.": "decoder/conv_out/",

    # Norm layers
    "weight": "weight",
    "bias": "bias",
}


def load_checkpoint(path: str) -> dict:
    """Load a checkpoint file and return the state dict."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".safetensors":
        print(f"Loading safetensors: {path}")
        return load_file(path)

    elif ext in [".pt", ".pth"]:
        print(f"Loading PyTorch checkpoint: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            # Check if it's a state dict or contains one
            if "state_dict" in checkpoint:
                return checkpoint["state_dict"]
            elif "model" in checkpoint:
                if isinstance(checkpoint["model"], dict):
                    return checkpoint["model"]
            return checkpoint
        return checkpoint

    elif ext == ".ckpt":
        print(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                return checkpoint["state_dict"]
            elif "model" in checkpoint:
                if isinstance(checkpoint["model"], dict):
                    return checkpoint["model"]
        return checkpoint

    else:
        raise ValueError(f"Unsupported file format: {ext}")


def remap_keys(state_dict: dict) -> dict:
    """Apply key remapping to match Rust module paths."""
    new_state_dict = {}

    for key, value in state_dict.items():
        new_key = key

        # Apply remapping rules (longest match first)
        for old_pattern, new_pattern in sorted(KEY_REMAPPING.items(), key=lambda x: -len(x[0])):
            if old_pattern in new_key:
                new_key = new_key.replace(old_pattern, new_pattern)
                break

        # Clean up double slashes and leading slashes
        new_key = new_key.replace("//", "/")
        new_key = new_key.lstrip("/")

        new_state_dict[new_key] = value

    return new_state_dict


def print_state_dict_info(state_dict: dict, title: str = "State Dict"):
    """Print information about a state dict."""
    print(f"\n{title}:")
    print(f"  Total tensors: {len(state_dict)}")

    # Group by prefix
    prefixes = {}
    for key in state_dict.keys():
        prefix = key.split("/")[0] if "/" in key else key.split(".")[0]
        prefixes[prefix] = prefixes.get(prefix, 0) + 1

    print("  Top-level modules:")
    for prefix, count in sorted(prefixes.items()):
        print(f"    {prefix}: {count} tensors")

    # Show a few examples
    print("  Example keys:")
    for i, key in enumerate(sorted(state_dict.keys())[:10]):
        shape = list(state_dict[key].shape)
        print(f"    {key}: {shape}")
    if len(state_dict) > 10:
        print(f"    ... and {len(state_dict) - 10} more")


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch weights to safetensors")
    parser.add_argument("--input", required=True, help="Input checkpoint path (.pt, .pth, .ckpt, .safetensors)")
    parser.add_argument("--output", required=True, help="Output safetensors path")
    parser.add_argument("--no-remap", action="store_true", help="Skip key remapping")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be converted without saving")
    args = parser.parse_args()

    # Load checkpoint
    state_dict = load_checkpoint(args.input)
    print_state_dict_info(state_dict, "Original State Dict")

    # Apply remapping
    if not args.no_remap:
        state_dict = remap_keys(state_dict)
        print_state_dict_info(state_dict, "After Remapping")

    # Save
    if not args.dry_run:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        save_file(state_dict, args.output)
        print(f"\nSaved to: {args.output}")

        # Verify
        loaded = load_file(args.output)
        print(f"Verified: {len(loaded)} tensors loaded from output file")
    else:
        print("\nDry run - no file saved")


if __name__ == "__main__":
    main()
