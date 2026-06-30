"""
Export multi-layer FFN weight deltas for FLUX.1-dev T2I erasure.

result.json encodes 76 modules in flat order:
  [0..18]  transformer_blocks[i].ff.net[0]          → edit net[2].weight columns
  [19..37] transformer_blocks[i].ff_context.net[0]  → edit net[2].weight columns
  [38..75] single_transformer_blocks[i].proj_mlp    → edit proj_mlp.weight rows

The hook in the notebook scales net[0]/proj_mlp *output* channels.
For ff/ff_context the equivalent weight edit is net[2].weight[:, channels] (columns).
For proj_mlp the equivalent weight edit is proj_mlp.weight[channels, :] (rows).
"""
import argparse
import json
import os
from pathlib import Path
from typing import List

import torch
from diffusers import FluxPipeline

SCRIPT_DIR = Path(__file__).parent
DEFAULT_MODEL_ID = "black-forest-labs/FLUX.1-dev"
DEFAULT_CACHE_DIR = f'{os.environ["HOME"]}/common_model/huggingface/'
DEFAULT_CHANNEL_DATA = str(SCRIPT_DIR / "result.json")
DEFAULT_INTER_RANGE = "20,35"   # flat-list indices; range(20,35) = ff_context[1..15]
DEFAULT_CHANNEL_NUM = 20
DEFAULT_SCALE = -3.0
DEFAULT_OUTPUT = str(SCRIPT_DIR / "ffn_channel_delta.pt")

NUM_FF = 19            # transformer_blocks  with ff
NUM_FF_CTX = 19        # transformer_blocks  with ff_context
# flat indices: 0..18 = ff, 19..37 = ff_context, 38..75 = proj_mlp


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export multi-layer FFN column/row delta patch for FLUX.1-dev T2I. "
            "Reads channel rankings from result.json and builds weight deltas "
            "equivalent to the forward-hook erasure in the notebook."
        )
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--channel-data", default=DEFAULT_CHANNEL_DATA,
        help="Path to result.json with flat 'indices' list of 76 entries.",
    )
    parser.add_argument(
        "--inter-range", default=DEFAULT_INTER_RANGE,
        help="START,END (exclusive) of flat module indices to patch, e.g. '20,35'.",
    )
    parser.add_argument(
        "--channel-num", type=int, default=DEFAULT_CHANNEL_NUM,
        help="Number of top channels to use per layer.",
    )
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def parse_inter_range(s: str):
    parts = s.split(",")
    if len(parts) != 2:
        raise ValueError(f"--inter-range must be 'START,END', got '{s}'")
    return range(int(parts[0]), int(parts[1]))


def flat_idx_to_layer(idx: int):
    """Map flat module index to (layer_path, modify_mode)."""
    if 0 <= idx < NUM_FF:
        return f"transformer_blocks.{idx}.ff.net.2", "columns"
    elif NUM_FF <= idx < NUM_FF + NUM_FF_CTX:
        i = idx - NUM_FF
        return f"transformer_blocks.{i}.ff_context.net.2", "columns"
    else:
        i = idx - NUM_FF - NUM_FF_CTX
        return f"single_transformer_blocks.{i}.proj_mlp", "rows"


def get_module_by_path(root, path: str):
    m = root
    for part in path.split("."):
        m = m[int(part)] if part.isdigit() else getattr(m, part)
    return m


def build_patch_entry(transformer, layer_path: str, modify_mode: str,
                      channels: List[int], scale: float) -> dict:
    layer = get_module_by_path(transformer, layer_path)
    if not isinstance(layer, torch.nn.Linear):
        raise TypeError(f"Expected Linear at {layer_path}, got {type(layer)}")

    with torch.no_grad():
        if modify_mode == "columns":
            if any(c >= layer.in_features for c in channels):
                raise ValueError(
                    f"{layer_path}: channel index out of range [0, {layer.in_features-1}]"
                )
            original = layer.weight[:, channels].detach().cpu().float()
        else:  # rows
            if any(c >= layer.out_features for c in channels):
                raise ValueError(
                    f"{layer_path}: channel index out of range [0, {layer.out_features-1}]"
                )
            original = layer.weight[channels, :].detach().cpu().float()

        delta = original * (scale - 1.0)

    return {
        "layer_path": layer_path,
        "param_name": "weight",
        "modify_mode": modify_mode,
        "channels": channels,
        "delta_slice": delta.contiguous(),
        "original_slice": original.contiguous(),
    }


def main():
    args = parse_args()
    inter_range = parse_inter_range(args.inter_range)

    with open(args.channel_data, "r") as f:
        data = json.load(f)
    all_indices = data["indices"]
    print(f"Loaded channel data: {len(all_indices)} flat entries")

    print(f"Loading transformer (bfloat16, CPU)...")
    pipe = FluxPipeline.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    transformer = pipe.transformer

    patches = []
    for flat_idx in inter_range:
        layer_path, modify_mode = flat_idx_to_layer(flat_idx)
        channels = all_indices[flat_idx][: args.channel_num]
        entry = build_patch_entry(transformer, layer_path, modify_mode, channels, args.scale)
        patches.append(entry)
        print(f"  flat[{flat_idx:2d}] {layer_path} ({modify_mode}) "
              f"channels[0:3]={channels[:3]}")

    patch_data = {
        "format": "ffn_column_delta_v2",
        "base_model_id": args.model_id,
        "scale": args.scale,
        "channel_num": args.channel_num,
        "inter_range": [inter_range.start, inter_range.stop],
        "patches": patches,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(patch_data, args.output)
    print(f"\nSaved {len(patches)}-layer delta patch to: {args.output}")
    print(f"  model: {args.model_id}  scale={args.scale}  channel_num={args.channel_num}")


if __name__ == "__main__":
    main()
