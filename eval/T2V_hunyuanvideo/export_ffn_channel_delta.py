"""
Export multi-layer FFN weight deltas for HunyuanVideo-1.5 T2V erasure.

Hooks in test_erased.py are on transformer_blocks[i].ff_context.net[0]
(and optionally ff.net[0]).  The equivalent weight edit is net[2].weight[:, channels]
(column scaling).

Supports both channel-data formats:
  Old flat format: {"indices": [[...], ...54...], "values": [...]}
    → treated as ffn_context only
  New nested format: {"ff": {"indices": ..., "values": ...},
                      "ffn_context": {"indices": ..., "values": ...}}
    → supports both ff and ffn_context

Default config matches test_erased.py best params:
  module_types=ffn_context, inter_range=range(20,30), channel_num=15, scale=-4.0
"""
import argparse
import json
import os
from pathlib import Path
from typing import List

import torch
from diffusers import HunyuanVideo15Pipeline

SCRIPT_DIR = Path(__file__).parent
DEFAULT_MODEL_ID = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
DEFAULT_CACHE_DIR = f'{os.environ["HOME"]}/common_model/huggingface/hub'
DEFAULT_CHANNEL_DATA = str(SCRIPT_DIR / "channel_data_t2v.json")
DEFAULT_MODULE_TYPES = "ffn_context"
DEFAULT_INTER_RANGE = "20,30"
DEFAULT_CHANNEL_NUM = 15
DEFAULT_SCALE = -4.0
DEFAULT_OUTPUT = str(SCRIPT_DIR / "ffn_channel_delta.pt")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export multi-layer FFN column delta patch for HunyuanVideo T2V. "
            "Reads channel rankings from a JSON file and builds weight deltas "
            "equivalent to the forward-hook erasure in test_erased.py."
        )
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--channel-data", default=DEFAULT_CHANNEL_DATA,
        help="Path to channel data JSON (old or new format).",
    )
    parser.add_argument(
        "--module-types", default=DEFAULT_MODULE_TYPES,
        help="Comma-separated module types: ff, ffn_context.",
    )
    parser.add_argument(
        "--inter-range", default=DEFAULT_INTER_RANGE,
        help="START,END (exclusive) layer indices, e.g. '20,30'.",
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


def load_channel_data(path: str) -> dict:
    """Return dict of {module_type: List[List[int]]} (per-layer channel lists)."""
    with open(path, "r") as f:
        raw = json.load(f)
    if set(raw.keys()) <= {"indices", "values"}:
        # Old flat format ({indices: [...], values: [...]}): treat as ffn_context only
        return {"ffn_context": raw["indices"]}
    # New nested format ({ff: {indices, values}, ffn_context: {indices, values}})
    return {module_type: sub["indices"] for module_type, sub in raw.items()}


def module_type_to_layer_path(module_type: str, layer_idx: int) -> str:
    if module_type == "ff":
        return f"transformer_blocks.{layer_idx}.ff.net.2"
    elif module_type == "ffn_context":
        return f"transformer_blocks.{layer_idx}.ff_context.net.2"
    else:
        raise ValueError(f"Unknown module_type: {module_type!r}")


def get_module_by_path(root, path: str):
    m = root
    for part in path.split("."):
        m = m[int(part)] if part.isdigit() else getattr(m, part)
    return m


def build_patch_entry(transformer, layer_path: str, channels: List[int],
                      scale: float) -> dict:
    layer = get_module_by_path(transformer, layer_path)
    if not isinstance(layer, torch.nn.Linear):
        raise TypeError(f"Expected Linear at {layer_path}, got {type(layer)}")
    if any(c >= layer.in_features for c in channels):
        raise ValueError(
            f"{layer_path}: channel index exceeds in_features={layer.in_features}"
        )
    with torch.no_grad():
        original = layer.weight[:, channels].detach().cpu().float()
        delta = original * (scale - 1.0)
    return {
        "layer_path": layer_path,
        "param_name": "weight",
        "modify_mode": "columns",
        "channels": channels,
        "delta_slice": delta.contiguous(),
        "original_slice": original.contiguous(),
    }


def main():
    args = parse_args()
    inter_range = parse_inter_range(args.inter_range)
    module_types = [t.strip() for t in args.module_types.split(",")]

    channel_data = load_channel_data(args.channel_data)
    for mt in module_types:
        if mt not in channel_data:
            raise ValueError(
                f"module_type '{mt}' not found in channel data "
                f"(available: {list(channel_data.keys())})"
            )
    print(f"Loaded channel data for: {list(channel_data.keys())}")

    print("Loading transformer (bfloat16, CPU)...")
    pipe = HunyuanVideo15Pipeline.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    transformer = pipe.transformer

    patches = []
    for mt in module_types:
        for i in inter_range:
            layer_path = module_type_to_layer_path(mt, i)
            channels = channel_data[mt][i][: args.channel_num]
            entry = build_patch_entry(transformer, layer_path, channels, args.scale)
            patches.append(entry)
            print(f"  {mt}[{i:2d}] {layer_path}  channels[0:3]={channels[:3]}")

    patch_data = {
        "format": "ffn_column_delta_v2",
        "base_model_id": args.model_id,
        "scale": args.scale,
        "channel_num": args.channel_num,
        "module_types": module_types,
        "inter_range": [inter_range.start, inter_range.stop],
        "patches": patches,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(patch_data, args.output)
    print(f"\nSaved {len(patches)}-layer delta patch to: {args.output}")
    print(f"  model: {args.model_id}  scale={args.scale}  "
          f"channel_num={args.channel_num}  module_types={module_types}")


if __name__ == "__main__":
    main()
