"""
Export multi-layer FFN weight deltas for FLUX.1-Kontext-dev TI2I erasure.

channel_data_ti2i.json structure:
  {
    "ff":          {"indices": [[...], ...19...], "values": ...},
    "ffn_context": {"indices": [[...], ...19...], "values": ...},
    "proj_mlp":    {"indices": [[...], ...38...], "values": ...},
  }

Hook-to-weight-edit equivalence:
  ff / ffn_context: hook on net[0] output  → edit net[2].weight[:, channels]  (columns)
  proj_mlp:         hook on proj_mlp output → edit proj_mlp.weight[channels, :] (rows)

Default config matches test_erased.py best params:
  ff: all 19 transformer_blocks (0-18)
  ffn_context: all 19 transformer_blocks (0-18)
  proj_mlp: first 10 single_transformer_blocks (0-9)
  channel_num=15, scale=-4.0
"""
import argparse
import json
import os
from pathlib import Path
from typing import List

import torch
from diffusers import FluxKontextPipeline

SCRIPT_DIR = Path(__file__).parent
DEFAULT_MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"
DEFAULT_CACHE_DIR = f'{os.environ["HOME"]}/common_model/huggingface/hub'
DEFAULT_CHANNEL_DATA = str(SCRIPT_DIR / "channel_data_ti2i.json")
DEFAULT_FF_RANGE = "0,19"
DEFAULT_FFN_CONTEXT_RANGE = "0,19"
DEFAULT_PROJ_MLP_RANGE = "0,10"
DEFAULT_CHANNEL_NUM = 15
DEFAULT_SCALE = -4.0
DEFAULT_OUTPUT = str(SCRIPT_DIR / "ffn_channel_delta.pt")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export multi-layer FFN delta patch for FLUX.1-Kontext-dev TI2I. "
            "Reads channel_data_ti2i.json and builds weight deltas for ff, "
            "ffn_context, and proj_mlp layers."
        )
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--channel-data", default=DEFAULT_CHANNEL_DATA)
    parser.add_argument("--ff-range", default=DEFAULT_FF_RANGE,
                        help="START,END for transformer_blocks ff layers.")
    parser.add_argument("--ffn-context-range", default=DEFAULT_FFN_CONTEXT_RANGE,
                        help="START,END for transformer_blocks ff_context layers.")
    parser.add_argument("--proj-mlp-range", default=DEFAULT_PROJ_MLP_RANGE,
                        help="START,END for single_transformer_blocks proj_mlp layers.")
    parser.add_argument("--channel-num", type=int, default=DEFAULT_CHANNEL_NUM)
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def parse_range(s: str):
    parts = s.split(",")
    if len(parts) != 2:
        raise ValueError(f"Range must be 'START,END', got '{s}'")
    return range(int(parts[0]), int(parts[1]))


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
                    f"{layer_path}: channel index exceeds in_features={layer.in_features}"
                )
            original = layer.weight[:, channels].detach().cpu().float()
        else:  # rows
            if any(c >= layer.out_features for c in channels):
                raise ValueError(
                    f"{layer_path}: channel index exceeds out_features={layer.out_features}"
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
    ff_range = parse_range(args.ff_range)
    ffn_ctx_range = parse_range(args.ffn_context_range)
    proj_mlp_range = parse_range(args.proj_mlp_range)

    with open(args.channel_data, "r") as f:
        channel_data = json.load(f)

    ff_indices = channel_data["ff"]["indices"]
    ffn_ctx_indices = channel_data["ffn_context"]["indices"]
    proj_mlp_indices = channel_data["proj_mlp"]["indices"]

    assert len(ff_indices) >= ff_range.stop, \
        f"ff needs {ff_range.stop} layers, got {len(ff_indices)}"
    assert len(ffn_ctx_indices) >= ffn_ctx_range.stop, \
        f"ffn_context needs {ffn_ctx_range.stop} layers, got {len(ffn_ctx_indices)}"
    assert len(proj_mlp_indices) >= proj_mlp_range.stop, \
        f"proj_mlp needs {proj_mlp_range.stop} layers, got {len(proj_mlp_indices)}"

    print("Loading transformer (bfloat16, CPU)...")
    pipe = FluxKontextPipeline.from_pretrained(
        args.model_id,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        local_files_only=True,
    )
    transformer = pipe.transformer

    patches = []

    for i in ff_range:
        layer_path = f"transformer_blocks.{i}.ff.net.2"
        channels = ff_indices[i][: args.channel_num]
        entry = build_patch_entry(transformer, layer_path, "columns", channels, args.scale)
        patches.append(entry)
        print(f"  ff[{i:2d}] {layer_path}  channels[0:3]={channels[:3]}")

    for i in ffn_ctx_range:
        layer_path = f"transformer_blocks.{i}.ff_context.net.2"
        channels = ffn_ctx_indices[i][: args.channel_num]
        entry = build_patch_entry(transformer, layer_path, "columns", channels, args.scale)
        patches.append(entry)
        print(f"  ffn_context[{i:2d}] {layer_path}  channels[0:3]={channels[:3]}")

    for i in proj_mlp_range:
        layer_path = f"single_transformer_blocks.{i}.proj_mlp"
        channels = proj_mlp_indices[i][: args.channel_num]
        entry = build_patch_entry(transformer, layer_path, "rows", channels, args.scale)
        patches.append(entry)
        print(f"  proj_mlp[{i:2d}] {layer_path}  channels[0:3]={channels[:3]}")

    patch_data = {
        "format": "ffn_column_delta_v2",
        "base_model_id": args.model_id,
        "scale": args.scale,
        "channel_num": args.channel_num,
        "ff_range": [ff_range.start, ff_range.stop],
        "ffn_context_range": [ffn_ctx_range.start, ffn_ctx_range.stop],
        "proj_mlp_range": [proj_mlp_range.start, proj_mlp_range.stop],
        "patches": patches,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(patch_data, args.output)
    print(f"\nSaved {len(patches)}-layer delta patch to: {args.output}")
    print(f"  model: {args.model_id}  scale={args.scale}  channel_num={args.channel_num}")
    print(f"  ff={list(ff_range)}  ffn_context={list(ffn_ctx_range)}  "
          f"proj_mlp={list(proj_mlp_range)}")


if __name__ == "__main__":
    main()
