import argparse
import os
from typing import List

import torch
from diffusers import UNet2DConditionModel


DEFAULT_MODEL_ID = "CompVis/stable-diffusion-v1-4"
DEFAULT_CACHE_DIR = f'{os.environ["HOME"]}/common_model/huggingface/'
DEFAULT_LAYER_PATH = "up_blocks.1.attentions.0.transformer_blocks.0.ff.net.2"
DEFAULT_CHANNELS = [5035, 1761, 4499, 185, 4920]
DEFAULT_SCALE = -2.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export the exact parameter delta that is equivalent to scaling "
            "specific FFN output channels in test_erase.py."
        )
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--layer-path", default=DEFAULT_LAYER_PATH)
    parser.add_argument(
        "--channels",
        default=",".join(str(x) for x in DEFAULT_CHANNELS),
        help="Comma-separated FFN channel indices.",
    )
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    parser.add_argument(
        "--output",
        default="ffn_channel_delta.pt",
        help="Path to save the delta patch.",
    )
    return parser.parse_args()


def parse_channels(channels_text: str) -> List[int]:
    channels = [int(part.strip()) for part in channels_text.split(",") if part.strip()]
    if not channels:
        raise ValueError("No channels were provided.")
    if len(channels) != len(set(channels)):
        raise ValueError(f"Duplicate channels found: {channels}")
    return channels


def get_module_by_path(root_module, path: str):
    module = root_module
    for part in path.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def build_delta_patch(args) -> dict:
    channels = parse_channels(args.channels)

    unet = UNet2DConditionModel.from_pretrained(
        args.model_id,
        subfolder="unet",
        cache_dir=args.cache_dir,
        torch_dtype=torch.float32,
    )
    target_layer = get_module_by_path(unet, args.layer_path)

    if not isinstance(target_layer, torch.nn.Linear):
        raise TypeError(
            f"Expected a Linear layer at {args.layer_path}, got {type(target_layer)}."
        )

    if any(channel < 0 or channel >= target_layer.in_features for channel in channels):
        raise ValueError(
            f"Channel indices must be in [0, {target_layer.in_features - 1}], got {channels}."
        )

    with torch.no_grad():
        original_slice = target_layer.weight[:, channels].detach().cpu().float()
        edited_slice = original_slice * args.scale
        delta_slice = edited_slice - original_slice

    return {
        "format": "ffn_column_delta_v1",
        "base_model_id": args.model_id,
        "unet_subfolder": "unet",
        "layer_path": args.layer_path,
        "param_name": "weight",
        "channels": channels,
        "scale": float(args.scale),
        "delta_slice": delta_slice.contiguous(),
        "original_slice": original_slice.contiguous(),
        "notes": (
            "This patch is exactly equivalent to multiplying the hooked output of "
            "ff.net[0] by `scale`, because the next layer is linear: y = W(Dx) + b."
        ),
    }


def main():
    args = parse_args()
    patch = build_delta_patch(args)
    torch.save(patch, args.output)

    print(f"Saved delta patch to: {args.output}")
    print(f"Base model: {patch['base_model_id']}")
    print(f"Target layer: {patch['layer_path']}")
    print(f"Channels: {patch['channels']}")
    print(f"Scale: {patch['scale']}")
    print(f"Delta slice shape: {tuple(patch['delta_slice'].shape)}")


if __name__ == "__main__":
    main()
