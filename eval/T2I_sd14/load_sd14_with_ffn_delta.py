import argparse
import os

import torch
from diffusers import StableDiffusionPipeline


DEFAULT_CACHE_DIR = f'{os.environ["HOME"]}/common_model/huggingface/'


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load the original SD1.4 pipeline, apply a saved FFN delta patch, "
            "and optionally save the edited model."
        )
    )
    parser.add_argument("--delta-path", required=True)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device to move the pipeline onto after loading.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="float32",
        choices=["float16", "float32", "bfloat16"],
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Override the base model id stored in the delta file.",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="If set, save the edited pipeline to this directory.",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str):
    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def get_module_by_path(root_module, path: str):
    module = root_module
    for part in path.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def apply_delta_patch(pipe, patch: dict):
    if patch.get("format") != "ffn_column_delta_v1":
        raise ValueError(f"Unsupported delta format: {patch.get('format')}")

    target_layer = get_module_by_path(pipe.unet, patch["layer_path"])
    param = getattr(target_layer, patch["param_name"])
    channels = patch["channels"]
    delta_slice = patch["delta_slice"].to(device=param.device, dtype=param.dtype)
    original_slice = patch["original_slice"].to(device=param.device, dtype=param.dtype)

    if delta_slice.shape != (param.shape[0], len(channels)):
        raise ValueError(
            f"Delta shape mismatch: expected {(param.shape[0], len(channels))}, "
            f"got {tuple(delta_slice.shape)}."
        )

    current_slice = param[:, channels]
    if not torch.allclose(current_slice, original_slice):
        raise RuntimeError(
            "Weight mismatch: current weight columns differ from the original slice "
            "stored in the delta. The patch may have already been applied, or the "
            "base model is different from the one used to export the delta."
        )

    with torch.no_grad():
        param[:, channels] += delta_slice


def main():
    args = parse_args()
    patch = torch.load(args.delta_path, map_location="cpu")
    model_id = args.model_id or patch["base_model_id"]
    torch_dtype = resolve_dtype(args.torch_dtype)

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        cache_dir=args.cache_dir,
        torch_dtype=torch_dtype,
    )
    apply_delta_patch(pipe, patch)
    pipe = pipe.to(args.device)

    print(f"Loaded base model: {model_id}")
    print(f"Applied delta: {args.delta_path}")
    print(f"Target layer: {patch['layer_path']}")
    print(f"Channels: {patch['channels']}")
    print(f"Scale: {patch['scale']}")
    print(f"Pipeline device: {args.device}")
    print(f"Pipeline dtype: {torch_dtype}")

    if args.save_dir is not None:
        pipe.save_pretrained(args.save_dir)
        print(f"Saved edited pipeline to: {args.save_dir}")

    results_dir = './results/nudity/'
    os.makedirs(results_dir, exist_ok=True)

    import pandas as pd
    df_i2p = pd.read_json(
        '/home/chenguanxu/nas/T2I/Concept_Erasure/prompt_data/UniErase-Bench/t2i_benchmark.jsonl',
        lines=True,
    )

    save_dir = f'{results_dir}/UniErase-Bench/'
    os.makedirs(save_dir, exist_ok=True)
    for _, row in df_i2p.iterrows():
        prompt_text = row['source_prompt']
        sid = row['source_id']
        seed = row.get('evaluation_seed', sid) or sid
        guidance = row.get('evaluation_guidance', 7.5) or 7.5

        save_path = f'{save_dir}{sid}.png'
        if os.path.exists(save_path):
            print(f'[skip] {save_path}')
            continue
        print(f'[{sid}] {prompt_text[:80]}... seed={seed} guidance={guidance}')
        generator = torch.Generator(device=args.device).manual_seed(int(seed))
        image = pipe(
            prompt_text,
            generator=generator,
            num_images_per_prompt=1,
            guidance_scale=float(guidance),
        ).images[0]
        image.save(save_path)




if __name__ == "__main__":
    main()
