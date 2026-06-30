"""
Load FLUX.1-dev, apply multi-layer FFN delta patch, run UniErase-Bench T2I.

Supports patch formats:
  ffn_column_delta_v2 — list of per-layer patches (produced by export_ffn_channel_delta.py)
  ffn_column_delta_v1 — single-layer patch (backward compat)
"""
import argparse
import json
import os
from pathlib import Path

import torch
from diffusers import FluxPipeline

DEFAULT_MODEL_ID = "black-forest-labs/FLUX.1-dev"
DEFAULT_CACHE_DIR = f'{os.environ["HOME"]}/common_model/huggingface/'
SCRIPT_DIR = Path(__file__).parent
BENCHMARK_PATH = SCRIPT_DIR / "../../prompt_data/UniErase-Bench/t2i_benchmark.jsonl"
OUTPUT_DIR = SCRIPT_DIR / "results" / "ffn_delta"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load FLUX.1-dev, apply a saved multi-layer FFN delta patch, "
            "and evaluate on UniErase-Bench T2I."
        )
    )
    parser.add_argument("--delta-path", required=True)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device-map", default="balanced")
    parser.add_argument(
        "--torch-dtype", default="bfloat16",
        choices=["float16", "float32", "bfloat16"],
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--save-dir", default=None,
                        help="If set, save the edited pipeline to this directory.")
    return parser.parse_args()


def resolve_dtype(name: str):
    return {"float16": torch.float16, "float32": torch.float32,
            "bfloat16": torch.bfloat16}[name]


def get_module_by_path(root, path: str):
    m = root
    for part in path.split("."):
        m = m[int(part)] if part.isdigit() else getattr(m, part)
    return m


def apply_delta_patches(transformer, patch_data: dict):
    fmt = patch_data.get("format", "")
    if fmt == "ffn_column_delta_v1":
        patches = [patch_data]
    elif fmt == "ffn_column_delta_v2":
        patches = patch_data["patches"]
    else:
        raise ValueError(f"Unsupported patch format: {fmt!r}")

    for p in patches:
        layer = get_module_by_path(transformer, p["layer_path"])
        param = getattr(layer, p["param_name"])
        channels = p["channels"]
        delta = p["delta_slice"].to(device=param.device, dtype=param.dtype)
        original = p["original_slice"].to(device=param.device, dtype=param.dtype)
        mode = p.get("modify_mode", "columns")

        if mode == "columns":
            current = param[:, channels]
            if not torch.allclose(current, original, atol=1e-2, rtol=1e-2):
                raise RuntimeError(
                    f"Weight mismatch at {p['layer_path']} (columns). "
                    "Patch may already be applied or model differs."
                )
            with torch.no_grad():
                param[:, channels] += delta
        elif mode == "rows":
            current = param[channels, :]
            if not torch.allclose(current, original, atol=1e-2, rtol=1e-2):
                raise RuntimeError(
                    f"Weight mismatch at {p['layer_path']} (rows). "
                    "Patch may already be applied or model differs."
                )
            with torch.no_grad():
                param[channels, :] += delta
        else:
            raise ValueError(f"Unknown modify_mode: {mode!r}")

    n = len(patches)
    print(f"Applied {n} layer patch{'es' if n != 1 else ''}.")


def run_benchmark(pipe, output_dir: Path, benchmark_path: Path):
    benchmark_data = []
    with open(benchmark_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                benchmark_data.append(json.loads(line))
    print(f"Loaded {len(benchmark_data)} benchmark entries.")

    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "benchmark_meta.jsonl"

    completed = set()
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    if r.get("status") == "success":
                        completed.add(r["index"])
    if completed:
        print(f"Resuming: {len(completed)} done, "
              f"{len(benchmark_data) - len(completed)} remaining.")

    meta_file = open(meta_path, "a", encoding="utf-8")
    try:
        for idx, item in enumerate(benchmark_data):
            if idx in completed:
                continue

            source_id = item["source_id"]
            source_prompt = item["source_prompt"]
            print(f"[{idx+1}/{len(benchmark_data)}] sid={source_id} "
                  f"{source_prompt[:80]}...")

            try:
                with torch.no_grad():
                    image = pipe(
                        source_prompt,
                        generator=torch.Generator().manual_seed(source_id),
                        num_images_per_prompt=1,
                    ).images[0]
            except Exception as e:
                print(f"  Pipeline failed: {e}")
                meta_file.write(json.dumps({
                    "index": idx, "source_id": source_id,
                    "source_prompt": source_prompt,
                    "status": "pipeline_failed", "error": str(e),
                    "output_path": None,
                }, ensure_ascii=False) + "\n")
                meta_file.flush()
                continue

            out_path = output_dir / f"idx{idx:04d}_sid{source_id}.png"
            try:
                image.save(str(out_path))
                print(f"  Saved: {out_path}")
            except Exception as e:
                print(f"  Save failed: {e}")
                meta_file.write(json.dumps({
                    "index": idx, "source_id": source_id,
                    "source_prompt": source_prompt,
                    "status": "save_failed", "error": str(e),
                    "output_path": None,
                }, ensure_ascii=False) + "\n")
                meta_file.flush()
                continue

            meta_file.write(json.dumps({
                "index": idx, "source_id": source_id,
                "source_prompt": source_prompt,
                "status": "success", "error": None,
                "output_path": str(out_path),
            }, ensure_ascii=False) + "\n")
            meta_file.flush()
    finally:
        meta_file.close()

    all_records = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_records.append(json.loads(line))
    success = sum(1 for r in all_records if r.get("status") == "success")
    failed = sum(1 for r in all_records if r.get("status") not in ("success", None))
    print(f"\nDone. {success} succeeded, {failed} failed out of "
          f"{len(benchmark_data)} total.")


def main():
    args = parse_args()
    patch_data = torch.load(args.delta_path, map_location="cpu",
                            weights_only=False)
    model_id = args.model_id or patch_data.get("base_model_id", DEFAULT_MODEL_ID)
    torch_dtype = resolve_dtype(args.torch_dtype)

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    print(f"Loading pipeline (dtype={args.torch_dtype}, device_map={args.device_map})...")
    pipe = FluxPipeline.from_pretrained(
        model_id,
        cache_dir=args.cache_dir,
        device_map=args.device_map,
        torch_dtype=torch_dtype,
    )

    apply_delta_patches(pipe.transformer, patch_data)

    scale = patch_data.get("scale", "?")
    channel_num = patch_data.get("channel_num", "?")
    print(f"Model: {model_id}  delta: {args.delta_path}")
    print(f"scale={scale}  channel_num={channel_num}")

    if args.save_dir is not None:
        pipe.save_pretrained(args.save_dir)
        print(f"Saved edited pipeline to: {args.save_dir}")

    run_benchmark(pipe, Path(args.output_dir), Path(BENCHMARK_PATH))


if __name__ == "__main__":
    main()
