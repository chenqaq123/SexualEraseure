import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import json
import torch
from diffusers import FluxKontextPipeline, FluxTransformer2DModel, BitsAndBytesConfig
from diffusers.utils import load_image

MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"
CACHE_DIR = "/home/chenguanxu/common_model/huggingface/hub"

BENCHMARK_PATH = "/home/chenguanxu/nas/T2I/Concept_Erasure/prompt_data/UniErase-Bench/ti2i_benchmark.jsonl"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

transformer = FluxTransformer2DModel.from_pretrained(
    MODEL_ID,
    subfolder="transformer",
    quantization_config=nf4_config,
    cache_dir=CACHE_DIR,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    local_files_only=True,
)

pipe = FluxKontextPipeline.from_pretrained(
    MODEL_ID,
    transformer=transformer,
    cache_dir=CACHE_DIR,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)

pipe.to("cuda:0")

# Read benchmark data
benchmark_data = []
with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            benchmark_data.append(json.loads(line))

print(f"Loaded {len(benchmark_data)} test cases from benchmark.")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

meta_path = os.path.join(OUTPUT_DIR, "benchmark_meta.jsonl")

# --- Resume: load already-completed indices ---
completed_indices = set()

# 1) Check existing result images
if os.path.exists(OUTPUT_DIR):
    for fname in os.listdir(OUTPUT_DIR):
        if fname.startswith("idx") and fname.endswith(".png"):
            try:
                idx = int(fname.split("_")[0].replace("idx", ""))
                completed_indices.add(idx)
            except ValueError:
                pass

# 2) Check existing meta file for failed cases (so we don't retry image_load_failed either)
if os.path.exists(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                completed_indices.add(rec["index"])

if completed_indices:
    print(f"Resuming: {len(completed_indices)} cases already completed, will skip them.")
else:
    print("Starting from scratch.")

# Run inference — open meta file in append mode
with open(meta_path, "a", encoding="utf-8") as meta_f:
    for idx, item in enumerate(benchmark_data):
        # Skip already completed
        if idx in completed_indices:
            continue

        source_id = item["source_id"]
        instruction = item["instruction"]
        reference_image_path = item["reference_image_path"]
        instruction_is_malicious = item.get("instruction_is_malicious", None)

        print(f"[{idx+1}/{len(benchmark_data)}] source_id={source_id}, malicious={instruction_is_malicious}, instruction={instruction[:80]}...")

        result_record = {
            "index": idx,
            "source_id": source_id,
            "instruction": instruction,
            "instruction_is_malicious": instruction_is_malicious,
            "reference_image_path": reference_image_path,
        }

        # Load reference image
        try:
            image = load_image(reference_image_path).convert("RGB").resize((1024, 1024))
        except Exception as e:
            print(f"  Failed to load image {reference_image_path}: {e}")
            result_record.update(status="image_load_failed", error=str(e), output_path=None)
            meta_f.write(json.dumps(result_record, ensure_ascii=False) + "\n")
            meta_f.flush()
            continue

        # Run pipeline
        try:
            with torch.no_grad():
                result = pipe(
                    image=image,
                    prompt=instruction,
                    guidance_scale=5,
                    generator=torch.Generator().manual_seed(42),
                ).images[0]
        except Exception as e:
            print(f"  Pipeline failed: {e}")
            result_record.update(status="pipeline_failed", error=str(e), output_path=None)
            meta_f.write(json.dumps(result_record, ensure_ascii=False) + "\n")
            meta_f.flush()
            continue

        # Save result image
        out_filename = f"idx{idx:04d}_sid{source_id}.png"
        out_path = os.path.join(OUTPUT_DIR, out_filename)
        result.save(out_path)
        print(f"  Saved: {out_path}")

        result_record.update(status="success", error=None, output_path=out_path)
        meta_f.write(json.dumps(result_record, ensure_ascii=False) + "\n")
        meta_f.flush()

# Summary
all_records = []
if os.path.exists(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_records.append(json.loads(line))

success_count = sum(1 for r in all_records if r.get("status") == "success")
fail_count = sum(1 for r in all_records if r.get("status") not in ("success", None))
total_done = len(all_records)
remaining = len(benchmark_data) - total_done
print(f"\nProgress: {total_done}/{len(benchmark_data)} done ({remaining} remaining).")
print(f"  Success: {success_count}, Failed: {fail_count}")
print(f"Results saved to: {OUTPUT_DIR}")
print(f"Metadata saved to: {meta_path}")
