import os
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

MODEL_ID = "zai-org/CogVideoX1.5-5B-I2V"
CACHE_DIR = "/home/mlsnrs/common_model/huggingface/hub"
BENCHMARK_PATH = "/home/mlsnrs/data/cgx/cgx_nas/T2I/Concept_Erasure/prompt_data/UniErase-Bench/ti2v_benchmark.jsonl"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

DTYPE = torch.bfloat16
pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    device_map="balanced",
    torch_dtype=DTYPE,
)
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

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

# Resume: load already completed indices
meta_path = os.path.join(OUTPUT_DIR, "benchmark_meta.jsonl")
completed_indices = set()
if os.path.exists(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r["status"] == "success":
                    completed_indices.add(r["index"])
    print(f"Resuming: {len(completed_indices)} already completed, {len(benchmark_data) - len(completed_indices)} remaining.")

# Open metadata file in append mode
meta_file = open(meta_path, "a", encoding="utf-8")

# Run inference
results = []
skipped = 0
for idx, item in enumerate(benchmark_data):
    source_id = item["source_id"]
    instruction = item["instruction"]
    reference_image_path = item["reference_image_path"]
    instruction_is_malicious = item.get("instruction_is_malicious", None)
    reference_image_prompt_is_malicious = item.get("reference_image_prompt_is_malicious", None)

    # Skip already completed
    if idx in completed_indices:
        skipped += 1
        continue

    print(f"[{idx+1}/{len(benchmark_data)}] source_id={source_id}, malicious=inst:{instruction_is_malicious}/ref:{reference_image_prompt_is_malicious}, instruction={instruction[:80]}...")

    # Load reference image
    try:
        image = load_image(reference_image_path).convert("RGB")
    except Exception as e:
        print(f"  Failed to load image {reference_image_path}: {e}")
        record = {
            "index": idx,
            "source_id": source_id,
            "instruction": instruction,
            "instruction_is_malicious": instruction_is_malicious,
            "reference_image_prompt_is_malicious": reference_image_prompt_is_malicious,
            "reference_image_path": reference_image_path,
            "status": "image_load_failed",
            "error": str(e),
            "output_path": None,
        }
        results.append(record)
        meta_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        meta_file.flush()
        continue

    # Run pipeline
    try:
        with torch.no_grad():
            video_frames = pipe(
                prompt=instruction,
                image=image,
                height=352,
                width=640,
                num_videos_per_prompt=1,
                num_inference_steps=50,
                num_frames=49,
                guidance_scale=6,
                generator=torch.Generator().manual_seed(42),
            ).frames[0]
    except Exception as e:
        print(f"  Pipeline failed: {e}")
        record = {
            "index": idx,
            "source_id": source_id,
            "instruction": instruction,
            "instruction_is_malicious": instruction_is_malicious,
            "reference_image_prompt_is_malicious": reference_image_prompt_is_malicious,
            "reference_image_path": reference_image_path,
            "status": "pipeline_failed",
            "error": str(e),
            "output_path": None,
        }
        results.append(record)
        meta_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        meta_file.flush()
        continue

    # Save result video
    out_filename = f"idx{idx:04d}_sid{source_id}.mp4"
    out_path = os.path.join(OUTPUT_DIR, out_filename)
    try:
        export_to_video(video_frames, out_path, fps=8)
        print(f"  Saved: {out_path}")
    except Exception as e:
        print(f"  Failed to save video: {e}")
        record = {
            "index": idx,
            "source_id": source_id,
            "instruction": instruction,
            "instruction_is_malicious": instruction_is_malicious,
            "reference_image_prompt_is_malicious": reference_image_prompt_is_malicious,
            "reference_image_path": reference_image_path,
            "status": "save_failed",
            "error": str(e),
            "output_path": None,
        }
        results.append(record)
        meta_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        meta_file.flush()
        continue

    record = {
        "index": idx,
        "source_id": source_id,
        "instruction": instruction,
        "instruction_is_malicious": instruction_is_malicious,
        "reference_image_prompt_is_malicious": reference_image_prompt_is_malicious,
        "reference_image_path": reference_image_path,
        "status": "success",
        "error": None,
        "output_path": out_path,
    }
    results.append(record)
    meta_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    meta_file.flush()

meta_file.close()

success_count = sum(1 for r in results if r["status"] == "success")
fail_count = sum(1 for r in results if r["status"] != "success")
print(f"\nDone. {success_count} succeeded, {fail_count} failed, {skipped} skipped out of {len(benchmark_data)} total.")
print(f"Results saved to: {OUTPUT_DIR}")
print(f"Metadata saved to: {meta_path}")
