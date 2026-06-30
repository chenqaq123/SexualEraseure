import os
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch
from diffusers import HunyuanVideo15Pipeline
from diffusers.utils import export_to_video
from pathlib import Path

# ============================================================
# Config
# ============================================================
MODEL_ID = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
CACHE_DIR = "/home/mlsnrs/common_model/huggingface/hub"
BENCHMARK_PATH = "/home/mlsnrs/data/cgx/cgx_nas/T2I/Concept_Erasure/prompt_data/UniErase-Bench/t2v_benchmark.jsonl"
CHANNEL_DATA_PATH = "/home/mlsnrs/data/cgx/cgx_nas/T2I/model_scripts/HunYuanVideo/result_ff_context.json"

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "erased_results"

DTYPE = torch.bfloat16

# Best erasure params from grid search
ERASURE_FACTOR = -4.0
CHANNEL_NUM = 10
START = 0
END = 15
INTER_RANGE = range(20, 30)

HEIGHT = 352
WIDTH = 640
NUM_FRAMES = 49
NUM_INFERENCE_STEPS = 50
FPS = 8

# ============================================================
# Load Pipeline
# ============================================================
print("Loading pipeline...")
pipe = HunyuanVideo15Pipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    cache_dir=CACHE_DIR,
    device_map="balanced",
)
pipe.vae.enable_tiling()
pipe.enable_attention_slicing()
print("Pipeline loaded.")

# ============================================================
# Load Channel Importance Data & Register Erasure Hooks
# ============================================================
print(f"Loading channel data from {CHANNEL_DATA_PATH}")
with open(CHANNEL_DATA_PATH, "r") as f:
    channel_data = json.load(f)
all_topk_indices = channel_data["indices"]
print(f"Loaded channel data: {len(all_topk_indices)} layers")

# Select modules and channel indices for INTER_RANGE
modules = [pipe.transformer.transformer_blocks[i].ff_context.net[0] for i in INTER_RANGE]
topk_indices = [all_topk_indices[i] for i in INTER_RANGE]

# Register erasure hooks
def get_hook(layer_idx):
    top_score_channels = topk_indices[layer_idx][START:END]

    def hook(module, input, output):
        with torch.no_grad():
            new_output = output.clone()
            new_output[:, :, top_score_channels] = ERASURE_FACTOR * new_output[:, :, top_score_channels]
        return new_output

    return hook

hooks = []
for i, module in enumerate(modules):
    hooks.append(module.register_forward_hook(get_hook(i)))

print(f"Erasure hooks registered: {len(hooks)} hooks on layers {INTER_RANGE.start}-{INTER_RANGE.stop}")
print(f"  channel_num={CHANNEL_NUM}, erasure_factor={ERASURE_FACTOR}, slice=[{START}:{END}]")

# ============================================================
# Read Benchmark Data
# ============================================================
benchmark_data = []
with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            benchmark_data.append(json.loads(line))

print(f"Loaded {len(benchmark_data)} test cases from benchmark.")

# ============================================================
# Create Output Directory
# ============================================================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
meta_path = OUTPUT_DIR / "benchmark_meta.jsonl"

# ============================================================
# Resume: Load Already Completed Indices
# ============================================================
completed_indices = set()
if meta_path.exists():
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r["status"] == "success":
                    completed_indices.add(r["index"])
    print(f"Resuming: {len(completed_indices)} already completed, {len(benchmark_data) - len(completed_indices)} remaining.")

# ============================================================
# Run Inference
# ============================================================
meta_file = open(meta_path, "a", encoding="utf-8")

results = []
skipped = 0
for idx, item in enumerate(benchmark_data):
    source_id = item["source_id"]
    source_prompt = item["source_prompt"]
    instruction_is_malicious = item.get("instruction_is_malicious", None)

    # Skip already completed
    if idx in completed_indices:
        skipped += 1
        continue

    print(f"[{idx+1}/{len(benchmark_data)}] source_id={source_id}, malicious={instruction_is_malicious}, prompt={source_prompt[:80]}...")

    try:
        with torch.no_grad():
            video_frames = pipe(
                prompt=source_prompt,
                height=HEIGHT,
                width=WIDTH,
                num_frames=NUM_FRAMES,
                num_inference_steps=NUM_INFERENCE_STEPS,
                generator=torch.Generator().manual_seed(42),
            ).frames[0]
    except Exception as e:
        print(f"  Pipeline failed: {e}")
        record = {
            "index": idx,
            "source_id": source_id,
            "source_prompt": source_prompt,
            "instruction_is_malicious": instruction_is_malicious,
            "status": "pipeline_failed",
            "error": str(e),
            "output_path": None,
        }
        results.append(record)
        meta_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        meta_file.flush()
        torch.cuda.empty_cache()
        continue

    # Save result video
    out_filename = f"idx{idx:04d}_sid{source_id}.mp4"
    out_path = OUTPUT_DIR / out_filename
    try:
        export_to_video(video_frames, str(out_path), fps=FPS)
        print(f"  Saved: {out_path}")
    except Exception as e:
        print(f"  Failed to save video: {e}")
        record = {
            "index": idx,
            "source_id": source_id,
            "source_prompt": source_prompt,
            "instruction_is_malicious": instruction_is_malicious,
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
        "source_prompt": source_prompt,
        "instruction_is_malicious": instruction_is_malicious,
        "status": "success",
        "error": None,
        "output_path": str(out_path),
    }
    results.append(record)
    meta_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    meta_file.flush()
    torch.cuda.empty_cache()

# ============================================================
# Cleanup
# ============================================================
meta_file.close()

for hook in hooks:
    hook.remove()
print("Erasure hooks removed.")

success_count = sum(1 for r in results if r["status"] == "success")
fail_count = sum(1 for r in results if r["status"] != "success")
print(f"\nDone. {success_count} succeeded, {fail_count} failed, {skipped} skipped out of {len(benchmark_data)} total.")
print(f"Results saved to: {OUTPUT_DIR}")
print(f"Metadata saved to: {meta_path}")
