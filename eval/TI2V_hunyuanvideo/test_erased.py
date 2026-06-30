"""
Run TI2V UniErase-Bench with erasure hooks applied.
Best parameters will be set at the top; adjust after grid search.
"""
import os
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch
from diffusers import HunyuanVideo15ImageToVideoPipeline
from diffusers.utils import export_to_video
from pathlib import Path
from PIL import Image

# ============================================================
# Config
# ============================================================
MODEL_ID = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v"
CACHE_DIR = "/home/mlsnrs/common_model/huggingface/hub"
BENCHMARK_PATH = "/home/mlsnrs/data/cgx/cgx_nas/T2I/Concept_Erasure/prompt_data/UniErase-Bench/ti2v_benchmark.jsonl"
BENCHMARK_IMG_DIR = "/home/mlsnrs/data/cgx/cgx_nas/T2I/Concept_Erasure/prompt_data/UniErase-Bench/reference_images/ti2v_benchmark"
CHANNEL_DATA_PATH = Path(__file__).parent / "channel_data_ti2v.json"

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "erased_results"

DTYPE = torch.bfloat16

# Best erasure params — adjust after grid search
ERASURE_FACTOR = -4.0
START = 0
END = 15
INTER_RANGE = range(10, 30)  # TODO: update after grid search
MODULE_TYPES = ["ffn_context", "ff"]  # both applied simultaneously

NUM_FRAMES = 49
NUM_INFERENCE_STEPS = 50
FPS = 8

PLACEHOLDER_IMAGE = Image.new("RGB", (640, 480), (128, 128, 128))

# ============================================================
# Load Pipeline
# ============================================================
print("Loading TI2V pipeline...")
pipe = HunyuanVideo15ImageToVideoPipeline.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    device_map="balanced",
    torch_dtype=DTYPE,
)
pipe.vae.enable_tiling()
print("Pipeline loaded.")

# ============================================================
# Load Channel Data & Register Hooks (both ff + ffn_context)
# ============================================================
print(f"Loading channel data from {CHANNEL_DATA_PATH}")
with open(CHANNEL_DATA_PATH, "r") as f:
    channel_data = json.load(f)

hooks = []
for module_type in MODULE_TYPES:
    topk_indices_all = channel_data[module_type]["indices"]
    if module_type == "ffn_context":
        modules = [pipe.transformer.transformer_blocks[i].ff_context.net[0] for i in INTER_RANGE]
    else:
        modules = [pipe.transformer.transformer_blocks[i].ff.net[0] for i in INTER_RANGE]
    topk_indices = [topk_indices_all[i] for i in INTER_RANGE]

    def get_hook(layer_idx, mt=module_type, tk=topk_indices):
        top_score_channels = tk[layer_idx][START:END]

        def hook(_module, _input, output):
            # Debug: check if indices are valid
            if output.shape[-1] <= max(top_score_channels):
                print(f"WARNING: Channel index out of range!")
                print(f"  output.shape: {output.shape}")
                print(f"  max channel index: {max(top_score_channels)}")
                print(f"  module_type: {mt}, layer_idx: {layer_idx}")
                return output  # Skip modification if indices invalid
            output[:, :, top_score_channels] *= ERASURE_FACTOR
            return output

        return hook

    for i, module in enumerate(modules):
        hooks.append(module.register_forward_hook(get_hook(i)))

print(f"Erasure hooks registered: {len(hooks)} hooks on layers {INTER_RANGE.start}-{INTER_RANGE.stop}")
print(f"  module_types={MODULE_TYPES}, erasure_factor={ERASURE_FACTOR}, slice=[{START}:{END}], channel_num={END-START}")

# ============================================================
# Read Benchmark Data
# ============================================================
benchmark_data = []
with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            item = json.loads(line)
            # Fix reference_image_path to local path
            fname = Path(item["reference_image_path"]).name
            item["reference_image_path"] = str(Path(BENCHMARK_IMG_DIR) / fname)
            benchmark_data.append(item)

print(f"Loaded {len(benchmark_data)} test cases from benchmark.")

# ============================================================
# Create Output & Resume
# ============================================================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
meta_path = OUTPUT_DIR / "benchmark_meta.jsonl"

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
    instruction = item["instruction"]
    reference_image_path = item["reference_image_path"]
    instruction_is_malicious = item.get("instruction_is_malicious", None)
    reference_image_prompt_is_malicious = item.get("reference_image_prompt_is_malicious", None)

    if idx in completed_indices:
        skipped += 1
        continue

    print(f"[{idx+1}/{len(benchmark_data)}] source_id={source_id}, malicious=inst:{instruction_is_malicious}/ref:{reference_image_prompt_is_malicious}, instruction={instruction[:80]}...")

    # Load reference image
    try:
        image = Image.open(reference_image_path).convert("RGB")
    except Exception as e:
        print(f"  Failed to load image: {e}")
        record = {
            "index": idx, "source_id": source_id, "instruction": instruction,
            "instruction_is_malicious": instruction_is_malicious,
            "reference_image_prompt_is_malicious": reference_image_prompt_is_malicious,
            "reference_image_path": reference_image_path,
            "status": "image_load_failed", "error": str(e), "output_path": None,
        }
        results.append(record)
        meta_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        meta_file.flush()
        continue

    try:
        with torch.no_grad():
            video_frames = pipe(
                prompt=instruction,
                image=image,
                num_frames=NUM_FRAMES,
                num_inference_steps=NUM_INFERENCE_STEPS,
                generator=torch.Generator().manual_seed(42),
            ).frames[0]
    except Exception as e:
        print(f"  Pipeline failed: {e}")
        record = {
            "index": idx, "source_id": source_id, "instruction": instruction,
            "instruction_is_malicious": instruction_is_malicious,
            "reference_image_prompt_is_malicious": reference_image_prompt_is_malicious,
            "reference_image_path": reference_image_path,
            "status": "pipeline_failed", "error": str(e), "output_path": None,
        }
        results.append(record)
        meta_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        meta_file.flush()
        torch.cuda.empty_cache()
        continue

    out_filename = f"idx{idx:04d}_sid{source_id}.mp4"
    out_path = OUTPUT_DIR / out_filename
    try:
        export_to_video(video_frames, str(out_path), fps=FPS)
        print(f"  Saved: {out_path}")
    except Exception as e:
        print(f"  Failed to save video: {e}")
        record = {
            "index": idx, "source_id": source_id, "instruction": instruction,
            "instruction_is_malicious": instruction_is_malicious,
            "reference_image_prompt_is_malicious": reference_image_prompt_is_malicious,
            "reference_image_path": reference_image_path,
            "status": "save_failed", "error": str(e), "output_path": None,
        }
        results.append(record)
        meta_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        meta_file.flush()
        continue

    record = {
        "index": idx, "source_id": source_id, "instruction": instruction,
        "instruction_is_malicious": instruction_is_malicious,
        "reference_image_prompt_is_malicious": reference_image_prompt_is_malicious,
        "reference_image_path": reference_image_path,
        "status": "success", "error": None, "output_path": str(out_path),
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
