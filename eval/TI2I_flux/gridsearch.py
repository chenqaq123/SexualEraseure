"""
Grid search for TI2I (FluxKontextPipeline) erasure parameters.
Searches over: inter_range for transformer_blocks only (ff + ffn_context).

Fixed parameters:
- erasure_factor: -4.0
- start:end: 0-15 (top-15 channels)
- ff and ffn_context are applied together with the same inter_range
- proj_mlp is NOT erased in this search

Flux architecture:
- 19 transformer_blocks (0-18): ff.net[0] + ff_context.net[0]
- 38 single_transformer_blocks (0-37): proj_mlp (not used in this search)
"""
import os
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import csv
import itertools
import torch
from diffusers import FluxKontextPipeline, FluxTransformer2DModel, BitsAndBytesConfig
from diffusers.utils import load_image
from pathlib import Path
from PIL import Image

# ============================================================
# Config
# ============================================================
DTYPE = torch.bfloat16
CACHE_DIR = "/home/mlsnrs/common_model/huggingface/hub"
MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"
CHANNEL_DATA_PATH = Path(__file__).parent / "channel_data_ti2i.json"
BENCHMARK_PATH = "/home/mlsnrs/data/cgx/cgx_nas/T2I/Concept_Erasure/prompt_data/UniErase-Bench/ti2i_benchmark.jsonl"

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "gridsearch_ti2i"

# Fixed parameters
ERASURE_FACTOR = -4.0
START = 0
END = 15
NUM_IMAGES_PER_TEST = 5  # Generate 5 images per test case

# Placeholder image for Kontext pipeline
PLACEHOLDER_IMAGE = Image.new("RGB", (1024, 1024), (128, 128, 128))

# ============================================================
# Grid Search Space
# Only search transformer_blocks (ff + ffn_context), not proj_mlp
# ============================================================

# Inter-range options for transformer_blocks (19 layers: 0-18)
TRANSFORMER_BLOCK_RANGES = [
    range(0, 19),   # All layers (0-18)
    range(0, 10),   # First 10 layers (0-9)
    range(10, 19),  # Last 10 layers (10-18)
]

# Total: 3 combinations to test

# ============================================================
# Load Pipeline
# ============================================================
print("Loading FluxKontextPipeline...")
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
print("Pipeline loaded.")

# ============================================================
# Load Channel Importance Data
# ============================================================
print(f"Loading channel data from {CHANNEL_DATA_PATH}")
with open(CHANNEL_DATA_PATH, "r") as f:
    channel_data = json.load(f)

# Verify channel data contains all required module types
MODULE_TYPES = ["ff", "ffn_context", "proj_mlp"]
for mtype in MODULE_TYPES:
    assert mtype in channel_data, f"Missing module type '{mtype}' in channel data"
    print(f"  {mtype}: {len(channel_data[mtype]['indices'])} layers, {len(channel_data[mtype]['indices'][0])} channels per layer")

# ============================================================
# Load Test Cases from UniErase-Bench
# ============================================================
def load_test_cases():
    """Load all malicious test cases from UniErase-Bench."""
    test_cases = []
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                if item.get("instruction_is_malicious", False):
                    img_path = item["reference_image_path"]
                    try:
                        img = load_image(img_path).convert("RGB").resize((1024, 1024))
                        test_cases.append({
                            "instruction": item["instruction"],
                            "image": img,
                            "reference_image_path": img_path
                        })
                    except Exception as e:
                        print(f"  WARNING: Failed to load {img_path}: {e}, skipping.")
                        continue
    return test_cases

print("Loading test cases from UniErase-Bench...")
test_cases = load_test_cases()
print(f"Loaded {len(test_cases)} test cases")


# ============================================================
# Generate All Combinations
# ============================================================
combos = []
for tf_range in TRANSFORMER_BLOCK_RANGES:
    combos.append((tf_range,))  # Only transformer_blocks range

print(f"Total parameter combinations: {len(combos)}")

# ============================================================
# Output
# ============================================================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
meta_path = OUTPUT_DIR / "meta.jsonl"
summary_path = OUTPUT_DIR / "summary.csv"

with open(summary_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["combo_idx", "erasure_factor", "start", "end", "channel_num",
                     "tf_start", "tf_end"])
    for i, (tf_range,) in enumerate(combos):
        writer.writerow([i, ERASURE_FACTOR, START, END, END - START,
                        tf_range.start, tf_range.stop])
print(f"Summary CSV saved to {summary_path}")

# ============================================================
# Resume
# ============================================================
completed_combos = set()
if meta_path.exists():
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r["status"] == "success":
                    completed_combos.add(r["combo_idx"])
    print(f"Resuming: {len(completed_combos)} already completed, {len(combos) - len(completed_combos)} remaining.")

# ============================================================
# Helper: Get modules by inter_range
# ============================================================
def get_modules(tf_range):
    """Get ff and ffn_context modules for given range."""
    ff_modules = [pipe.transformer.transformer_blocks[i].ff.net[0] for i in tf_range]
    ffn_context_modules = [pipe.transformer.transformer_blocks[i].ff_context.net[0] for i in tf_range]
    return ff_modules, ffn_context_modules

# ============================================================
# Grid Search Loop
# ============================================================
meta_file = open(meta_path, "a", encoding="utf-8")

for combo_idx, (tf_range,) in enumerate(combos):
    if combo_idx in completed_combos:
        continue

    channel_num = END - START
    combo_tag = f"ef{ERASURE_FACTOR}_ch{channel_num}_tf{tf_range.start}_{tf_range.stop}"

    print(f"\n[{combo_idx+1}/{len(combos)}] {combo_tag}")
    print(f"  transformer_blocks: {tf_range.start}-{tf_range.stop} (no proj_mlp)")

    # Select modules and channel indices
    ff_topk_indices = [channel_data["ff"]["indices"][i] for i in range(19)]  # All 19 layers
    ffn_context_topk_indices = [channel_data["ffn_context"]["indices"][i] for i in range(19)]

    # Get modules for current inter_range
    ff_modules, ffn_context_modules = get_modules(tf_range)

    # Get corresponding topk indices for selected layers
    ff_indices = [ff_topk_indices[i] for i in tf_range]
    ffn_context_indices = [ffn_context_topk_indices[i] for i in tf_range]

    # Define hooks
    def make_ff_hook(layer_idx):
        top_channels = ff_indices[layer_idx][START:END]
        def hook(module, input, output):
            with torch.no_grad():
                output[:, :, top_channels] *= ERASURE_FACTOR
            return output
        return hook

    def make_ffn_context_hook(layer_idx):
        top_channels = ffn_context_indices[layer_idx][START:END]
        def hook(module, input, output):
            with torch.no_grad():
                output[:, :, top_channels] *= ERASURE_FACTOR
            return output
        return hook

    # Register hooks for ff and ffn_context only
    hooks = []
    for i, module in enumerate(ff_modules):
        hooks.append(module.register_forward_hook(make_ff_hook(i)))
    for i, module in enumerate(ffn_context_modules):
        hooks.append(module.register_forward_hook(make_ffn_context_hook(i)))

    # Generate images for each test case
    combo_success = 0
    combo_fail = 0
    total_images = len(test_cases) * NUM_IMAGES_PER_TEST

    for case_idx, test_case in enumerate(test_cases):
        instruction = test_case["instruction"]
        ref_image = test_case["image"]

        for img_idx in range(NUM_IMAGES_PER_TEST):
            seed = case_idx * NUM_IMAGES_PER_TEST + img_idx
            out_name = f"{combo_tag}_case{case_idx}_img{img_idx}_s{seed}.png"
            out_path = OUTPUT_DIR / out_name

            if out_path.exists():
                combo_success += 1
                continue

            try:
                with torch.no_grad():
                    generator = torch.Generator().manual_seed(seed)
                    result = pipe(
                        image=ref_image,
                        prompt=instruction,
                        guidance_scale=5,
                        generator=generator,
                    ).images[0]
                result.save(str(out_path))
                combo_success += 1
                print(f"  Saved: {out_name}")
            except Exception as e:
                combo_fail += 1
                print(f"  Failed: {out_name} - {e}")

            # Aggressive memory cleanup
            del result, generator
            torch.cuda.empty_cache()
            import gc
            gc.collect()

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Record
    record = {
        "combo_idx": combo_idx,
        "combo_tag": combo_tag,
        "erasure_factor": ERASURE_FACTOR,
        "start": START,
        "end": END,
        "channel_num": channel_num,
        "tf_start": tf_range.start,
        "tf_end": tf_range.stop,
        "status": "success" if combo_fail == 0 else "partial",
        "success_count": combo_success,
        "fail_count": combo_fail,
        "total_images": total_images,
    }
    meta_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    meta_file.flush()

    print(f"  Combo done: {combo_success}/{total_images} succeeded, {combo_fail} failed")

meta_file.close()

# ============================================================
# Summary
# ============================================================
total_success = total_partial = total_failed = 0
if meta_path.exists():
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line.strip())
                if r["status"] == "success":
                    total_success += 1
                elif r["status"] == "partial":
                    total_partial += 1
                else:
                    total_failed += 1

print(f"\n{'='*50}")
print(f"TI2I Grid search complete!")
print(f"  Success: {total_success}, Partial: {total_partial}, Failed: {total_failed}")
print(f"  Total combos: {len(combos)}")
print(f"  Test cases: {len(test_cases)}")
print(f"  Images per test: {NUM_IMAGES_PER_TEST}")
print(f"  Results in: {OUTPUT_DIR}")
print(f"{'='*50}")
