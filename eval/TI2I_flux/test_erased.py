"""
Run TI2I (FluxKontextPipeline) UniErase-Bench with erasure hooks applied.

Current erasure configuration:
- ff: all 19 transformer_blocks (0-18), top-15 channels
- ffn_context: all 19 transformer_blocks (0-18), top-15 channels
- proj_mlp: first 10 single_transformer_blocks (0-9), top-15 channels
- erasure_factor: -4.0

Flux architecture:
- 19 transformer_blocks (0-18): ff.net[0] + ff_context.net[0]
- 38 single_transformer_blocks (0-37): proj_mlp
"""
import os
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch
from diffusers import FluxKontextPipeline, FluxTransformer2DModel, BitsAndBytesConfig
from diffusers.utils import load_image
from pathlib import Path

# ============================================================
# Config
# ============================================================
MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"
CACHE_DIR = "/home/mlsnrs/common_model/huggingface/hub"
BENCHMARK_PATH = "/home/mlsnrs/data/cgx/cgx_nas/T2I/Concept_Erasure/prompt_data/UniErase-Bench/ti2i_benchmark.jsonl"
CHANNEL_DATA_PATH = Path(__file__).parent / "channel_data_ti2i.json"

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "erased_results"

# Best erasure params — adjust after grid search
ERASURE_FACTOR = -4.0
START = 0
END = 15

# Erasure ranges for different module types
# Note: channel_data currently has 19 layers, adjust if regenerated
FF_RANGE = range(0, 19)        # All 19 transformer_blocks (mm_blocks)
FFN_CONTEXT_RANGE = range(0, 19)  # All 19 transformer_blocks (mm_blocks)
PROJ_MLP_RANGE = range(0, 10)  # First 10 single_transformer_blocks

GUIDANCE_SCALE = 5

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
# Load Channel Data & Register Hooks
# ============================================================
print(f"Loading channel data from {CHANNEL_DATA_PATH}")
with open(CHANNEL_DATA_PATH, "r") as f:
    channel_data = json.load(f)

# Verify channel data has enough layers
assert len(channel_data["ff"]["indices"]) >= 19, "ff channel data should have at least 19 layers"
assert len(channel_data["ffn_context"]["indices"]) >= 19, "ffn_context channel data should have at least 19 layers"
assert len(channel_data["proj_mlp"]["indices"]) >= 10, "proj_mlp channel data should have at least 10 layers"

# Get modules and indices for ff (all 19 transformer_blocks)
ff_modules = [pipe.transformer.transformer_blocks[i].ff.net[0] for i in FF_RANGE]
ff_indices = [channel_data["ff"]["indices"][i] for i in FF_RANGE]

# Get modules and indices for ffn_context (all 19 transformer_blocks)
ffn_context_modules = [pipe.transformer.transformer_blocks[i].ff_context.net[0] for i in FFN_CONTEXT_RANGE]
ffn_context_indices = [channel_data["ffn_context"]["indices"][i] for i in FFN_CONTEXT_RANGE]

# Get modules and indices for proj_mlp (first 10 single_transformer_blocks)
proj_mlp_modules = [pipe.transformer.single_transformer_blocks[i].proj_mlp for i in PROJ_MLP_RANGE]
proj_mlp_indices = [channel_data["proj_mlp"]["indices"][i] for i in PROJ_MLP_RANGE]

# Register hooks for ff
def make_ff_hook(layer_idx):
    top_channels = ff_indices[layer_idx][START:END]
    def hook(module, input, output):
        with torch.no_grad():
            output[:, :, top_channels] *= ERASURE_FACTOR
        return output
    return hook

# Register hooks for ffn_context
def make_ffn_context_hook(layer_idx):
    top_channels = ffn_context_indices[layer_idx][START:END]
    def hook(module, input, output):
        with torch.no_grad():
            output[:, :, top_channels] *= ERASURE_FACTOR
        return output
    return hook

# Register hooks for proj_mlp
def make_proj_mlp_hook(layer_idx):
    top_channels = proj_mlp_indices[layer_idx][START:END]
    def hook(module, input, output):
        with torch.no_grad():
            output[:, :, top_channels] *= ERASURE_FACTOR
        return output
    return hook

# Register all hooks
hooks = []
for i, module in enumerate(ff_modules):
    hooks.append(module.register_forward_hook(make_ff_hook(i)))
for i, module in enumerate(ffn_context_modules):
    hooks.append(module.register_forward_hook(make_ffn_context_hook(i)))
for i, module in enumerate(proj_mlp_modules):
    hooks.append(module.register_forward_hook(make_proj_mlp_hook(i)))

print(f"Erasure hooks registered: {len(hooks)} total hooks")
print(f"  ff: {len(ff_modules)} hooks on layers {FF_RANGE.start}-{FF_RANGE.stop}")
print(f"  ffn_context: {len(ffn_context_modules)} hooks on layers {FFN_CONTEXT_RANGE.start}-{FFN_CONTEXT_RANGE.stop}")
print(f"  proj_mlp: {len(proj_mlp_modules)} hooks on layers {PROJ_MLP_RANGE.start}-{PROJ_MLP_RANGE.stop}")
print(f"  erasure_factor={ERASURE_FACTOR}, slice=[{START}:{END}], channel_num={END-START}")

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
        image = load_image(reference_image_path).convert("RGB").resize((1024, 1024))
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
            result_img = pipe(
                image=image,
                prompt=instruction,
                guidance_scale=GUIDANCE_SCALE,
                generator=torch.Generator().manual_seed(42),
            ).images[0]
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

    out_filename = f"idx{idx:04d}_sid{source_id}.png"
    out_path = OUTPUT_DIR / out_filename
    try:
        result_img.save(str(out_path))
        print(f"  Saved: {out_path}")
    except Exception as e:
        print(f"  Failed to save image: {e}")
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
