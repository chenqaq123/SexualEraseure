"""
Select important channels for TI2V (HunyuanVideo 1.5 I2V) erasure.
Outputs: channel_data_ti2v.json
Contains {ff: {indices, values}, ffn_context: {indices, values}}
for all 54 transformer layers.

Key masking logic:
- ff_context: only sum over VALID (non-padding) text tokens,
  using encoder_attention_mask captured from the transformer forward
- ff: sum over all visual tokens (no padding)

HunyuanVideo architecture:
- 54 transformer_blocks: ff.net[0] + ff_context.net[0]
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import json
from diffusers import HunyuanVideo15ImageToVideoPipeline
from pathlib import Path
from PIL import Image

DTYPE = torch.bfloat16
CACHE_DIR = "/home/chenguanxu/common_model/huggingface/hub"
SCRIPT_DIR = Path(__file__).parent
MODEL_ID = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v"

NUM_LAYERS = 54
K = 200

# Forget/retain prompts
D_FORGET = ["a nude woman", "a naked woman"] * 5
D_RETAIN = ["a dressed woman", "a dressed woman"] * 5

PLACEHOLDER_IMAGE = Image.new("RGB", (640, 480), (128, 128, 128))
NUM_FRAMES = 33


def collect_scores(pipe, modules, prompts, module_type, image=None, num_frames=33):
    """Run pipe on prompts and collect per-layer activation sums for given modules.

    For ff_context: only sum over valid (non-padding) text tokens using
    encoder_attention_mask captured from the transformer model's forward.
    For ff: sum over all visual tokens (no padding).
    """
    # Capture encoder_attention_mask from the transformer model's forward call
    # This is [batch, text_seq_len] with 1=valid, 0=padding
    _encoder_mask = {"mask": None}

    def _capture_encoder_mask_hook(module, args, kwargs):
        if "encoder_attention_mask" in kwargs:
            _encoder_mask["mask"] = kwargs["encoder_attention_mask"]
        elif len(args) >= 4:
            _encoder_mask["mask"] = args[3]

    mask_handle = pipe.transformer.register_forward_pre_hook(
        _capture_encoder_mask_hook, with_kwargs=True
    )

    sum_acc = [None] * len(modules)
    count_acc = [0] * len(modules)

    handles = []
    for idx, module in enumerate(modules):
        def make_hook(i):
            def hook(m, inp, output):
                out = output.detach().float()

                if module_type == "ffn_context":
                    # Use encoder_attention_mask: only sum valid text tokens
                    mask = _encoder_mask["mask"]  # [batch, text_seq_len]
                    if mask is not None:
                        m_2d = mask.float()  # [batch, text_seq_len]
                        while m_2d.dim() > 2:
                            m_2d = m_2d.squeeze(1)
                        if m_2d.shape[1] == out.shape[1]:
                            m_3d = m_2d.unsqueeze(-1).to(out.device, dtype=out.dtype)
                            contribution = (out * m_3d).sum(dim=(0, 1)).cpu()
                            n_valid = m_3d.sum().item()
                        else:
                            contribution = out.sum(dim=(0, 1)).cpu()
                            n_valid = out.shape[0] * out.shape[1]
                    else:
                        contribution = out.sum(dim=(0, 1)).cpu()
                        n_valid = out.shape[0] * out.shape[1]
                else:
                    # ff: visual tokens, no padding — sum all
                    contribution = out.sum(dim=(0, 1)).cpu()
                    n_valid = out.shape[0] * out.shape[1]

                if sum_acc[i] is None:
                    sum_acc[i] = contribution
                else:
                    sum_acc[i] += contribution
                count_acc[i] += n_valid
            return hook
        handles.append(module.register_forward_hook(make_hook(idx)))

    try:
        for i, p in enumerate(prompts):
            print(f"    Prompt {i+1}/{len(prompts)}: {p}")
            with torch.no_grad():
                gen = torch.Generator().manual_seed(i)
                pipe(
                    prompt=p,
                    image=image or PLACEHOLDER_IMAGE,
                    num_frames=num_frames,
                    generator=gen,
                )
            torch.cuda.empty_cache()
    finally:
        for h in handles:
            h.remove()
        mask_handle.remove()

    means = [sum_acc[i] / count_acc[i] if sum_acc[i] is not None else torch.zeros(1) for i in range(len(modules))]
    return means


def compute_topk(forget_scores, retain_scores, k=K):
    """Compute importance scores and return top-k indices and values."""
    topk_indices_list = []
    topk_values_list = []

    for n in range(len(forget_scores)):
        forget_score = forget_scores[n]
        retain_score = retain_scores[n]

        importance_score = torch.abs(forget_score) / torch.maximum(retain_score, torch.tensor(5e-2))
        topk512 = torch.topk(importance_score, k=min(512, len(importance_score))).indices
        mask = torch.zeros(forget_score.shape, dtype=torch.bool)
        mask[topk512] = True

        masked_forget = torch.where(mask, forget_score, torch.tensor(0.0))
        masked_retain = torch.where(mask, retain_score, torch.tensor(0.0))

        vector = masked_forget - masked_retain
        topk = torch.topk(torch.abs(vector), k=min(k, len(vector)))
        topk_indices_list.append(topk.indices.numpy().tolist())
        topk_values_list.append(vector[topk.indices].numpy().tolist())

    return {"indices": topk_indices_list, "values": topk_values_list}


# ============================================================
# Load Pipeline
# ============================================================
print("Loading TI2V pipeline...")
pipe = HunyuanVideo15ImageToVideoPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    cache_dir=CACHE_DIR,
    device_map="balanced",
)
pipe.vae.enable_tiling()
print("Pipeline loaded.")

# ============================================================
# Collect channel data for all module types
# ============================================================
result = {}

# 1. ff modules (transformer_blocks[i].ff.net[0])
print(f"\n{'='*60}")
print("Processing: ff (transformer_blocks)")
print(f"{'='*60}")

ff_modules = [pipe.transformer.transformer_blocks[i].ff.net[0] for i in range(NUM_LAYERS)]
print("Collecting forget scores...")
forget_scores = collect_scores(pipe, ff_modules, D_FORGET, module_type="ff", image=PLACEHOLDER_IMAGE, num_frames=NUM_FRAMES)
print("Collecting retain scores...")
retain_scores = collect_scores(pipe, ff_modules, D_RETAIN, module_type="ff", image=PLACEHOLDER_IMAGE, num_frames=NUM_FRAMES)
print("Computing top-k channels...")
result["ff"] = compute_topk(forget_scores, retain_scores)

# 2. ffn_context modules (transformer_blocks[i].ff_context.net[0])
print(f"\n{'='*60}")
print("Processing: ffn_context (transformer_blocks)")
print(f"{'='*60}")

ffn_context_modules = [pipe.transformer.transformer_blocks[i].ff_context.net[0] for i in range(NUM_LAYERS)]
print("Collecting forget scores...")
forget_scores = collect_scores(pipe, ffn_context_modules, D_FORGET, module_type="ffn_context", image=PLACEHOLDER_IMAGE, num_frames=NUM_FRAMES)
print("Collecting retain scores...")
retain_scores = collect_scores(pipe, ffn_context_modules, D_RETAIN, module_type="ffn_context", image=PLACEHOLDER_IMAGE, num_frames=NUM_FRAMES)
print("Computing top-k channels...")
result["ffn_context"] = compute_topk(forget_scores, retain_scores)

# ============================================================
# Save
# ============================================================
out_path = SCRIPT_DIR / "channel_data_ti2v.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nTI2V channel data saved to {out_path}")
print(f"  ff: {len(result['ff']['indices'])} layers x {len(result['ff']['indices'][0])} channels")
print(f"  ffn_context: {len(result['ffn_context']['indices'])} layers x {len(result['ffn_context']['indices'][0])} channels")
print("All done!")
