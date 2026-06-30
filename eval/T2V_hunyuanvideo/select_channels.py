"""
Select important channels for T2V and TI2V (HunyuanVideo 1.5) erasure.
Outputs: channel_data_t2v.json and channel_data_ti2v.json
Each contains {ff: {indices, values}, ffn_context: {indices, values}} for all 54 transformer layers.

Key: ff_context hooks only sum over VALID (non-padding) text tokens,
using encoder_attention_mask captured from the transformer model.
ff hooks sum over all visual tokens (no padding in visual tokens).
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import json
from diffusers import HunyuanVideo15Pipeline, HunyuanVideo15ImageToVideoPipeline
from diffusers.utils import load_image
from pathlib import Path

DTYPE = torch.bfloat16
CACHE_DIR = "/home/mlsnrs/common_model/huggingface/hub"
SCRIPT_DIR = Path(__file__).parent
NUM_LAYERS = 54
K = 200

# Forget/retain prompts
D_FORGET = ["a nude woman", "a naked woman"] * 5
D_RETAIN = ["a dressed woman", "a dressed woman"] * 5


def collect_scores(pipe, modules, prompts, module_type, is_i2v=False, image=None, num_frames=33):
    """Run pipe on prompts and collect per-layer activation sums for given modules.

    For ff_context: only sum over valid (non-padding) text tokens using
    encoder_attention_mask captured from the transformer model's forward.
    For ff: sum over all visual tokens (no padding).
    """
    # Capture encoder_attention_mask from the transformer model's forward call
    # This is [batch, text_seq_len] with 1=valid, 0=padding
    _encoder_mask = {"mask": None}

    def _capture_encoder_mask_hook(module, args, kwargs):
        # HunyuanVideoTransformer3DModel.forward() takes:
        #   hidden_states, timestep, encoder_hidden_states, encoder_attention_mask, ...
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
                        # Handle potential extra dims
                        while m_2d.dim() > 2:
                            m_2d = m_2d.squeeze(1)
                        # Ensure mask matches output seq_len
                        if m_2d.shape[1] == out.shape[1]:
                            m_3d = m_2d.unsqueeze(-1).to(out.device, dtype=out.dtype)
                            contribution = (out * m_3d).sum(dim=(0, 1)).cpu()
                            n_valid = m_3d.sum().item()
                        else:
                            # Fallback: mask shape doesn't match output
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
                if is_i2v and image is not None:
                    pipe(prompt=p, image=image, num_frames=num_frames, generator=gen)
                else:
                    pipe(prompt=p, height=360, width=640, num_frames=num_frames, generator=gen)
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


def run_for_pipeline(pipe_name, pipe, is_i2v=False, image=None):
    """Collect channel data for both ff and ff_context from a pipeline."""
    print(f"\n{'='*60}")
    print(f"Processing: {pipe_name}")
    print(f"{'='*60}")

    ff_modules = [pipe.transformer.transformer_blocks[i].ff.net[0] for i in range(NUM_LAYERS)]
    ffn_context_modules = [pipe.transformer.transformer_blocks[i].ff_context.net[0] for i in range(NUM_LAYERS)]

    result = {}

    for module_type, modules in [("ff", ff_modules), ("ffn_context", ffn_context_modules)]:
        print(f"\n--- {module_type} ---")
        print("Collecting forget scores...")
        forget_scores = collect_scores(pipe, modules, D_FORGET, module_type=module_type, is_i2v=is_i2v, image=image)
        print("Collecting retain scores...")
        retain_scores = collect_scores(pipe, modules, D_RETAIN, module_type=module_type, is_i2v=is_i2v, image=image)
        print("Computing top-k channels...")
        result[module_type] = compute_topk(forget_scores, retain_scores)

    return result


# ============================================================
# 1. T2V Pipeline
# ============================================================
print("Loading T2V pipeline...")
t2v_pipe = HunyuanVideo15Pipeline.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
    torch_dtype=DTYPE,
    cache_dir=CACHE_DIR,
    device_map="balanced",
)
t2v_pipe.vae.enable_tiling()
t2v_pipe.enable_attention_slicing()

t2v_data = run_for_pipeline("T2V", t2v_pipe, is_i2v=False)

out_path = SCRIPT_DIR / "channel_data_t2v.json"
with open(out_path, "w") as f:
    json.dump(t2v_data, f, indent=2)
print(f"T2V channel data saved to {out_path}")

del t2v_pipe
torch.cuda.empty_cache()

# ============================================================
# 2. TI2V Pipeline
# ============================================================
print("\nLoading TI2V pipeline...")
ti2v_pipe = HunyuanVideo15ImageToVideoPipeline.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v",
    torch_dtype=DTYPE,
    cache_dir=CACHE_DIR,
    device_map="balanced",
)
ti2v_pipe.vae.enable_tiling()

from PIL import Image
placeholder_image = Image.new("RGB", (640, 480), (128, 128, 128))

ti2v_data = run_for_pipeline("TI2V", ti2v_pipe, is_i2v=True, image=placeholder_image)

out_path = SCRIPT_DIR / "channel_data_ti2v.json"
with open(out_path, "w") as f:
    json.dump(ti2v_data, f, indent=2)
print(f"TI2V channel data saved to {out_path}")

del ti2v_pipe
torch.cuda.empty_cache()

print("\nAll done!")
