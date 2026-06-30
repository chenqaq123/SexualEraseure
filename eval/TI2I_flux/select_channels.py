"""
Select important channels for TI2I (FluxKontextPipeline) erasure.
Outputs: channel_data_ti2i.json
Contains {ff: {indices, values}, ffn_context: {indices, values}, proj_mlp: {indices, values}}
for all transformer layers.

Key masking logic:
- ff_context: only sum over VALID (non-padding) text tokens, using tokenizer attention_mask
- ff: sum over all visual tokens (no padding)
- proj_mlp: processes combined [text + visual] sequence; only mask text portion

Flux architecture:
- 19 transformer_blocks (0-18): ff.net[0] + ff_context.net[0]
- 38 single_transformer_blocks (0-37): proj_mlp
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import json
from diffusers import FluxKontextPipeline, FluxTransformer2DModel, BitsAndBytesConfig
from diffusers.utils import load_image
from pathlib import Path
from PIL import Image

DTYPE = torch.bfloat16
CACHE_DIR = "/home/mlsnrs/common_model/huggingface/hub"
SCRIPT_DIR = Path(__file__).parent
MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"

NUM_TRANSFORMER_BLOCKS = 19
NUM_SINGLE_BLOCKS = 38
K = 200
MAX_SEQUENCE_LENGTH = 512  # T5 tokenizer max length used by Flux

# Forget/retain prompts
D_FORGET = ["a nude woman", "a naked woman"] * 5
D_RETAIN = ["a dressed woman", "a dressed woman"] * 5

PLACEHOLDER_IMAGE = Image.new("RGB", (1024, 1024), (128, 128, 128))


def tokenize_prompts(tokenizer, prompts):
    """Tokenize prompts and return attention_mask [batch, seq_len].
    Flux uses tokenizer_2 (T5) with padding="max_length" and max_length=512.
    """
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    return text_inputs.attention_mask  # [batch, seq_len]


def collect_scores(pipe, modules, prompts, module_type):
    """Run pipe on prompts and collect per-layer activation sums for given modules.

    For ff_context: only sum valid (non-padding) text tokens using tokenizer mask.
    For ff: sum all visual tokens (no padding).
    For proj_mlp: sum over combined [text+visual] sequence, masking text padding positions.
    """
    # Pre-compute text token masks for all prompts
    text_masks = tokenize_prompts(pipe.tokenizer_2, prompts)  # [num_prompts, 512]

    sum_acc = [None] * len(modules)
    count_acc = [0] * len(modules)

    # Capture encoder_seq_len per forward pass for proj_mlp
    _context_info = {"encoder_seq_len": 0}

    def _capture_context_hook(module, args, kwargs):
        # FluxTransformerBlock.forward: (hidden_states, encoder_hidden_states, temb, ...)
        # args[1] = encoder_hidden_states
        if len(args) >= 2:
            _context_info["encoder_seq_len"] = args[1].shape[1]

    if module_type == "proj_mlp":
        context_handle = pipe.transformer.transformer_blocks[0].register_forward_pre_hook(
            _capture_context_hook, with_kwargs=True
        )

    handles = []
    for idx, module in enumerate(modules):
        def make_hook(i):
            def hook(m, inp, output):
                out = output.detach().float()

                if module_type == "ffn_context":
                    # ff_context output: [batch, text_seq_len, hidden_dim]
                    # Use text_mask from tokenizer for this prompt
                    mask = _current_text_mask.to(out.device).float()  # [text_seq_len]
                    if mask.shape[0] == out.shape[1]:
                        m_2d = mask.unsqueeze(0).expand(out.shape[0], -1)  # [batch, text_seq_len]
                        m_3d = m_2d.unsqueeze(-1)  # [batch, text_seq_len, 1]
                        contribution = (out * m_3d).sum(dim=(0, 1)).cpu()
                        n_valid = m_3d.sum().item()
                    else:
                        contribution = out.sum(dim=(0, 1)).cpu()
                        n_valid = out.shape[0] * out.shape[1]

                elif module_type == "proj_mlp":
                    # proj_mlp output: [batch, text_seq_len + visual_seq_len, hidden_dim]
                    # First text_seq_len positions are text (need masking), rest are visual (all valid)
                    text_len = _context_info["encoder_seq_len"]
                    visual_len = out.shape[1] - text_len
                    text_mask = _current_text_mask.to(out.device).float()  # [text_seq_len]
                    if text_mask.shape[0] == text_len:
                        visual_mask = torch.ones(visual_len, device=out.device)
                        combined_mask = torch.cat([text_mask, visual_mask])  # [text+visual]
                        m_3d = combined_mask.unsqueeze(0).unsqueeze(-1).expand(out.shape[0], -1, -1)
                        contribution = (out * m_3d).sum(dim=(0, 1)).cpu()
                        n_valid = m_3d.sum().item()
                    else:
                        contribution = out.sum(dim=(0, 1)).cpu()
                        n_valid = out.shape[0] * out.shape[1]

                else:
                    # ff: visual tokens, no padding
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
            # Set the text mask for this prompt (shared with hooks via closure)
            _current_text_mask = text_masks[i]  # [seq_len]

            with torch.no_grad():
                gen = torch.Generator().manual_seed(i)
                pipe(image=PLACEHOLDER_IMAGE, prompt=p, guidance_scale=5, generator=gen)
            torch.cuda.empty_cache()
    finally:
        for h in handles:
            h.remove()
        if module_type == "proj_mlp":
            context_handle.remove()

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
    torch_dtype=DTYPE,
    device_map={"": 0},
    local_files_only=True,
)

pipe = FluxKontextPipeline.from_pretrained(
    MODEL_ID,
    transformer=transformer,
    cache_dir=CACHE_DIR,
    torch_dtype=DTYPE,
    local_files_only=True,
)
pipe.to("cuda:0")
print("Pipeline loaded.")

# ============================================================
# Collect channel data for all module types
# ============================================================
result = {}

# 1. ff modules (transformer_blocks[i].ff.net[0])
print(f"\n{'='*60}")
print("Processing: ff (transformer_blocks)")
print(f"{'='*60}")

ff_modules = [pipe.transformer.transformer_blocks[i].ff.net[0] for i in range(NUM_TRANSFORMER_BLOCKS)]
print("Collecting forget scores...")
forget_scores = collect_scores(pipe, ff_modules, D_FORGET, module_type="ff")
print("Collecting retain scores...")
retain_scores = collect_scores(pipe, ff_modules, D_RETAIN, module_type="ff")
print("Computing top-k channels...")
result["ff"] = compute_topk(forget_scores, retain_scores)

# 2. ffn_context modules (transformer_blocks[i].ff_context.net[0])
print(f"\n{'='*60}")
print("Processing: ffn_context (transformer_blocks)")
print(f"{'='*60}")

ffn_context_modules = [pipe.transformer.transformer_blocks[i].ff_context.net[0] for i in range(NUM_TRANSFORMER_BLOCKS)]
print("Collecting forget scores...")
forget_scores = collect_scores(pipe, ffn_context_modules, D_FORGET, module_type="ffn_context")
print("Collecting retain scores...")
retain_scores = collect_scores(pipe, ffn_context_modules, D_RETAIN, module_type="ffn_context")
print("Computing top-k channels...")
result["ffn_context"] = compute_topk(forget_scores, retain_scores)

# 3. proj_mlp modules (single_transformer_blocks[i].proj_mlp)
print(f"\n{'='*60}")
print("Processing: proj_mlp (single_transformer_blocks)")
print(f"{'='*60}")

proj_mlp_modules = [pipe.transformer.single_transformer_blocks[i].proj_mlp for i in range(NUM_SINGLE_BLOCKS)]
print("Collecting forget scores...")
forget_scores = collect_scores(pipe, proj_mlp_modules, D_FORGET, module_type="proj_mlp")
print("Collecting retain scores...")
retain_scores = collect_scores(pipe, proj_mlp_modules, D_RETAIN, module_type="proj_mlp")
print("Computing top-k channels...")
result["proj_mlp"] = compute_topk(forget_scores, retain_scores)

# ============================================================
# Save
# ============================================================
out_path = SCRIPT_DIR / "channel_data_ti2i.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nTI2I channel data saved to {out_path}")
print(f"  ff: {len(result['ff']['indices'])} layers x {len(result['ff']['indices'][0])} channels")
print(f"  ffn_context: {len(result['ffn_context']['indices'])} layers x {len(result['ffn_context']['indices'][0])} channels")
print(f"  proj_mlp: {len(result['proj_mlp']['indices'])} layers x {len(result['proj_mlp']['indices'][0])} channels")
print("All done!")
