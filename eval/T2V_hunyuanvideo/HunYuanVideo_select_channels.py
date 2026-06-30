import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3,4"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import json
from diffusers import HunyuanVideo15Pipeline

DTYPE = torch.bfloat16
MODEL_ID = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
cache_dir = "/home/mlsnrs/data/cgx/models"
pipe = HunyuanVideo15Pipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    cache_dir=cache_dir,
    device_map="balanced"
)

pipe.vae.enable_tiling()
pipe.enable_attention_slicing()

NUM_LAYERS = 54
modules = [pipe.transformer.transformer_blocks[i].ff_context.net[0] for i in range(NUM_LAYERS)]

D_forget = ["a nude woman", "a naked woman"] * 5
D_retain = ["a dressed woman", "a dressed woman"] * 5

# ──────────────────────────────────────────────
# 1. Capture the REORDERED encoder_attention_mask
#    from the first transformer block's input
# ──────────────────────────────────────────────
_current_block_mask = {"mask": None}


def _capture_block_mask_hook(module, args, kwargs):
    # block forward signature: (hidden_states, encoder_hidden_states, temb, attention_mask, freqs_cis)
    # attention_mask is the 4th positional arg (index 3)
    if len(args) >= 4:
        _current_block_mask["mask"] = args[3]  # reordered mask, shape (B, 1985)
    elif "attention_mask" in kwargs:
        _current_block_mask["mask"] = kwargs["attention_mask"]


_mask_handle = pipe.transformer.transformer_blocks[0].register_forward_pre_hook(
    _capture_block_mask_hook, with_kwargs=True
)

# ──────────────────────────────────────────────
# 2. Online accumulators
# ──────────────────────────────────────────────


def _make_accumulators(n_layers):
    return [None] * n_layers, [0] * n_layers


def _register_hooks(sum_acc, count_acc):
    handles = []
    for idx, module in enumerate(modules):

        def make_hook(i):
            def hook(m, inp, output):
                out = output.detach().float()  # (B, 1985, 8192)

                mask = _current_block_mask["mask"]
                if mask is not None:
                    m_flat = mask
                    while m_flat.dim() > 2:
                        m_flat = m_flat.squeeze(1)
                    # Now m_flat should be (B, 1985) — reordered mask
                    # Valid tokens (True) are at the front, padding (False) at the back

                    m_3d = m_flat.unsqueeze(-1).to(out.device, dtype=out.dtype)

                    if i == 0 and count_acc[0] == 0:
                        n_valid = int(m_3d.sum().item())
                        seq_len = out.shape[1]
                        print(f"重排后 mask: shape={list(m_flat.shape)}, "
                              f"有效token={n_valid}, padding={seq_len - n_valid}")
                        print(f"  mask前20: {m_flat[0, :20].tolist()}")
                        print(f"  mask后20: {m_flat[0, -20:].tolist()}")

                        # 看有效 token 区域的 norm
                        norms = out[0].norm(dim=-1)
                        print(f"  有效区域 norm (前{min(n_valid,10)}个): "
                              f"{[f'{x:.4f}' for x in norms[:min(n_valid,10)].tolist()]}")
                        print(f"  padding区域 norm (最后10个): "
                              f"{[f'{x:.4f}' for x in norms[-10:].tolist()]}")

                    contribution = (out * m_3d).sum(dim=(0, 1)).cpu()
                    n_valid = m_3d.sum().item()
                else:
                    contribution = out.sum(dim=(0, 1)).cpu()
                    n_valid = out.shape[0] * out.shape[1]

                if sum_acc[i] is None:
                    sum_acc[i] = contribution
                else:
                    sum_acc[i] += contribution
                count_acc[i] += n_valid

            return hook

        handles.append(module.register_forward_hook(make_hook(idx)))
    return handles


def _collect_scores(prompts, pipe):
    sum_acc, count_acc = _make_accumulators(NUM_LAYERS)
    handles = _register_hooks(sum_acc, count_acc)
    try:
        for i, p in enumerate(prompts):
            print(f"  Prompt {i + 1}/{len(prompts)}: {p}")
            with torch.no_grad():
                generator = torch.Generator().manual_seed(i)
                pipe(
                    prompt=p,
                    height=360,
                    width=640,
                    num_frames=33,
                    generator=generator,
                )
            torch.cuda.empty_cache()
    finally:
        for h in handles:
            h.remove()

    means = []
    for i in range(NUM_LAYERS):
        means.append(sum_acc[i] / count_acc[i])
        sum_acc[i] = None
    return means


# ──────────────────────────────────────────────
# 3. Collect
# ──────────────────────────────────────────────
print("=== Collecting forget scores ===")
forget_scores = _collect_scores(D_forget, pipe)

print("=== Collecting retain scores ===")
retain_scores = _collect_scores(D_retain, pipe)

_mask_handle.remove()

# ──────────────────────────────────────────────
# 4. Importance & top-k
# ──────────────────────────────────────────────
topk_indices_list = []
topk_values_list = []

for n in range(NUM_LAYERS):
    forget_score = forget_scores[n]
    retain_score = retain_scores[n]

    importance_score = torch.abs(forget_score) / torch.maximum(
        retain_score, torch.tensor(5e-2)
    )
    topk512 = torch.topk(importance_score, k=512).indices
    mask = torch.zeros(forget_score.shape, dtype=torch.bool)
    mask[topk512] = True

    masked_forget = torch.where(mask, forget_score, torch.tensor(0.0))
    masked_retain = torch.where(mask, retain_score, torch.tensor(0.0))

    vector = masked_forget - masked_retain
    k = 200
    topk = torch.topk(torch.abs(vector), k=k)
    topk_indices_list.append(topk.indices.numpy().tolist())
    topk_values_list.append(vector[topk.indices].numpy().tolist())

data = {"indices": topk_indices_list, "values": topk_values_list}
with open("result_ff_context.json", "w") as f:
    json.dump(data, f, indent=2)

print("Done — saved result_ff_context.json")