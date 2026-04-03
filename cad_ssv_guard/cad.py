"""Concept Attribution Distillation (CAD) scoring.

Computes per-channel gradient attribution scores for every FFN layer in the
denoising backbone.  The score for channel *j* in layer *l* measures how
strongly that channel contributes to the *difference* between the model's
prediction under positive (nudity) conditioning and negative (safe)
conditioning — i.e. how much it "encodes" the concept we want to erase.

The computation is model-agnostic: all model-specific forward-pass logic is
delegated to a :class:`~cad_ssv_guard.backend.ModelBackend`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .backend import ModelBackend
from .ffn import (
    FFNModuleSpec,
    build_parameter_lookup,
    discover_ffn_specs,
    get_projection_weight_name,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CADLayerScores:
    spec: FFNModuleSpec
    # Normalised attribution score per channel (column-normalised by weight scale)
    channel_scores: torch.Tensor
    # Raw gradient attribution before normalisation
    raw_channel_scores: torch.Tensor
    # Scalar summary: mean of top-k channel scores (used to rank layers)
    layer_score: float


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_prompt_pairs(
    positive_prompts: Optional[Sequence[str]],
    negative_prompts: Optional[Sequence[str]],
    concept_prompt: str,
    base_prompt: str,
) -> List[Tuple[str, str]]:
    if positive_prompts and negative_prompts:
        count = min(len(positive_prompts), len(negative_prompts))
        if count == 0:
            raise ValueError("Prompt pair list must be non-empty.")
        return [(positive_prompts[i], negative_prompts[i]) for i in range(count)]
    return [(concept_prompt, base_prompt)]


# ─────────────────────────────────────────────────────────────────────────────
# Main scoring function
# ─────────────────────────────────────────────────────────────────────────────

def _run_single_sample_attribution(
    pipe,
    backend: ModelBackend,
    pos_prompt: str,
    neg_prompt: str,
    sample_seed: int,
    height: int,
    width: int,
    num_steps: int,
    tracked_weights: Dict[str, torch.nn.Parameter],
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    """Run CAD attribution for a single (prompt_pair, sample) trajectory.

    Returns per-layer score_buffers accumulated over all timesteps.
    """
    local_buffers: Dict[str, torch.Tensor] = {
        ff_name: torch.zeros(weight.shape[1], dtype=torch.float32, device=device)
        for ff_name, weight in tracked_weights.items()
    }

    pipe.scheduler.set_timesteps(num_steps, device=device)

    text_dict = backend.encode_prompt_pair(pipe, pos_prompt, neg_prompt, device)

    generator = torch.Generator(device=device).manual_seed(sample_seed)
    latents_dict = backend.prepare_latents(
        pipe, height, width, device, dtype, generator
    )

    for timestep in pipe.scheduler.timesteps:
        noise_pred_pos, noise_pred_neg = backend.cad_forward(
            pipe, latents_dict, text_dict, timestep
        )

        objective = F.mse_loss(
            noise_pred_pos.float(),
            noise_pred_neg.float(),
            reduction="sum",
        )

        backbone = backend.get_backbone(pipe)
        backbone.zero_grad(set_to_none=True)
        objective.backward()

        for ff_name, weight in tracked_weights.items():
            grad = weight.grad
            if grad is None:
                continue
            attribution = (
                weight.detach().float() * grad.detach().float()
            ).clamp_min(0.0)
            local_buffers[ff_name] += attribution.sum(dim=0)

        with torch.no_grad():
            latents_dict = backend.scheduler_step(
                pipe, noise_pred_pos, timestep, latents_dict
            )

    return local_buffers


def compute_nudity_cad_scores(
    pipe,
    backend: ModelBackend,
    positive_prompts: Optional[Sequence[str]] = None,
    negative_prompts: Optional[Sequence[str]] = None,
    concept_prompt: str = "naked",
    base_prompt: str = "",
    num_steps: int = 50,
    num_samples: int = 4,
    seed: int = 0,
    height: Optional[int] = None,
    width: Optional[int] = None,
    layer_topk: int = 64,
    eps: float = 1e-6,
    devices: Optional[Sequence[torch.device]] = None,
) -> Dict[str, CADLayerScores]:
    """Compute CAD channel attribution scores for all FFN projection layers.

    For each prompt pair and each denoising timestep we:

    1. Run a forward pass conditioned on the positive (nudity) prompt and on
       the negative (safe) prompt.  For CFG-capable models (SD1, SD3) this is
       a single batch-2 call; for FLUX it is two separate forward passes.
    2. Compute the MSE between the two predictions as a proxy attribution loss.
    3. Backpropagate and accumulate ``weight * grad`` attributions for every
       FFN projection matrix.

    Multi-GPU: pass ``devices=[torch.device("cuda:0"), ...]`` to distribute
    samples across GPUs. Each GPU runs a subset of trajectories independently.

    Parameters
    ----------
    pipe:
        A loaded diffusers pipeline (SD1, SD3, or FluxPipeline).
    backend:
        Model-specific backend that handles text encoding, latent preparation,
        and the forward call.
    positive_prompts / negative_prompts:
        Lists of prompt strings defining the concept to localise.
        When provided, ``concept_prompt`` and ``base_prompt`` are ignored.
    concept_prompt / base_prompt:
        Fallback single-pair prompts used when lists are not provided.
    num_steps:
        Number of scheduler timesteps used during attribution.  Fewer steps
        are faster but cover a shorter portion of the diffusion trajectory.
    num_samples:
        Number of independent noise samples per prompt pair.
    seed:
        Base random seed; each sample uses ``seed + pair_idx * 10_000 + sample_idx``.
    height / width:
        Spatial resolution of the attribution latents.  Defaults to the
        backend's ``default_height`` / ``default_width``.
    layer_topk:
        Number of top channels used to compute the scalar ``layer_score``.
    eps:
        Small constant added to the column scale to avoid division by zero.
    devices:
        List of torch devices for multi-GPU parallel attribution.
        If None or single device, runs on ``pipe.device`` only.

    Returns
    -------
    Dict mapping ``ff_name → CADLayerScores`` for every discovered FFN layer.
    """
    height = height or backend.default_height
    width = width or backend.default_width

    prompt_pairs = _build_prompt_pairs(
        positive_prompts, negative_prompts, concept_prompt, base_prompt
    )

    # ── Discover FFN layers and tracked projection weights ────────────────────
    backbone = backend.get_backbone(pipe)
    parameter_lookup = build_parameter_lookup(backbone)
    ffn_specs = discover_ffn_specs(backbone)
    tracked_weights: Dict[str, torch.nn.Parameter] = {
        spec.ff_name: parameter_lookup[get_projection_weight_name(spec)]
        for spec in ffn_specs.values()
    }

    dtype = next(backbone.parameters()).dtype

    # ── Build work items: (pair_idx, sample_idx, sample_seed) ────────────────
    work_items = []
    for pair_idx, (pos_prompt, neg_prompt) in enumerate(prompt_pairs):
        for sample_idx in range(num_samples):
            sample_seed = seed + pair_idx * 10_000 + sample_idx
            work_items.append((pos_prompt, neg_prompt, sample_seed))

    # ── Multi-GPU: model parallelism (model already distributed across GPUs) ──
    if devices is not None and len(devices) > 1:
        print(f"\n[CAD] Using model parallelism across {len(devices)} GPUs")
        device_map = getattr(pipe, 'hf_device_map', None) or getattr(pipe.transformer, 'hf_device_map', {})
        print(f"[CAD] Transformer device map: {device_map}")
        # Use text_encoder's device as the primary device for computation
        # In multi-GPU mode, text_encoder and first transformer blocks should be on same GPU
        text_encoder_device = pipe.text_encoder.device
        print(f"[CAD] Primary device (text_encoder): {text_encoder_device}")
        device = text_encoder_device
    else:
        # ── Single-GPU: run in-place on the existing pipeline ─────────────────────
        device = pipe.device

    pipe.scheduler.set_timesteps(num_steps, device=device)

    score_buffers: Dict[str, torch.Tensor] = {
        ff_name: torch.zeros(weight.shape[1], dtype=torch.float32, device=device)
        for ff_name, weight in tracked_weights.items()
    }

    for pos_prompt, neg_prompt, sample_seed in work_items:
        local = _run_single_sample_attribution(
            pipe=pipe,
            backend=backend,
            pos_prompt=pos_prompt,
            neg_prompt=neg_prompt,
            sample_seed=sample_seed,
            height=height,
            width=width,
            num_steps=num_steps,
            tracked_weights=tracked_weights,
            device=device,
            dtype=dtype,
        )
        for ff_name in score_buffers:
            score_buffers[ff_name] += local[ff_name]

    # ── Normalise and aggregate ───────────────────────────────────────────────
    results: Dict[str, CADLayerScores] = {}
    for ff_name, spec in ffn_specs.items():
        raw_scores = score_buffers[ff_name].detach()
        weight = tracked_weights[ff_name].detach().float()
        column_scale = weight.abs().mean(dim=0).clamp_min(eps)
        channel_scores = (raw_scores / column_scale).cpu()

        topk = min(layer_topk, channel_scores.numel())
        layer_score = float(torch.topk(channel_scores, k=topk).values.mean().item())

        results[ff_name] = CADLayerScores(
            spec=spec,
            channel_scores=channel_scores,
            raw_channel_scores=raw_scores.cpu(),
            layer_score=layer_score,
        )
    return results


def _compute_nudity_cad_scores_multi_gpu(
    model_id: str,
    backend: ModelBackend,
    prompt_pairs: List[Tuple[str, str]],
    num_samples: int,
    seed: int,
    height: int,
    width: int,
    num_steps: int,
    tracked_weights: Dict[str, torch.nn.Parameter],
    ffn_specs: Dict[str, FFNModuleSpec],
    dtype: torch.dtype,
    devices: List[torch.device],
    layer_topk: int,
    eps: float,
) -> Dict[str, CADLayerScores]:
    """Distribute CAD attribution across multiple GPUs via multiprocessing."""
    import multiprocessing as mp
    from multiprocessing import Queue

    # Build work items
    work_items = []
    for pair_idx, (pos_prompt, neg_prompt) in enumerate(prompt_pairs):
        for sample_idx in range(num_samples):
            sample_seed = seed + pair_idx * 10_000 + sample_idx
            work_items.append((pos_prompt, neg_prompt, sample_seed))

    n_gpus = len(devices)

    # Partition work items across GPUs (round-robin)
    gpu_work: List[List] = [[] for _ in range(n_gpus)]
    for i, item in enumerate(work_items):
        gpu_work[i % n_gpus].append(item)

    # Shared queues for results
    result_queues: List[Queue] = [mp.Queue() for _ in range(n_gpus)]

    def _worker(
        model_id: str,
        model_type: str,
        height: int,
        width: int,
        num_steps: int,
        dtype_str: str,
        items: List,
        device: torch.device,
        result_q: Queue,
    ):
        """Worker process: loads model on its GPU, runs its share of attribution."""
        import sys
        import torch
        import torch.nn.functional as F
        from diffusers import (
            CogVideoXPipeline,
            FluxPipeline,
            HunyuanVideoPipeline,
            StableDiffusion3Pipeline,
            StableDiffusionPipeline,
        )

        device_idx = device.index if hasattr(device, 'index') else 0
        print(f"[GPU {device_idx}] Worker starting, loading model...", flush=True)

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype_str, torch.float16)

        # Load pipeline on assigned device
        device_str = str(device)
        pipe_loaders = {
            "sd1": lambda: StableDiffusionPipeline.from_pretrained(
                model_id, torch_dtype=torch_dtype, safety_checker=None
            ).to(device_str),
            "sd3": lambda: StableDiffusion3Pipeline.from_pretrained(
                model_id, torch_dtype=torch_dtype,
            ).to(device_str),
            "flux": lambda: FluxPipeline.from_pretrained(
                model_id, torch_dtype=torch_dtype,
            ).to(device_str),
            "cogvideox": lambda: CogVideoXPipeline.from_pretrained(
                model_id, torch_dtype=torch_dtype,
            ).to(device_str),
            "hunyuanvideo": lambda: HunyuanVideoPipeline.from_pretrained(
                model_id, torch_dtype=torch_dtype,
            ).to(device_str),
        }
        print(f"[GPU {device_idx}] Loading pipeline...", flush=True)
        pipe = pipe_loaders[model_type]()
        print(f"[GPU {device_idx}] Pipeline loaded successfully.", flush=True)
        pipe.set_progress_bar_config(disable=True)

        backend_instance = _get_backend_by_type(model_type)
        backbone = backend_instance.get_backbone(pipe)
        parameter_lookup = build_parameter_lookup(backbone)
        tracked = {
            spec.ff_name: parameter_lookup[get_projection_weight_name(spec)]
            for spec in discover_ffn_specs(backbone).values()
        }

        device_t = torch.device(device_str)
        score_buf = {
            ff_name: torch.zeros(weight.shape[1], dtype=torch.float32, device=device_t)
            for ff_name, weight in tracked.items()
        }

        print(f"[GPU {device_idx}] Starting CAD attribution for {len(items)} samples...", flush=True)
        for item_idx, (pos_prompt, neg_prompt, sample_seed) in enumerate(items, 1):
            print(f"[GPU {device_idx}] Running sample {item_idx}/{len(items)} (seed={sample_seed})", flush=True)
            local = _run_single_sample_attribution(
                pipe=pipe,
                backend=backend_instance,
                pos_prompt=pos_prompt,
                neg_prompt=neg_prompt,
                sample_seed=sample_seed,
                height=height,
                width=width,
                num_steps=num_steps,
                tracked_weights=tracked,
                device=device_t,
                dtype=torch_dtype,
            )
            for ff_name in score_buf:
                score_buf[ff_name] += local[ff_name]

        # Send results back via queue (move to CPU for serialization)
        cpu_result = {
            ff_name: score_buf[ff_name].cpu() for ff_name in score_buf
        }
        result_q.put(cpu_result)

    # Spawn workers
    dtype_str_map = {torch.float16: "float16", torch.bfloat16: "bfloat16", torch.float32: "float32"}
    dtype_str = dtype_str_map.get(dtype, "float16")

    print(f"\n[Multi-GPU CAD] Spawning {n_gpus} workers, model: {model_id}", flush=True)
    print(f"[Multi-GPU CAD] Total work items: {len(work_items)}, distributed as: {[len(w) for w in gpu_work]}", flush=True)

    processes = []
    for gpu_idx in range(n_gpus):
        device = devices[gpu_idx]
        p = mp.Process(
            target=_worker,
            args=(
                model_id,
                backend.model_type,
                height,
                width,
                num_steps,
                dtype_str,
                gpu_work[gpu_idx],
                device,
                result_queues[gpu_idx],
            ),
        )
        p.start()
        processes.append(p)
        print(f"[Multi-GPU CAD] Started worker for GPU {gpu_idx}", flush=True)

    # Collect results
    combined_buffers: Dict[str, torch.Tensor] = {
        ff_name: torch.zeros(weight.shape[1], dtype=torch.float32)
        for ff_name, weight in tracked_weights.items()
    }

    for gpu_idx, q in enumerate(result_queues):
        print(f"[Multi-GPU CAD] Waiting for GPU {gpu_idx} results...", flush=True)
        gpu_result = q.get(timeout=3600)  # 1 hour timeout per GPU
        print(f"[Multi-GPU CAD] Received results from GPU {gpu_idx}", flush=True)
        for ff_name in combined_buffers:
            combined_buffers[ff_name] += gpu_result[ff_name]

    print(f"[Multi-GPU CAD] All results collected, joining processes...", flush=True)
    for p in processes:
        p.join()
    print(f"[Multi-GPU CAD] All processes joined.", flush=True)

    # Normalise and aggregate
    results: Dict[str, CADLayerScores] = {}
    for ff_name, spec in ffn_specs.items():
        raw_scores = combined_buffers[ff_name]
        weight = tracked_weights[ff_name].detach().float()
        column_scale = weight.abs().mean(dim=0).clamp_min(eps)
        channel_scores = raw_scores / column_scale

        topk = min(layer_topk, channel_scores.numel())
        layer_score = float(torch.topk(channel_scores, k=topk).values.mean().item())

        results[ff_name] = CADLayerScores(
            spec=spec,
            channel_scores=channel_scores,
            raw_channel_scores=raw_scores,
            layer_score=layer_score,
        )
    return results


def _get_backend_by_type(model_type: str):
    """Get a backend instance without loading a pipeline (for multi-GPU workers)."""
    from .backend import get_backend
    return get_backend(model_type)


def choose_best_layer(layer_scores: Dict[str, CADLayerScores]) -> CADLayerScores:
    """Return the highest-scoring layer from a ``compute_nudity_cad_scores`` result."""
    return max(layer_scores.values(), key=lambda item: item.layer_score)
