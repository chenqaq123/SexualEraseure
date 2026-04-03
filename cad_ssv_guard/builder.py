"""Guard artifact builder.

Orchestrates the full CAD + SSV pipeline:

1. **CAD localisation** — gradient attribution identifies which FFN layers and
   channels respond most strongly to the target concept.
2. **Layer prior weighting** — architecture-aware Gaussian prior favors
   mid-depth "semantic encoding" layers, avoiding early (layout) and late
   (detail) layers.
3. **SSV scoring** — activation statistics (positive minus negative mean)
   further filter channels that are both concept-sensitive *and* have
   meaningful activation differences.
4. **Artifact assembly** — the top-ranked channels and their steering vectors
   are packed into a :class:`~cad_ssv_guard.artifact.GuardArtifact`.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from .artifact import GuardArtifact, GuardLayer
from .backend import ModelBackend
from .cad import CADLayerScores, compute_nudity_cad_scores
from .ssv import collect_mean_activation, compute_ssv_scores, normalize_nonnegative
from .layer_prior import (
    compute_layer_prior_weights,
    apply_prior_to_scores,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _topk_indices(values: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, values.numel())
    if k <= 0:
        raise ValueError("k must be positive.")
    return torch.topk(values, k=k).indices


def _allocate_layer_budgets(
    ranked_layers,
    total_budget: int,
    min_per_layer: int,
) -> List[int]:
    """Distribute ``total_budget`` channels across layers proportionally to
    their layer scores, guaranteeing at least ``min_per_layer`` per layer."""
    num_layers = len(ranked_layers)
    if num_layers == 0:
        return []

    total_budget = max(total_budget, num_layers)
    min_per_layer = max(1, min(min_per_layer, total_budget // num_layers))
    budgets = [min_per_layer] * num_layers
    remaining = total_budget - sum(budgets)
    if remaining <= 0:
        return budgets

    scores = torch.tensor(
        [layer.layer_score for layer in ranked_layers], dtype=torch.float32
    )
    if float(scores.sum().item()) <= 0.0:
        scores = torch.ones_like(scores) / num_layers
    else:
        scores = scores / scores.sum()

    fractional = scores * remaining
    floor_alloc = torch.floor(fractional).to(torch.int64)
    budgets = [b + int(e) for b, e in zip(budgets, floor_alloc.tolist())]

    leftover = remaining - int(floor_alloc.sum().item())
    if leftover > 0:
        order = torch.argsort(fractional - floor_alloc.float(), descending=True).tolist()
        for idx in order[:leftover]:
            budgets[idx] += 1
    return budgets


# ─────────────────────────────────────────────────────────────────────────────
# Main builder
# ─────────────────────────────────────────────────────────────────────────────

def build_nudity_guard(
    pipe,
    backend: ModelBackend,
    positive_prompts: List[str],
    negative_prompts: List[str],
    target: str = "nudity",
    concept_prompt: str = "naked",
    base_prompt: str = "",
    cad_steps: int = 50,
    cad_num_samples: int = 4,
    num_inference_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    num_layers: int = 3,
    cad_candidate_topk: int = 256,
    steering_topk: int = 32,
    total_steering_channels: Optional[int] = None,
    min_channels_per_layer: int = 8,
    alpha: float = 1.0,
    seed: int = 0,
    height: Optional[int] = None,
    width: Optional[int] = None,
    use_layer_prior: bool = True,
    prior_strength: float = 1.0,
    cad_devices: Optional[Sequence[torch.device]] = None,
) -> GuardArtifact:
    """Build a concept-erasing guard artifact for SD1, SD3, FLUX, CogVideoX, HunyuanVideo.

    Parameters
    ----------
    pipe:
        Loaded diffusers pipeline.
    backend:
        Model-specific backend (``get_backend("sd1" | "sd3" | "flux" | "cogvideox" | "hunyuanvideo")``).
    positive_prompts:
        Prompts describing the concept to suppress (e.g. nudity).
    negative_prompts:
        Safe/neutral prompts used as the reference baseline.
    target:
        Human-readable label stored in the artifact (e.g. ``"nudity"``).
    concept_prompt / base_prompt:
        Fallback single-pair prompts when the prompt lists are empty.
    cad_steps:
        Number of scheduler timesteps for CAD attribution.
    cad_num_samples:
        Number of noise samples per prompt pair for CAD.
    num_inference_steps:
        Steps used when collecting SSV activation statistics.
        Defaults to ``backend.default_inference_steps``.
    guidance_scale:
        Guidance scale for SSV collection.
        Defaults to ``backend.default_guidance_scale``.
    num_layers:
        How many FFN layers (ranked by CAD layer score) to include.
    cad_candidate_topk:
        Number of top CAD channels considered as candidates per layer.
    steering_topk:
        Per-layer budget hint (total may be adjusted by ``_allocate_layer_budgets``).
    total_steering_channels:
        Hard total budget across all layers.  Defaults to
        ``steering_topk × num_layers``.
    min_channels_per_layer:
        Minimum channel allocation per layer.
    alpha:
        Steering strength stored in the artifact (can be overridden at
        inference time).
    seed:
        Base random seed.
    height / width:
        Resolution used for CAD attribution.
    use_layer_prior:
        If True, apply architecture-aware Gaussian prior to layer scores,
        favoring mid-depth "semantic encoding" layers.
    prior_strength:
        Strength of the layer prior (0.0 = pure data-driven, 1.0 = full prior).

    Returns
    -------
    GuardArtifact
        Ready-to-save artifact with steering vectors for the selected channels.
    """
    num_inference_steps = num_inference_steps or backend.default_inference_steps
    guidance_scale = guidance_scale or backend.default_guidance_scale
    height = height or backend.default_height
    width = width or backend.default_width

    # ── Step 1 : CAD — rank layers and channels by concept sensitivity ────────
    layer_scores = compute_nudity_cad_scores(
        pipe=pipe,
        backend=backend,
        positive_prompts=positive_prompts,
        negative_prompts=negative_prompts,
        concept_prompt=concept_prompt,
        base_prompt=base_prompt,
        num_steps=cad_steps,
        num_samples=cad_num_samples,
        seed=seed,
        height=height,
        width=width,
        devices=cad_devices,
    )

    backbone = backend.get_backbone(pipe)
    module_lookup = dict(backbone.named_modules())

    # Apply architecture-aware layer prior to favor semantic encoding layers
    if use_layer_prior:
        layer_names = [ls.spec.ff_name for ls in layer_scores.values()]
        prior_weights = compute_layer_prior_weights(
            layer_names=layer_names,
            model_type=backend.model_type,
        )
        raw_scores = {
            ls.spec.ff_name: ls.layer_score
            for ls in layer_scores.values()
        }
        adjusted_scores = apply_prior_to_scores(
            layer_scores=raw_scores,
            prior_weights=prior_weights,
            prior_strength=prior_strength,
        )
        # Create copies of CADLayerScores with adjusted scores for ranking
        adjusted_layer_scores = []
        for ls in layer_scores.values():
            adjusted_ls = CADLayerScores(
                spec=ls.spec,
                channel_scores=ls.channel_scores,
                raw_channel_scores=ls.raw_channel_scores,
                layer_score=adjusted_scores.get(ls.spec.ff_name, ls.layer_score),
            )
            adjusted_layer_scores.append(adjusted_ls)
        ranked_layers = sorted(
            adjusted_layer_scores,
            key=lambda x: x.layer_score,
            reverse=True,
        )
    else:
        ranked_layers = sorted(
            layer_scores.values(),
            key=lambda x: x.layer_score,
            reverse=True,
        )
    ranked_layers = ranked_layers[: min(num_layers, len(ranked_layers))]

    if total_steering_channels is None:
        total_steering_channels = steering_topk * len(ranked_layers)

    budgets = _allocate_layer_budgets(
        ranked_layers=ranked_layers,
        total_budget=total_steering_channels,
        min_per_layer=min_channels_per_layer,
    )

    # ── Step 2 : SSV — collect activation statistics and build steering vectors
    # For video models, use reduced frame count during SSV collection
    is_video = hasattr(backend, "cad_num_frames")
    ssv_num_frames = getattr(backend, "cad_num_frames", None) if is_video else None

    guard_layers: List[GuardLayer] = []
    for layer_idx, (layer, layer_budget) in enumerate(zip(ranked_layers, budgets)):
        hidden_module = module_lookup[layer.spec.hidden_module_name]

        positive_stats = collect_mean_activation(
            pipe=pipe,
            module=hidden_module,
            prompts=positive_prompts,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            num_frames=ssv_num_frames,
        )
        negative_stats = collect_mean_activation(
            pipe=pipe,
            module=hidden_module,
            prompts=negative_prompts,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed + 10_000 + layer_idx * 1_000,
            num_frames=ssv_num_frames,
        )

        # Combine CAD and SSV scores: prefer channels that are *both*
        # concept-sensitive (CAD) *and* have large activation differences (SSV).
        cad_scores = normalize_nonnegative(layer.channel_scores)
        ssv_scores = normalize_nonnegative(
            compute_ssv_scores(positive_stats.mean, negative_stats.mean)
        )
        candidate_indices = _topk_indices(cad_scores, cad_candidate_topk)
        combined_scores = cad_scores[candidate_indices] * ssv_scores[candidate_indices]
        top_local = _topk_indices(combined_scores, layer_budget)
        selected_channels = candidate_indices[top_local]

        # Steering vector: mean positive activation minus mean negative activation
        # projected onto the selected channel subset.
        steering_vector = (
            (positive_stats.mean - negative_stats.mean)[selected_channels].float()
        )

        guard_layers.append(
            GuardLayer(
                ff_name=layer.spec.ff_name,
                hidden_module_name=layer.spec.hidden_module_name,
                projection_module_name=layer.spec.projection_module_name,
                selected_channels=selected_channels.tolist(),
                cad_candidate_channels=candidate_indices.tolist(),
                steering_vector=steering_vector.tolist(),
                layer_score=layer.layer_score,
                positive_hook_calls=positive_stats.count,
                negative_hook_calls=negative_stats.count,
            )
        )

    # ── Step 3 : Pack into artifact ───────────────────────────────────────────
    return GuardArtifact(
        model_id=pipe.config._name_or_path,
        target=target,
        base_prompt=base_prompt,
        layers=guard_layers,
        alpha=alpha,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        cad_steps=cad_steps,
        cad_num_samples=cad_num_samples,
        positive_prompts=positive_prompts,
        negative_prompts=negative_prompts,
        metadata={
            "model_type": backend.model_type,
            "backbone": "transformer" if hasattr(pipe, "transformer") else "unet",
            "selected_ffn_layers": [layer.ff_name for layer in guard_layers],
            "num_layers": len(guard_layers),
            "layer_budgets": budgets,
            "cad_layer_score_type": "normalized_topk_mean",
            "cad_channel_normalization": "raw_score / mean_abs_weight_column",
            "use_layer_prior": use_layer_prior,
            "prior_strength": prior_strength,
        },
    )
