from __future__ import annotations

from typing import List

import torch

from .artifact import GuardArtifact, GuardLayer
from .cad import compute_nudity_cad_scores
from .ssv import collect_mean_activation, compute_ssv_scores, normalize_nonnegative


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
    num_layers = len(ranked_layers)
    if num_layers == 0:
        return []

    total_budget = max(total_budget, num_layers)
    min_per_layer = max(1, min(min_per_layer, total_budget // num_layers))
    budgets = [min_per_layer for _ in range(num_layers)]
    remaining = total_budget - sum(budgets)
    if remaining <= 0:
        return budgets

    scores = torch.tensor([layer.layer_score for layer in ranked_layers], dtype=torch.float32)
    if float(scores.sum().item()) <= 0.0:
        scores = torch.ones_like(scores) / num_layers
    else:
        scores = scores / scores.sum()

    fractional = scores * remaining
    floor_alloc = torch.floor(fractional).to(torch.int64)
    budgets = [budget + int(extra) for budget, extra in zip(budgets, floor_alloc.tolist())]
    leftover = remaining - int(floor_alloc.sum().item())
    if leftover > 0:
        order = torch.argsort(fractional - floor_alloc.float(), descending=True).tolist()
        for idx in order[:leftover]:
            budgets[idx] += 1
    return budgets


def build_nudity_guard(
    pipe,
    positive_prompts: List[str],
    negative_prompts: List[str],
    target: str = "nudity",
    concept_prompt: str = "naked",
    base_prompt: str = "",
    cad_steps: int = 50,
    cad_num_samples: int = 4,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    num_layers: int = 3,
    cad_candidate_topk: int = 256,
    steering_topk: int = 32,
    total_steering_channels: int | None = None,
    min_channels_per_layer: int = 8,
    alpha: float = 2.0,
    seed: int = 0,
) -> GuardArtifact:
    layer_scores = compute_nudity_cad_scores(
        pipe=pipe,
        positive_prompts=positive_prompts,
        negative_prompts=negative_prompts,
        concept_prompt=concept_prompt,
        base_prompt=base_prompt,
        num_steps=cad_steps,
        num_samples=cad_num_samples,
        guidance_scale=guidance_scale,
        seed=seed,
    )
    module_lookup = dict(pipe.unet.named_modules())
    ranked_layers = sorted(
        layer_scores.values(),
        key=lambda item: item.layer_score,
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

    guard_layers = []
    for layer_idx, (layer, layer_budget) in enumerate(zip(ranked_layers, budgets)):
        hidden_module = module_lookup[layer.spec.hidden_module_name]

        positive_stats = collect_mean_activation(
            pipe=pipe,
            module=hidden_module,
            prompts=positive_prompts,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        negative_stats = collect_mean_activation(
            pipe=pipe,
            module=hidden_module,
            prompts=negative_prompts,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed + 10_000 + layer_idx * 1_000,
        )

        cad_scores = normalize_nonnegative(layer.channel_scores)
        ssv_scores = normalize_nonnegative(
            compute_ssv_scores(positive_stats.mean, negative_stats.mean)
        )
        candidate_indices = _topk_indices(cad_scores, cad_candidate_topk)
        combined_scores = cad_scores[candidate_indices] * ssv_scores[candidate_indices]
        top_local = _topk_indices(combined_scores, layer_budget)
        selected_channels = candidate_indices[top_local]
        steering_vector = (positive_stats.mean - negative_stats.mean)[selected_channels].float()

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
            "selected_ffn_layers": [layer.ff_name for layer in guard_layers],
            "num_layers": len(guard_layers),
            "layer_budgets": budgets,
            "cad_layer_score_type": "normalized_topk_mean",
            "cad_channel_normalization": "raw_score / mean_abs_weight_column",
        },
    )
