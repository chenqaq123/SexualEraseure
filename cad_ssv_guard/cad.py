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
) -> Dict[str, CADLayerScores]:
    """Compute CAD channel attribution scores for all FFN projection layers.

    For each prompt pair and each denoising timestep we:

    1. Run a forward pass conditioned on the positive (nudity) prompt and on
       the negative (safe) prompt.  For CFG-capable models (SD1, SD3) this is
       a single batch-2 call; for FLUX it is two separate forward passes.
    2. Compute the MSE between the two predictions as a proxy attribution loss.
    3. Backpropagate and accumulate ``weight * grad`` attributions for every
       FFN projection matrix.

    The raw attributions are column-normalised by the mean absolute magnitude
    of each weight column, yielding scale-invariant per-channel scores.

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

    Returns
    -------
    Dict mapping ``ff_name → CADLayerScores`` for every discovered FFN layer.
    """
    height = height or backend.default_height
    width = width or backend.default_width

    device = pipe.device
    pipe.scheduler.set_timesteps(num_steps, device=device)

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

    # Accumulate attributions in float32 on the device
    score_buffers: Dict[str, torch.Tensor] = {
        ff_name: torch.zeros(weight.shape[1], dtype=torch.float32, device=device)
        for ff_name, weight in tracked_weights.items()
    }

    dtype = next(backbone.parameters()).dtype

    # ── Main attribution loop ─────────────────────────────────────────────────
    for pair_idx, (pos_prompt, neg_prompt) in enumerate(prompt_pairs):
        text_dict = backend.encode_prompt_pair(pipe, pos_prompt, neg_prompt, device)

        for sample_idx in range(num_samples):
            sample_seed = seed + pair_idx * 10_000 + sample_idx
            generator = torch.Generator(device=device).manual_seed(sample_seed)
            latents_dict = backend.prepare_latents(
                pipe, height, width, device, dtype, generator
            )

            for timestep in pipe.scheduler.timesteps:
                noise_pred_pos, noise_pred_neg = backend.cad_forward(
                    pipe, latents_dict, text_dict, timestep
                )

                # Squared-error loss: how much does the positive prediction
                # differ from the negative one at this timestep?
                objective = F.mse_loss(
                    noise_pred_pos.float(),
                    noise_pred_neg.float(),
                    reduction="sum",
                )

                backbone.zero_grad(set_to_none=True)
                objective.backward()

                # Gradient attribution: ∑_row (W ⊙ ∇W).clamp(0) summed over
                # the output dimension → one score per input (hidden) channel.
                for ff_name, weight in tracked_weights.items():
                    grad = weight.grad
                    if grad is None:
                        continue
                    attribution = (
                        weight.detach().float() * grad.detach().float()
                    ).clamp_min(0.0)
                    score_buffers[ff_name] += attribution.sum(dim=0)

                # Advance the latent trajectory for the next timestep.
                # Without this, every step receives the same pure-noise input
                # and the attribution degenerates to a single-timestep estimate.
                with torch.no_grad():
                    latents_dict = backend.scheduler_step(
                        pipe, noise_pred_pos, timestep, latents_dict
                    )

    # ── Normalise and aggregate ───────────────────────────────────────────────
    results: Dict[str, CADLayerScores] = {}
    for ff_name, spec in ffn_specs.items():
        raw_scores = score_buffers[ff_name].detach()
        weight = tracked_weights[ff_name].detach().float()
        # Column-normalise by mean absolute weight magnitude to make scores
        # scale-invariant across layers with different weight norms.
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


def choose_best_layer(layer_scores: Dict[str, CADLayerScores]) -> CADLayerScores:
    """Return the highest-scoring layer from a ``compute_nudity_cad_scores`` result."""
    return max(layer_scores.values(), key=lambda item: item.layer_score)
