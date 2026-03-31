from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from .ffn import FFNModuleSpec, build_parameter_lookup, discover_ffn_specs, get_projection_weight_name


@dataclass
class CADLayerScores:
    spec: FFNModuleSpec
    channel_scores: torch.Tensor
    raw_channel_scores: torch.Tensor
    layer_score: float


def _encode_prompts(pipe, prompts: List[str], device: torch.device) -> torch.Tensor:
    text_inputs = pipe.tokenizer(
        prompts,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        return pipe.text_encoder(text_inputs.input_ids.to(device))[0]


def _build_prompt_pairs(
    positive_prompts: Sequence[str] | None,
    negative_prompts: Sequence[str] | None,
    concept_prompt: str,
    base_prompt: str,
) -> List[Tuple[str, str]]:
    if positive_prompts and negative_prompts:
        pair_count = min(len(positive_prompts), len(negative_prompts))
        if pair_count == 0:
            raise ValueError("Prompt pair count must be positive.")
        return [(positive_prompts[i], negative_prompts[i]) for i in range(pair_count)]
    return [(concept_prompt, base_prompt)]


def compute_nudity_cad_scores(
    pipe,
    positive_prompts: Sequence[str] | None = None,
    negative_prompts: Sequence[str] | None = None,
    concept_prompt: str = "naked",
    base_prompt: str = "",
    num_steps: int = 50,
    num_samples: int = 4,
    guidance_scale: float = 3.0,
    seed: int = 0,
    height: int = 512,
    width: int = 512,
    layer_topk: int = 64,
    eps: float = 1e-6,
) -> Dict[str, CADLayerScores]:
    del guidance_scale

    device = pipe.device
    pipe.scheduler.set_timesteps(num_steps, device=device)
    prompt_pairs = _build_prompt_pairs(
        positive_prompts=positive_prompts,
        negative_prompts=negative_prompts,
        concept_prompt=concept_prompt,
        base_prompt=base_prompt,
    )

    parameter_lookup = build_parameter_lookup(pipe.unet)
    ffn_specs = discover_ffn_specs(pipe.unet)
    tracked_weights = {}
    for spec in ffn_specs.values():
        weight_name = get_projection_weight_name(spec)
        tracked_weights[spec.ff_name] = parameter_lookup[weight_name]

    score_buffers = {
        ff_name: torch.zeros(weight.shape[1], dtype=torch.float32, device=device)
        for ff_name, weight in tracked_weights.items()
    }

    latent_shape = (
        1,
        pipe.unet.config.in_channels,
        height // pipe.vae_scale_factor,
        width // pipe.vae_scale_factor,
    )

    for pair_idx, (positive_prompt, negative_prompt) in enumerate(prompt_pairs):
        text_embeddings = _encode_prompts(pipe, [positive_prompt, negative_prompt], device=device)
        for sample_idx in range(num_samples):
            sample_seed = seed + pair_idx * 10_000 + sample_idx
            generator = torch.Generator(device=device).manual_seed(sample_seed)
            latents = torch.randn(latent_shape, generator=generator, device=device, dtype=pipe.unet.dtype)

            for timestep in pipe.scheduler.timesteps:
                latent_input = pipe.scheduler.scale_model_input(latents, timestep)
                latent_input = latent_input.repeat(2, 1, 1, 1)

                noise_pred = pipe.unet(
                    latent_input,
                    timestep,
                    encoder_hidden_states=text_embeddings,
                ).sample

                objective = F.mse_loss(
                    noise_pred[0].float(),
                    noise_pred[1].detach().float(),
                    reduction="sum",
                )

                pipe.unet.zero_grad(set_to_none=True)
                objective.backward()

                for ff_name, weight in tracked_weights.items():
                    grad = weight.grad
                    if grad is None:
                        continue
                    attribution = (weight.detach().float() * grad.detach().float()).clamp_min(0.0)
                    score_buffers[ff_name] += attribution.sum(dim=0)

                # Update latents along the nudity trajectory for next timestep.
                # Without this, every timestep receives the same pure-noise input,
                # making the gradient attribution completely unreliable.
                with torch.no_grad():
                    latents = pipe.scheduler.step(
                        noise_pred[0].detach(), timestep, latents
                    ).prev_sample

    results = {}
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


def choose_best_layer(layer_scores: Dict[str, CADLayerScores]) -> CADLayerScores:
    return max(layer_scores.values(), key=lambda item: item.layer_score)
