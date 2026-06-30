from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Activation reduction helpers
# ─────────────────────────────────────────────────────────────────────────────

def reduce_hidden_activation(
    hidden: torch.Tensor,
    cond_only: bool = True,
) -> torch.Tensor:
    """Reduce an FFN hidden activation tensor to a single channel vector.

    Parameters
    ----------
    hidden : Tensor
        Raw activation from a forward hook.  Common shapes:
        - (B, seq, d_ff)  — DiT / transformer (SD3, FLUX, CogVideoX …)
        - (B, H*W, d_ff)  — UNet spatial attention
        - (B, d_ff)        — fully-collapsed
        - (d_ff,)          — already reduced
    cond_only : bool
        When True **and** the batch dimension is > 1, only the last batch
        element is kept before averaging.  For CFG models (SD1, SD3) this
        discards the unconditional pass; for FLUX (no CFG, batch=1) this
        has no effect.

    Returns
    -------
    Tensor of shape (d_ff,).
    """
    # Step 1 – isolate conditional pass for CFG models
    if cond_only and hidden.ndim >= 2 and hidden.shape[0] > 1:
        hidden = hidden[-1:]          # (1, ..., d_ff)

    # Step 2 – average over every leading dimension (batch, seq / spatial)
    if hidden.ndim > 1:
        dims = tuple(range(hidden.ndim - 1))
        return hidden.float().mean(dim=dims)   # (d_ff,)
    return hidden.float()


# ─────────────────────────────────────────────────────────────────────────────
# Activation collector
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ActivationStats:
    mean: torch.Tensor
    count: int


class ActivationCollector:
    """Forward-hook based activation accumulator.

    Supports two usage modes:

    1. **Aggregate** (``collect_mean_activation``):
       Register once, let all inference calls accumulate into a single mean.

    2. **Per-run** (``collect_activations_per_sample``):
       Register once, call ``reset()`` between runs and ``finalize()`` after
       each run to extract per-run mean vectors.
    """

    def __init__(self, module: torch.nn.Module, cond_only: bool = True):
        self.module = module
        self.cond_only = cond_only
        self._sum: Optional[torch.Tensor] = None
        self._count: int = 0
        self.handle = None

    # ── Public control ────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear accumulators (call between per-run collections)."""
        self._sum = None
        self._count = 0

    def finalize(self) -> ActivationStats:
        """Return mean activation and hook-call count since last reset."""
        if self._sum is None or self._count == 0:
            raise RuntimeError("No activations were collected since last reset.")
        return ActivationStats(mean=self._sum / self._count, count=self._count)

    # ── Hook ─────────────────────────────────────────────────────────────────

    def _hook(self, _module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        reduced = reduce_hidden_activation(hidden, cond_only=self.cond_only)
        reduced_cpu = reduced.detach().cpu().float()
        if self._sum is None:
            self._sum = torch.zeros_like(reduced_cpu)
        self._sum += reduced_cpu
        self._count += 1
        return output

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self):
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


# ─────────────────────────────────────────────────────────────────────────────
# Aggregated collection (legacy / channel-selection SSV)
# ─────────────────────────────────────────────────────────────────────────────

def collect_mean_activation(
    pipe,
    module,
    prompts: Iterable[str],
    num_inference_steps: int,
    guidance_scale: float,
    seed: int = 0,
    num_frames: Optional[int] = None,
    cond_only: bool = True,
) -> ActivationStats:
    """Collect a single mean activation vector averaged over all prompts and
    all denoising timesteps.

    Used for SSV channel-selection scoring (where per-prompt variance is not
    needed, only the mean difference between positive and negative sets).
    """
    with ActivationCollector(module, cond_only=cond_only) as collector:
        for idx, prompt in enumerate(prompts):
            generator = torch.Generator(device=pipe.device).manual_seed(seed + idx)
            kwargs = dict(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="latent",
            )
            if num_frames is not None:
                kwargs["num_frames"] = num_frames
            pipe(**kwargs)
    return collector.finalize()


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample collection (for PCA concept-direction estimation)
# ─────────────────────────────────────────────────────────────────────────────

def collect_activations_per_sample(
    pipe,
    module,
    prompts: List[str],
    num_inference_steps: int,
    guidance_scale: float,
    seeds: List[int],
    num_frames: Optional[int] = None,
    cond_only: bool = True,
) -> torch.Tensor:
    """Collect one activation vector per (prompt, seed) combination.

    Each vector is the mean over all denoising timesteps for that single
    inference run.  Different seeds give different noise initialisations,
    sampling diverse trajectories through activation space and enlarging
    the concept-diff matrix used for PCA.

    Parameters
    ----------
    prompts : list of str
        Positive or negative prompt strings.
    seeds : list of int
        Random seeds to iterate over for each prompt.
    cond_only : bool
        If True (default), discard the unconditional CFG pass (batch[-1]
        is kept).  Has no effect for FLUX (single-pass, batch=1).

    Returns
    -------
    Tensor of shape (N_prompts × N_seeds, d_ff).
    Row order: prompt index varies slowly, seed index varies fast.
    E.g. for 3 prompts × 4 seeds → 12 rows ordered as
    (p0,s0), (p0,s1), (p0,s2), (p0,s3), (p1,s0), …
    """
    samples: List[torch.Tensor] = []

    with ActivationCollector(module, cond_only=cond_only) as collector:
        total = len(prompts) * len(seeds)
        done = 0
        for prompt in prompts:
            for seed in seeds:
                collector.reset()
                generator = torch.Generator(device=pipe.device).manual_seed(seed)
                kwargs = dict(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    output_type="latent",
                )
                if num_frames is not None:
                    kwargs["num_frames"] = num_frames
                pipe(**kwargs)
                stats = collector.finalize()
                samples.append(stats.mean)       # (d_ff,)
                done += 1
                if done % max(1, total // 5) == 0 or done == total:
                    print(f"    [SSV] collected {done}/{total} samples", flush=True)

    return torch.stack(samples, dim=0)           # (N_prompts × N_seeds, d_ff)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-layer parallel collection (amortises forward passes across all layers)
# ─────────────────────────────────────────────────────────────────────────────

def collect_activations_per_sample_multilayer(
    pipe,
    modules: List[nn.Module],
    prompts: List[str],
    num_inference_steps: int,
    guidance_scale: float,
    seeds: List[int],
    num_frames: Optional[int] = None,
    cond_only: bool = True,
) -> List[torch.Tensor]:
    """Collect per-sample activation vectors for *all* modules in one pass.

    Instead of running ``N_layers`` independent sets of forward passes (one per
    module), this function registers hooks on every module simultaneously and
    runs exactly ``N_prompts × N_seeds`` forward passes, regardless of how many
    modules are requested.

    Parameters
    ----------
    modules : list of nn.Module
        FFN hidden-layer modules to monitor (one per selected layer).
    prompts : list of str
        Positive *or* negative prompt strings for this collection phase.
    seeds : list of int
        Random seeds; each prompt is run once per seed.
    cond_only : bool
        Discard the unconditional CFG pass for CFG models (SD1, SD3).
        Has no effect for FLUX (batch=1).

    Returns
    -------
    List of Tensors, one per module, each of shape
    ``(N_prompts × N_seeds, d_ff)``.
    Row order mirrors :func:`collect_activations_per_sample`:
    prompt index varies slowly, seed index varies fast.
    """
    collectors = [ActivationCollector(m, cond_only=cond_only) for m in modules]
    # samples_per_module[i] accumulates one (d_ff,) tensor per (prompt, seed)
    samples_per_module: List[List[torch.Tensor]] = [[] for _ in modules]

    with ExitStack() as stack:
        for collector in collectors:
            stack.enter_context(collector)   # registers all hooks at once

        total = len(prompts) * len(seeds)
        done = 0
        for prompt in prompts:
            for seed in seeds:
                for collector in collectors:
                    collector.reset()

                generator = torch.Generator(device=pipe.device).manual_seed(seed)
                kwargs = dict(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    output_type="latent",
                )
                if num_frames is not None:
                    kwargs["num_frames"] = num_frames
                pipe(**kwargs)

                for i, collector in enumerate(collectors):
                    stats = collector.finalize()
                    samples_per_module[i].append(stats.mean)   # (d_ff,)

                done += 1
                if done % max(1, total // 5) == 0 or done == total:
                    print(f"    [SSV] collected {done}/{total} samples "
                          f"({len(modules)} layers in parallel)", flush=True)

    return [torch.stack(s, dim=0) for s in samples_per_module]


# ─────────────────────────────────────────────────────────────────────────────
# Scoring utilities
# ─────────────────────────────────────────────────────────────────────────────

def compute_ssv_scores(
    positive_mean: torch.Tensor,
    negative_mean: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Relative activation difference: |pos - neg| / |neg|."""
    return (positive_mean - negative_mean).abs() / negative_mean.abs().clamp_min(eps)


def cosine_alignment(features: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(features, vector.expand_as(features), dim=-1)


def normalize_nonnegative(values: torch.Tensor) -> torch.Tensor:
    values = values.clamp_min(0.0)
    if float(values.max().item()) == 0.0:
        return torch.zeros_like(values)
    return values / values.max()


def normalize_signed(values: torch.Tensor) -> torch.Tensor:
    """Normalize signed scores to [-1, 1] range.

    Preserves the sign information: positive values indicate concept-amplifying
    channels, negative values indicate concept-suppressing channels.
    Both are useful for concept erasure.
    """
    max_abs = values.abs().max()
    if max_abs.item() < 1e-10:
        return torch.zeros_like(values)
    return values / max_abs
