from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
import torch.nn.functional as F


def reduce_hidden_activation(hidden: torch.Tensor) -> torch.Tensor:
    if hidden.ndim == 4:
        return hidden.float().mean(dim=(0, 2, 3))
    if hidden.ndim == 3:
        return hidden.float().mean(dim=(0, 1))
    if hidden.ndim == 2:
        return hidden.float().mean(dim=0)
    if hidden.ndim == 1:
        return hidden.float()
    raise ValueError(f"Unsupported activation shape: {tuple(hidden.shape)}")


@dataclass
class ActivationStats:
    mean: torch.Tensor
    count: int


class ActivationCollector:
    def __init__(self, module):
        self.module = module
        self.sum = None
        self.count = 0
        self.handle = None

    def _hook(self, _module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        reduced = reduce_hidden_activation(hidden)
        reduced_cpu = reduced.detach().to(dtype=torch.float32, device="cpu")
        if self.sum is None:
            self.sum = torch.zeros_like(reduced_cpu, dtype=torch.float32, device="cpu")
        self.sum += reduced_cpu
        self.count += 1
        return output

    def __enter__(self):
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()

    def finalize(self) -> ActivationStats:
        if self.sum is None or self.count == 0:
            raise RuntimeError("No activations were collected.")
        return ActivationStats(mean=self.sum / self.count, count=self.count)


def collect_mean_activation(
    pipe,
    module,
    prompts: Iterable[str],
    num_inference_steps: int,
    guidance_scale: float,
    seed: int = 0,
) -> ActivationStats:
    with ActivationCollector(module) as collector:
        for idx, prompt in enumerate(prompts):
            generator = torch.Generator(device=pipe.device).manual_seed(seed + idx)
            pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="latent",
            )
    return collector.finalize()


def compute_ssv_scores(
    positive_mean: torch.Tensor,
    negative_mean: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    return (positive_mean - negative_mean).abs() / negative_mean.abs().clamp_min(eps)


def cosine_alignment(features: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(features, vector.expand_as(features), dim=-1)


def normalize_nonnegative(values: torch.Tensor) -> torch.Tensor:
    values = values.clamp_min(0.0)
    if float(values.max().item()) == 0.0:
        return torch.zeros_like(values)
    return values / values.max()
