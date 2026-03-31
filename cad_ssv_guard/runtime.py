from __future__ import annotations

from typing import Optional

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

from .artifact import GuardArtifact
from .steering import MultiLayerSparseSteeringController


def create_sd_pipeline(
    model_id: str,
    device: str,
    torch_dtype: torch.dtype = torch.float16,
):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        safety_checker=None,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe.set_progress_bar_config(disable=False)
    return pipe


def register_guard(
    pipe,
    artifact: GuardArtifact,
    device: Optional[torch.device] = None,
    alpha: Optional[float] = None,
):
    controller = MultiLayerSparseSteeringController.from_artifact(
        artifact=artifact,
        device=device,
        alpha=alpha,
    )
    return controller.register(pipe)
