"""Pipeline factories and guard registration.

``create_pipeline`` is the single entry-point for loading any supported model;
it dispatches on ``model_type`` so callers never need to import diffusers
pipeline classes directly.
"""

from __future__ import annotations

from typing import Optional

import torch
from diffusers import (
    CogVideoXPipeline,
    DDIMScheduler,
    FluxPipeline,
    HunyuanVideoPipeline,
    StableDiffusion3Pipeline,
    StableDiffusionPipeline,
)

from .artifact import GuardArtifact
from .steering import MultiLayerSparseSteeringController


# ─────────────────────────────────────────────────────────────────────────────
# Per-model loaders
# ─────────────────────────────────────────────────────────────────────────────

def create_sd_pipeline(
    model_id: str,
    device: str,
    torch_dtype: torch.dtype = torch.float16,
):
    """Load a Stable Diffusion 1.x pipeline with a DDIM scheduler."""
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        safety_checker=None,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe.set_progress_bar_config(disable=False)
    return pipe


def create_sd3_pipeline(
    model_id: str,
    device: str,
    torch_dtype: torch.dtype = torch.bfloat16,
):
    """Load a Stable Diffusion 3 pipeline.

    Uses ``bfloat16`` by default (recommended for SD3 on Ampere/Ada GPUs).
    Keeps the model's native FlowMatchEulerDiscreteScheduler.
    """
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
    ).to(device)
    pipe.set_progress_bar_config(disable=False)
    return pipe


def create_flux_pipeline(
    model_id: str,
    device: str,
    torch_dtype: torch.dtype = torch.bfloat16,
):
    """Load a FLUX.1 pipeline (dev or schnell).

    Uses ``bfloat16`` by default.  The guidance embedding mode
    (dev vs schnell) is inferred automatically from the model config.
    """
    pipe = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
    ).to(device)
    pipe.set_progress_bar_config(disable=False)
    return pipe


# ─────────────────────────────────────────────────────────────────────────────
# Unified factory
# ─────────────────────────────────────────────────────────────────────────────

_DTYPE_DEFAULTS = {
    "sd1": torch.float16,
    "sd3": torch.bfloat16,
    "flux": torch.bfloat16,
}


def create_pipeline(
    model_type: str,
    model_id: str,
    device: str,
    torch_dtype: Optional[torch.dtype] = None,
):
    """Unified pipeline factory dispatching on *model_type*.

    Parameters
    ----------
    model_type : ``"sd1"`` | ``"sd3"`` | ``"flux"``
    model_id :
        HuggingFace Hub model ID or local path, e.g.
        ``"CompVis/stable-diffusion-v1-4"``,
        ``"stabilityai/stable-diffusion-3-medium-diffusers"``,
        ``"black-forest-labs/FLUX.1-dev"``.
    device :
        PyTorch device string, e.g. ``"cuda:0"`` or ``"cpu"``.
    torch_dtype :
        Override the default dtype for the model type.
    """
    key = model_type.lower()
    dtype = torch_dtype or _DTYPE_DEFAULTS.get(key, torch.float16)

    if key == "sd1":
        return create_sd_pipeline(model_id, device, dtype)
    elif key == "sd3":
        return create_sd3_pipeline(model_id, device, dtype)
    elif key == "flux":
        return create_flux_pipeline(model_id, device, dtype)
    else:
        raise ValueError(
            f"Unknown model_type {model_type!r}.  "
            "Valid choices: sd1, sd3, flux"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Guard registration
# ─────────────────────────────────────────────────────────────────────────────

def register_guard(
    pipe,
    artifact: GuardArtifact,
    device: Optional[torch.device] = None,
    alpha: Optional[float] = None,
) -> MultiLayerSparseSteeringController:
    """Attach forward hooks that apply the learned steering vectors during
    inference and return a controller object.

    The controller exposes:

    * ``controller.set_enabled(bool)`` — toggle steering per-step.
    * ``controller.remove()``          — detach all hooks when done.

    Works for SD1, SD3, and FLUX pipelines without any model-specific logic
    because the hook attachment goes through
    :func:`~cad_ssv_guard.steering._get_backbone`, which handles both
    ``pipe.unet`` and ``pipe.transformer`` transparently.
    """
    controller = MultiLayerSparseSteeringController.from_artifact(
        artifact=artifact,
        device=device,
        alpha=alpha,
    )
    return controller.register(pipe)
