"""Model-specific backend implementations for CAD gradient attribution.

Each backend encapsulates the differences between:
  - SD1  : UNet, single CLIP encoder, standard CFG (batch of [uncond, cond])
  - SD3  : MMDiT transformer, triple encoder (CLIP-L/G + T5), standard CFG
  - FLUX : DiT transformer, dual encoder (CLIP-L + T5), guidance distillation
            — NO standard CFG, so CAD requires two separate forward passes.

The rest of the pipeline (FFN discovery, SSV statistics, steering hooks) is
model-agnostic and shared across all backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class ModelBackend(ABC):
    """Abstract interface for model-specific forward-pass logic used in CAD scoring."""

    model_type: str  # "sd1" | "sd3" | "flux"

    # ── Capability flags ──────────────────────────────────────────────────────

    @property
    @abstractmethod
    def supports_cfg(self) -> bool:
        """True when the model uses a [uncond, cond] batch for guidance."""
        ...

    # ── Default hyper-parameters (can be overridden from CLI) ─────────────────

    @property
    @abstractmethod
    def default_height(self) -> int: ...

    @property
    @abstractmethod
    def default_width(self) -> int: ...

    @property
    @abstractmethod
    def default_guidance_scale(self) -> float: ...

    @property
    @abstractmethod
    def default_inference_steps(self) -> int: ...

    # ── Backbone accessor ─────────────────────────────────────────────────────

    def get_backbone(self, pipe: Any) -> nn.Module:
        """Return the primary denoising backbone (transformer or unet)."""
        if hasattr(pipe, "transformer"):
            return pipe.transformer
        return pipe.unet

    # ── Model-specific operations ─────────────────────────────────────────────

    @abstractmethod
    def encode_prompt_pair(
        self,
        pipe: Any,
        positive_prompt: str,
        negative_prompt: str,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Encode a (positive, negative) prompt pair into conditioning tensors.

        The returned dict is passed verbatim to :meth:`cad_forward`.
        For CFG-capable models the dict contains a batch-2 tensor
        ([positive, negative]) so a single transformer call suffices.
        For FLUX the dict contains separate pos/neg tensors.
        """
        ...

    @abstractmethod
    def prepare_latents(
        self,
        pipe: Any,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator,
    ) -> Dict[str, Any]:
        """Prepare initial noisy latents (and any ancillary tensors) for one
        CAD attribution trajectory."""
        ...

    @abstractmethod
    def cad_forward(
        self,
        pipe: Any,
        latents_dict: Dict[str, Any],
        text_dict: Dict[str, torch.Tensor],
        timestep: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one denoising forward pass for gradient attribution.

        Returns ``(noise_pred_pos, noise_pred_neg)`` where:

        * ``noise_pred_pos`` — prediction under positive (nudity) conditioning;
          **requires grad** so the caller can call ``.backward()`` on a loss
          that uses this output.
        * ``noise_pred_neg`` — prediction under safe/negative conditioning;
          **detached**, used only as the MSE target.
        """
        ...

    @abstractmethod
    def scheduler_step(
        self,
        pipe: Any,
        noise_pred_pos: torch.Tensor,
        timestep: torch.Tensor,
        latents_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Advance the latent state by one denoising step and return an
        updated ``latents_dict``."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# SD1 backend
# ─────────────────────────────────────────────────────────────────────────────

class SD1Backend(ModelBackend):
    """Stable Diffusion 1.x — UNet, single CLIP encoder, DDIM, standard CFG."""

    model_type = "sd1"

    @property
    def supports_cfg(self) -> bool:
        return True

    @property
    def default_height(self) -> int:
        return 512

    @property
    def default_width(self) -> int:
        return 512

    @property
    def default_guidance_scale(self) -> float:
        return 7.5

    @property
    def default_inference_steps(self) -> int:
        return 30

    def encode_prompt_pair(self, pipe, positive_prompt, negative_prompt, device):
        text_inputs = pipe.tokenizer(
            [positive_prompt, negative_prompt],
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            # Shape: (2, seq_len, hidden_dim) — row 0 = positive, row 1 = negative
            embeds = pipe.text_encoder(text_inputs.input_ids.to(device))[0]
        return {"encoder_hidden_states": embeds}

    def prepare_latents(self, pipe, height, width, device, dtype, generator):
        shape = (
            1,
            pipe.unet.config.in_channels,
            height // pipe.vae_scale_factor,
            width // pipe.vae_scale_factor,
        )
        return {"latents": torch.randn(shape, generator=generator, device=device, dtype=dtype)}

    def cad_forward(self, pipe, latents_dict, text_dict, timestep):
        latents = latents_dict["latents"]
        # Repeat latent for batch-2 [positive_cond, negative_cond]
        latent_input = pipe.scheduler.scale_model_input(latents, timestep).repeat(2, 1, 1, 1)
        noise_pred = pipe.unet(
            latent_input,
            timestep,
            encoder_hidden_states=text_dict["encoder_hidden_states"],
        ).sample  # (2, C, H, W)
        # [0] = positive prediction (with grad), [1] = negative (detach)
        return noise_pred[0:1], noise_pred[1:2].detach()

    def scheduler_step(self, pipe, noise_pred_pos, timestep, latents_dict):
        prev = pipe.scheduler.step(
            noise_pred_pos.detach(), timestep, latents_dict["latents"]
        ).prev_sample
        return {"latents": prev}


# ─────────────────────────────────────────────────────────────────────────────
# SD3 backend
# ─────────────────────────────────────────────────────────────────────────────

class SD3Backend(ModelBackend):
    """Stable Diffusion 3 — MMDiT transformer, CLIP-L/G + T5 encoders,
    FlowMatch scheduler, standard CFG."""

    model_type = "sd3"

    @property
    def supports_cfg(self) -> bool:
        return True

    @property
    def default_height(self) -> int:
        return 1024

    @property
    def default_width(self) -> int:
        return 1024

    @property
    def default_guidance_scale(self) -> float:
        return 7.0

    @property
    def default_inference_steps(self) -> int:
        return 28

    def encode_prompt_pair(self, pipe, positive_prompt, negative_prompt, device):
        with torch.no_grad():
            pos_embeds, neg_embeds, pos_pooled, neg_pooled = pipe.encode_prompt(
                prompt=positive_prompt,
                prompt_2=positive_prompt,
                prompt_3=positive_prompt,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt,
                negative_prompt_3=negative_prompt,
                device=device,
                do_classifier_free_guidance=True,
            )
        # Batch-2: row 0 = positive, row 1 = negative
        return {
            "encoder_hidden_states": torch.cat([pos_embeds, neg_embeds], dim=0),
            "pooled_projections": torch.cat([pos_pooled, neg_pooled], dim=0),
        }

    def prepare_latents(self, pipe, height, width, device, dtype, generator):
        shape = (
            1,
            pipe.transformer.config.in_channels,
            height // pipe.vae_scale_factor,
            width // pipe.vae_scale_factor,
        )
        return {"latents": torch.randn(shape, generator=generator, device=device, dtype=dtype)}

    def cad_forward(self, pipe, latents_dict, text_dict, timestep):
        latents = latents_dict["latents"]
        # FlowMatchEulerDiscreteScheduler has no scale_model_input;
        # SD3 passes latents directly to the transformer.
        latent_input = latents.repeat(2, 1, 1, 1)
        # SD3 transformer expects timestep expanded to the batch dimension
        t = timestep.expand(latent_input.shape[0])
        noise_pred = pipe.transformer(
            hidden_states=latent_input,
            encoder_hidden_states=text_dict["encoder_hidden_states"],
            pooled_projections=text_dict["pooled_projections"],
            timestep=t,
            return_dict=False,
        )[0]  # (2, C, H, W)
        return noise_pred[0:1], noise_pred[1:2].detach()

    def scheduler_step(self, pipe, noise_pred_pos, timestep, latents_dict):
        prev = pipe.scheduler.step(
            noise_pred_pos.detach(), timestep, latents_dict["latents"]
        ).prev_sample
        return {"latents": prev}


# ─────────────────────────────────────────────────────────────────────────────
# FLUX backend
# ─────────────────────────────────────────────────────────────────────────────

def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """Rearrange (B, C, H, W) → (B, H/2·W/2, C·4) for FLUX transformer input.

    FLUX packs every 2×2 spatial patch into the channel dimension before
    feeding the transformer, effectively halving the spatial resolution and
    quadrupling the channel count.
    """
    B, C, H, W = latents.shape
    x = latents.view(B, C, H // 2, 2, W // 2, 2)
    x = x.permute(0, 2, 4, 1, 3, 5)
    return x.reshape(B, (H // 2) * (W // 2), C * 4)


def _prepare_img_ids(
    h_patches: int,
    w_patches: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build FLUX RoPE image position IDs of shape (H·W, 3).

    The first channel is always 0 (image token marker).
    Channels 1 and 2 carry the row / column indices respectively.
    """
    ids = torch.zeros(h_patches, w_patches, 3, device=device, dtype=dtype)
    ids[..., 1] = ids[..., 1] + torch.arange(h_patches, device=device)[:, None]
    ids[..., 2] = ids[..., 2] + torch.arange(w_patches, device=device)[None, :]
    return ids.reshape(h_patches * w_patches, 3)


class FluxBackend(ModelBackend):
    """FLUX.1-dev / FLUX.1-schnell backend.

    FLUX replaces classifier-free guidance with a *guidance distillation*
    mechanism: a scalar guidance embedding is injected directly into the
    transformer, and only a **single** conditioning pass is run per step.

    Because there is no batch-of-2 trick available, CAD attribution requires
    **two separate transformer calls** per timestep:

    1. Positive pass (nudity prompt)  — gradients enabled, used for backprop.
    2. Negative pass (safe prompt)    — ``torch.no_grad()``, MSE target only.

    FLUX latents are packed from (B, 16, H, W) into (B, H/2·W/2, 64) before
    entering the transformer. All scheduler steps operate on packed latents.
    """

    model_type = "flux"

    # FLUX VAE always outputs 16 channels; packing multiplies by 4 → 64.
    _VAE_CHANNELS = 16

    @property
    def supports_cfg(self) -> bool:
        return False  # uses guidance embedding, not batched CFG

    @property
    def default_height(self) -> int:
        return 1024

    @property
    def default_width(self) -> int:
        return 1024

    @property
    def default_guidance_scale(self) -> float:
        # Guidance *embedding* scale for FLUX-dev (ignored by FLUX-schnell).
        return 3.5

    @property
    def default_inference_steps(self) -> int:
        return 28

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_guidance(
        self,
        pipe: Any,
        batch_size: int,
        guidance_scale: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Return a guidance tensor for FLUX-dev, or None for FLUX-schnell."""
        if not pipe.transformer.config.guidance_embeds:
            return None
        return torch.full((batch_size,), guidance_scale, device=device, dtype=dtype)

    def _txt_ids(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """All-zero text position IDs of shape (seq_len, 3)."""
        return torch.zeros(seq_len, 3, device=device, dtype=dtype)

    def _transformer_call(
        self,
        pipe: Any,
        latents: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        txt_ids: torch.Tensor,
        img_ids: torch.Tensor,
        timestep: torch.Tensor,
        guidance: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return pipe.transformer(
            hidden_states=latents,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep / 1000.0,  # FLUX expects t ∈ [0, 1]
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            return_dict=False,
        )[0]

    # ── Backend interface ─────────────────────────────────────────────────────

    def encode_prompt_pair(self, pipe, positive_prompt, negative_prompt, device):
        with torch.no_grad():
            pos_embeds, pos_pooled = pipe.encode_prompt(
                prompt=positive_prompt,
                prompt_2=positive_prompt,
                device=device,
            )
            neg_embeds, neg_pooled = pipe.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt,
                device=device,
            )
        return {
            "pos_encoder_hidden_states": pos_embeds,
            "pos_pooled_projections": pos_pooled,
            "neg_encoder_hidden_states": neg_embeds,
            "neg_pooled_projections": neg_pooled,
        }

    def prepare_latents(self, pipe, height, width, device, dtype, generator):
        h_lat = height // pipe.vae_scale_factor  # e.g. 128 for 1024px
        w_lat = width // pipe.vae_scale_factor
        raw = torch.randn(
            (1, self._VAE_CHANNELS, h_lat, w_lat),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        packed = _pack_latents(raw)
        # img_ids: one position per packed patch → (H/2·W/2, 3)
        img_ids = _prepare_img_ids(h_lat // 2, w_lat // 2, device, dtype)
        return {"latents": packed, "img_ids": img_ids}

    def cad_forward(self, pipe, latents_dict, text_dict, timestep):
        latents = latents_dict["latents"]
        img_ids = latents_dict["img_ids"]
        device, dtype = latents.device, latents.dtype

        guidance = self._make_guidance(pipe, latents.shape[0], 3.5, device, dtype)

        # ── Positive pass: gradients ON ──────────────────────────────────────
        pos_txt_ids = self._txt_ids(
            text_dict["pos_encoder_hidden_states"].shape[1], device, dtype
        )
        noise_pred_pos = self._transformer_call(
            pipe,
            latents,
            text_dict["pos_encoder_hidden_states"],
            text_dict["pos_pooled_projections"],
            pos_txt_ids,
            img_ids,
            timestep,
            guidance,
        )

        # ── Negative pass: no gradients ──────────────────────────────────────
        with torch.no_grad():
            neg_txt_ids = self._txt_ids(
                text_dict["neg_encoder_hidden_states"].shape[1], device, dtype
            )
            noise_pred_neg = self._transformer_call(
                pipe,
                latents,
                text_dict["neg_encoder_hidden_states"],
                text_dict["neg_pooled_projections"],
                neg_txt_ids,
                img_ids,
                timestep,
                guidance,
            )

        return noise_pred_pos, noise_pred_neg.detach()

    def scheduler_step(self, pipe, noise_pred_pos, timestep, latents_dict):
        prev = pipe.scheduler.step(
            noise_pred_pos.detach(), timestep, latents_dict["latents"]
        ).prev_sample
        # Preserve img_ids and any other ancillary tensors
        return {**latents_dict, "latents": prev}


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

_REGISTRY: Dict[str, ModelBackend] = {
    "sd1": SD1Backend(),
    "sd3": SD3Backend(),
    "flux": FluxBackend(),
}


def get_backend(model_type: str) -> ModelBackend:
    """Return the singleton backend for *model_type* (case-insensitive)."""
    key = model_type.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown model_type {model_type!r}.  "
            f"Valid choices: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key]
