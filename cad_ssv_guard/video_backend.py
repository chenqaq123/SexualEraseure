"""Video model backend implementations for CAD gradient attribution.

Backends for:
  - CogVideoX : 3D DiT with full 3D attention (spatial + temporal),
                 single T5-XXL text encoder, standard CFG.
  - HunyuanVideo : Dual-stream DiT (double_blocks → single_blocks),
                   LLAMA text encoder + CLIP vision, flow-matching scheduler.

Both models operate on 3D latent tensors (B, C, F, H, W) where F is the
temporal (frame) dimension.  The CAD attribution procedure is identical to
image models except:
  1. Latent preparation includes the temporal dimension.
  2. Some models require ``num_frames`` and ``fps`` arguments.
  3. For memory efficiency, we use shorter durations (fewer frames) during
     attribution than at full-resolution generation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .backend import ModelBackend


# ─────────────────────────────────────────────────────────────────────────────
# CogVideoX backend
# ─────────────────────────────────────────────────────────────────────────────

class CogVideoXBackend(ModelBackend):
    """CogVideoX — 3D DiT, T5-XXL text encoder, DPM-Multistep scheduler, standard CFG.

    CogVideoX uses a 3D VAE that compresses video (B, C, F, H, W) into
    (B, C', F/t_ratio, H/h_ratio, W/w_ratio).  The transformer operates on
    a flattened sequence of 3D patches.

    Supported model IDs:
      - ``"THUDM/CogVideoX-2b"``
      - ``"THUDM/CogVideoX-5b"``
      - ``"THUDM/CogVideoX-5b-I2V"`` (image-to-video variant)
    """

    model_type = "cogvideox"

    @property
    def supports_cfg(self) -> bool:
        return True

    @property
    def default_height(self) -> int:
        return 480

    @property
    def default_width(self) -> int:
        return 720

    @property
    def default_guidance_scale(self) -> float:
        return 6.0

    @property
    def default_inference_steps(self) -> int:
        return 50

    @property
    def default_num_frames(self) -> int:
        """Default number of output frames (must satisfy (F-1) % temporal_compress == 0)."""
        return 49  # CogVideoX default: ~6 seconds at 8fps

    @property
    def cad_num_frames(self) -> int:
        """Reduced frame count for CAD attribution (saves memory)."""
        return 13  # Minimal valid frame count for CogVideoX

    def get_backbone(self, pipe: Any) -> nn.Module:
        return pipe.transformer

    def encode_prompt_pair(self, pipe, positive_prompt, negative_prompt, device):
        with torch.no_grad():
            # CogVideoX uses a single T5-XXL text encoder
            pos_embeds, neg_embeds = pipe.encode_prompt(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=True,
                device=device,
            )
        # Batch-2: row 0 = positive, row 1 = negative
        return {
            "encoder_hidden_states": torch.cat([pos_embeds, neg_embeds], dim=0),
        }

    def prepare_latents(self, pipe, height, width, device, dtype, generator):
        num_frames = self.cad_num_frames

        # Use the pipeline's own prepare_latents to get correctly shaped latents.
        # CogVideoXPipeline.prepare_latents returns (batch, channels, frames, h, w)
        # with the correct VAE latent channel count (typically 16).
        latents = pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=pipe.vae.config.latent_channels,
            num_frames=num_frames,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        # latents shape: (1, C_vae, F_lat, H_lat, W_lat)
        return {"latents": latents, "num_frames": num_frames}

    def cad_forward(self, pipe, latents_dict, text_dict, timestep):
        latents = latents_dict["latents"]

        # CogVideoX transformer expects (B, C, F, H, W) with C = VAE latent channels (16).
        # Repeat latent for batch-2 [positive_cond, negative_cond]
        latent_input = latents.repeat(2, 1, 1, 1, 1)

        # Expand timestep to batch dimension
        t = timestep.expand(latent_input.shape[0])

        noise_pred = pipe.transformer(
            hidden_states=latent_input,
            encoder_hidden_states=text_dict["encoder_hidden_states"],
            timestep=t,
            return_dict=False,
        )[0]  # (2, C, F, H, W)

        return noise_pred[0:1], noise_pred[1:2].detach()

    def scheduler_step(self, pipe, noise_pred_pos, timestep, latents_dict):
        prev = pipe.scheduler.step(
            noise_pred_pos.detach(), timestep, latents_dict["latents"]
        ).prev_sample
        return {**latents_dict, "latents": prev}


# ─────────────────────────────────────────────────────────────────────────────
# HunyuanVideo backend
# ─────────────────────────────────────────────────────────────────────────────

class HunyuanVideoBackend(ModelBackend):
    """HunyuanVideo — Dual-stream DiT, LLAMA + CLIP encoders, flow-matching.

    HunyuanVideo uses a dual-stream architecture:
      - ``double_blocks``: process image/video tokens and text tokens in parallel
        with cross-attention (similar to FLUX).
      - ``single_blocks``: merge both streams into a single sequence for the
        remaining layers.

    Key differences from image models:
      - Uses LLAMA as the primary text encoder (with CLIP for pooled embeddings).
      - Flow-matching scheduler (similar to SD3/FLUX).
      - No standard CFG by default — uses guidance embedding (like FLUX).
        For CAD, we run two separate forward passes.

    Supported model IDs:
      - ``"tencent/HunyuanVideo"``
    """

    model_type = "hunyuanvideo"

    @property
    def supports_cfg(self) -> bool:
        return False  # Uses guidance embedding, not batched CFG

    @property
    def default_height(self) -> int:
        return 544

    @property
    def default_width(self) -> int:
        return 960

    @property
    def default_guidance_scale(self) -> float:
        return 6.0

    @property
    def default_inference_steps(self) -> int:
        return 50

    @property
    def default_num_frames(self) -> int:
        return 45

    @property
    def cad_num_frames(self) -> int:
        """Reduced frame count for CAD attribution."""
        return 13

    def get_backbone(self, pipe: Any) -> nn.Module:
        return pipe.transformer

    def _make_guidance(
        self,
        pipe: Any,
        batch_size: int,
        guidance_scale: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Return guidance tensor if the model supports guidance embedding."""
        if hasattr(pipe.transformer.config, "guidance_embeds") and pipe.transformer.config.guidance_embeds:
            return torch.full((batch_size,), guidance_scale, device=device, dtype=dtype)
        return None

    def encode_prompt_pair(self, pipe, positive_prompt, negative_prompt, device):
        with torch.no_grad():
            # HunyuanVideo encode_prompt returns (prompt_embeds, pooled_prompt_embeds,
            # prompt_attention_mask) or similar depending on version
            pos_result = pipe.encode_prompt(
                prompt=positive_prompt,
                device=device,
            )
            neg_result = pipe.encode_prompt(
                prompt=negative_prompt,
                device=device,
            )

        # Handle different return formats
        if isinstance(pos_result, tuple):
            pos_embeds = pos_result[0]
            pos_attention_mask = pos_result[2] if len(pos_result) > 2 else None
            pos_pooled = pos_result[1] if len(pos_result) > 1 else None
        else:
            pos_embeds = pos_result
            pos_attention_mask = None
            pos_pooled = None

        if isinstance(neg_result, tuple):
            neg_embeds = neg_result[0]
            neg_attention_mask = neg_result[2] if len(neg_result) > 2 else None
            neg_pooled = neg_result[1] if len(neg_result) > 1 else None
        else:
            neg_embeds = neg_result
            neg_attention_mask = None
            neg_pooled = None

        result = {
            "pos_encoder_hidden_states": pos_embeds,
            "neg_encoder_hidden_states": neg_embeds,
        }
        if pos_pooled is not None:
            result["pos_pooled_projections"] = pos_pooled
            result["neg_pooled_projections"] = neg_pooled
        if pos_attention_mask is not None:
            result["pos_attention_mask"] = pos_attention_mask
            result["neg_attention_mask"] = neg_attention_mask

        return result

    def prepare_latents(self, pipe, height, width, device, dtype, generator):
        num_frames = self.cad_num_frames

        # Use the pipeline's own prepare_latents for correct latent shape.
        latents = pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=pipe.vae.config.latent_channels,
            num_frames=num_frames,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        return {"latents": latents, "num_frames": num_frames}

    def _transformer_call(
        self,
        pipe: Any,
        latents: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        pooled_projections: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        guidance: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kwargs = {
            "hidden_states": latents,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "return_dict": False,
        }
        if pooled_projections is not None:
            kwargs["pooled_projections"] = pooled_projections
        if attention_mask is not None:
            kwargs["encoder_attention_mask"] = attention_mask
        if guidance is not None:
            kwargs["guidance"] = guidance

        return pipe.transformer(**kwargs)[0]

    def cad_forward(self, pipe, latents_dict, text_dict, timestep):
        latents = latents_dict["latents"]
        device, dtype = latents.device, latents.dtype

        guidance = self._make_guidance(
            pipe, latents.shape[0], self.default_guidance_scale, device, dtype
        )

        # ── Positive pass: gradients ON ──────────────────────────────────────
        noise_pred_pos = self._transformer_call(
            pipe,
            latents,
            text_dict["pos_encoder_hidden_states"],
            timestep,
            pooled_projections=text_dict.get("pos_pooled_projections"),
            attention_mask=text_dict.get("pos_attention_mask"),
            guidance=guidance,
        )

        # ── Negative pass: no gradients ──────────────────────────────────────
        with torch.no_grad():
            noise_pred_neg = self._transformer_call(
                pipe,
                latents,
                text_dict["neg_encoder_hidden_states"],
                timestep,
                pooled_projections=text_dict.get("neg_pooled_projections"),
                attention_mask=text_dict.get("neg_attention_mask"),
                guidance=guidance,
            )

        return noise_pred_pos, noise_pred_neg.detach()

    def scheduler_step(self, pipe, noise_pred_pos, timestep, latents_dict):
        prev = pipe.scheduler.step(
            noise_pred_pos.detach(), timestep, latents_dict["latents"]
        ).prev_sample
        return {**latents_dict, "latents": prev}
