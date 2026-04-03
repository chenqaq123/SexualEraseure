from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from .artifact import GuardArtifact, GuardLayer


@dataclass
class SparseSteeringController:
    module_name: str
    channel_indices: torch.Tensor
    steering_vector: torch.Tensor
    alpha: float = 1.0
    enabled: bool = True

    @classmethod
    def from_layer(
        cls,
        layer: GuardLayer,
        device: Optional[torch.device] = None,
        alpha: Optional[float] = None,
        default_alpha: float = 1.0,
    ) -> "SparseSteeringController":
        vector = torch.tensor(layer.steering_vector, dtype=torch.float32, device=device)
        indices = torch.tensor(layer.selected_channels, dtype=torch.long, device=device)
        return cls(
            module_name=layer.hidden_module_name,
            channel_indices=indices,
            steering_vector=vector,
            alpha=default_alpha if alpha is None else alpha,
        )

    def _get_sv_hat(self, dtype: torch.dtype) -> torch.Tensor:
        """Return the unit-norm steering vector (cached computation)."""
        sv = self.steering_vector.to(dtype)
        norm = sv.norm().clamp_min(1e-8)
        return sv / norm

    def _project_and_steer(self, selected: torch.Tensor) -> torch.Tensor:
        """
        Remove the nudity component from `selected` via orthogonal projection.

        For each position, compute the scalar projection onto the steering direction,
        clamp to non-negative (only suppress nudity-aligned activations),
        then subtract alpha * projection * unit_vector.

        This is correct concept erasure: delta ∝ (act · sv̂) * sv̂,
        so the magnitude of the edit scales with the actual nudity signal strength,
        not the tiny raw steering vector values.
        """
        sv_hat = self._get_sv_hat(selected.dtype)  # (n_channels,)
        # Projection: scalar per spatial/sequence position
        proj = (selected * sv_hat).sum(dim=-1, keepdim=True)  # (..., 1)
        proj = proj.clamp_min(0.0)  # only suppress positive nudity component
        delta = self.alpha * proj * sv_hat  # (..., n_channels)
        return selected - delta

    def _steer_last_dim(self, output: torch.Tensor) -> torch.Tensor:
        steered = output.clone()
        selected = steered[..., self.channel_indices]
        steered[..., self.channel_indices] = self._project_and_steer(selected).to(output.dtype)
        return steered

    def _steer_channel_dim(self, output: torch.Tensor) -> torch.Tensor:
        # 4D: (batch, channels, h, w) → permute to last-dim, steer, permute back
        steered = output.clone()
        # Work on selected channels only; permute to (batch, h, w, n_channels)
        selected = steered[:, self.channel_indices, :, :].permute(0, 2, 3, 1)
        sv_hat = self._get_sv_hat(selected.dtype)
        proj = (selected * sv_hat).sum(dim=-1, keepdim=True).clamp_min(0.0)
        delta = self.alpha * proj * sv_hat
        steered[:, self.channel_indices, :, :] = (selected - delta).permute(0, 3, 1, 2).to(output.dtype)
        return steered

    def steer(self, output: torch.Tensor) -> torch.Tensor:
        if output.ndim in (2, 3):
            return self._steer_last_dim(output)
        if output.ndim == 4:
            return self._steer_channel_dim(output)
        raise ValueError(f"Unsupported activation shape: {tuple(output.shape)}")

    def register(self, module):
        def hook(_module, _inputs, output):
            if not self.enabled:
                return output
            hidden = output[0] if isinstance(output, tuple) else output

            # CFG convention in diffusers: batch = [uncond, cond] when size == 2.
            # Only steer the conditional pass; the unconditional baseline must stay
            # intact so the CFG guidance direction is preserved.
            if hidden.shape[0] == 2:
                steered = hidden.clone()
                steered[1:2] = self.steer(hidden[1:2])
            else:
                steered = self.steer(hidden)

            if isinstance(output, tuple):
                return (steered, *output[1:])
            return steered

        return module.register_forward_hook(hook)


def _get_backbone(pipe):
    """Return the denoising backbone: `pipe.transformer` for DiT-based models
    (SD3, FLUX, etc.) or `pipe.unet` for UNet-based models (SD1/SD2/SDXL)."""
    if hasattr(pipe, "transformer"):
        return pipe.transformer
    return pipe.unet


def attach_controller(pipe, controller: SparseSteeringController):
    module_lookup = dict(_get_backbone(pipe).named_modules())
    module = module_lookup[controller.module_name]
    return controller.register(module)


@dataclass
class MultiLayerSparseSteeringController:
    controllers: List[SparseSteeringController]
    _handles: List = None  # hook handles, populated by register()

    @classmethod
    def from_artifact(
        cls,
        artifact: GuardArtifact,
        device: Optional[torch.device] = None,
        alpha: Optional[float] = None,
    ) -> "MultiLayerSparseSteeringController":
        controllers = [
            SparseSteeringController.from_layer(
                layer=layer,
                device=device,
                alpha=alpha,
                default_alpha=artifact.alpha,
            )
            for layer in artifact.layers
        ]
        return cls(controllers=controllers)

    def register(self, pipe) -> "MultiLayerSparseSteeringController":
        self._handles = [attach_controller(pipe, ctrl) for ctrl in self.controllers]
        return self

    def set_enabled(self, enabled: bool) -> None:
        """Toggle steering on/off for all layers (used for timestep scheduling)."""
        for ctrl in self.controllers:
            ctrl.enabled = enabled

    def remove(self) -> None:
        """Remove all forward hooks from the model."""
        if self._handles:
            for h in self._handles:
                h.remove()
            self._handles = []
