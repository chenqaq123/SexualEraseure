from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch.nn as nn


@dataclass(frozen=True)
class FFNModuleSpec:
    ff_name: str
    hidden_module_name: str
    projection_module_name: str


def discover_ffn_specs(backbone: nn.Module) -> Dict[str, FFNModuleSpec]:
    """Discover FFN layer pairs (hidden activation + output projection).

    Supports multiple FFN naming conventions used across architectures:
      - ``ff.net.0`` / ``ff.net.2``  — standard diffusers FeedForward
        (SD1, SD3, FLUX, CogVideoX)
      - ``ff_context.net.0`` / ``ff_context.net.2``  — HunyuanVideo text-stream FFN
      - ``ff_norm.net.0`` / ``ff_norm.net.2``  — variant with built-in norm
    """
    hidden_modules = {}
    projection_modules = {}

    # Patterns for hidden (activation) and projection (output) modules
    hidden_suffixes = (".net.0",)
    projection_suffixes = (".net.2",)
    # FFN stem names to search for
    ff_stems = ("ff", "ff_context", "ff_norm")

    for name, module in backbone.named_modules():
        for stem in ff_stems:
            stem_dot = f"{stem}."
            for suffix in hidden_suffixes:
                full_suffix = f"{stem}{suffix}"
                if name.endswith(full_suffix):
                    ff_name = name[: -len(suffix)]
                    hidden_modules[ff_name] = name
            for suffix in projection_suffixes:
                full_suffix = f"{stem}{suffix}"
                if name.endswith(full_suffix) and isinstance(module, nn.Linear):
                    ff_name = name[: -len(suffix)]
                    projection_modules[ff_name] = name

    specs = {}
    for ff_name, hidden_name in hidden_modules.items():
        projection_name = projection_modules.get(ff_name)
        if projection_name is None:
            continue
        specs[ff_name] = FFNModuleSpec(
            ff_name=ff_name,
            hidden_module_name=hidden_name,
            projection_module_name=projection_name,
        )
    return specs


def build_module_lookup(root: nn.Module) -> Dict[str, nn.Module]:
    return dict(root.named_modules())


def build_parameter_lookup(root: nn.Module) -> Dict[str, nn.Parameter]:
    return dict(root.named_parameters())


def get_projection_weight_name(spec: FFNModuleSpec) -> str:
    return f"{spec.projection_module_name}.weight"


def iter_projection_specs(backbone: nn.Module) -> Iterable[Tuple[FFNModuleSpec, nn.Linear]]:
    module_lookup = build_module_lookup(backbone)
    for spec in discover_ffn_specs(backbone).values():
        projection = module_lookup[spec.projection_module_name]
        if isinstance(projection, nn.Linear):
            yield spec, projection
