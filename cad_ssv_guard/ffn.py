from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch.nn as nn


@dataclass(frozen=True)
class FFNModuleSpec:
    ff_name: str
    hidden_module_name: str
    projection_module_name: str


def discover_ffn_specs(unet: nn.Module) -> Dict[str, FFNModuleSpec]:
    hidden_modules = {}
    projection_modules = {}

    for name, module in unet.named_modules():
        if name.endswith("ff.net.0"):
            hidden_modules[name[: -len(".net.0")]] = name
        elif name.endswith("ff.net.2") and isinstance(module, nn.Linear):
            projection_modules[name[: -len(".net.2")]] = name

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


def iter_projection_specs(unet: nn.Module) -> Iterable[Tuple[FFNModuleSpec, nn.Linear]]:
    module_lookup = build_module_lookup(unet)
    for spec in discover_ffn_specs(unet).values():
        projection = module_lookup[spec.projection_module_name]
        if isinstance(projection, nn.Linear):
            yield spec, projection
