"""Layer selection with architecture-aware prior constraints.

Different layers in diffusion models serve different roles:
  - **Early layers** (~0-30% depth): spatial layout, composition, low-level structure
  - **Middle layers** (~30-70% depth): high-level semantic concepts (nudity, objects, styles)
  - **Late layers** (~70-100% depth): fine details, textures, color refinement

Prior knowledge constrains the layer selection to the "semantic encoding zone",
preventing modifications to layout-critical early layers or detail-critical late
layers, which would cause structural distortion or quality degradation.

References:
  - SSV-Guard (KDD'26): causal tracing shows up_blocks.1.0 (layer 7, ~50% depth)
    has strongest causal effect for nudity concept in SD1.4.
  - LRR-V (ICLR'26): refusal vectors applied to layers 17-18 in Open-Sora (~45-47%
    depth of 38 total blocks).
  - CAD (NeurIPS'25): FFN layers in mid-depth have highest attribution scores.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Architecture-specific depth configurations
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SemanticZoneConfig:
    """Configuration for the semantic encoding zone of an architecture.

    Parameters
    ----------
    center : float
        Center of the Gaussian prior (normalized depth in [0, 1]).
    sigma : float
        Standard deviation of the Gaussian (controls zone width).
    hard_min : float
        Hard lower bound — layers shallower than this are excluded.
    hard_max : float
        Hard upper bound — layers deeper than this are excluded.
    """
    center: float = 0.50
    sigma: float = 0.20
    hard_min: float = 0.15
    hard_max: float = 0.85


# Architecture-specific zone configurations
_ZONE_CONFIGS: Dict[str, SemanticZoneConfig] = {
    # UNet: semantic content encoded in mid → early up_blocks
    "sd1": SemanticZoneConfig(center=0.50, sigma=0.20, hard_min=0.20, hard_max=0.80),

    # SD3 MMDiT: semantic zone in middle third of transformer stack
    "sd3": SemanticZoneConfig(center=0.45, sigma=0.20, hard_min=0.15, hard_max=0.80),

    # FLUX DiT: dual→single stream, semantics in mid-range
    "flux": SemanticZoneConfig(center=0.45, sigma=0.22, hard_min=0.15, hard_max=0.80),

    # CogVideoX 3D DiT: similar to image DiT but with temporal attention
    # Semantic concepts reside in middle blocks
    "cogvideox": SemanticZoneConfig(center=0.45, sigma=0.22, hard_min=0.15, hard_max=0.80),

    # HunyuanVideo dual-stream DiT: double_blocks handle semantics,
    # single_blocks handle fine-grained merging
    "hunyuanvideo": SemanticZoneConfig(center=0.42, sigma=0.22, hard_min=0.10, hard_max=0.78),
}


# ─────────────────────────────────────────────────────────────────────────────
# Depth computation
# ─────────────────────────────────────────────────────────────────────────────

def _extract_block_index(layer_name: str) -> Optional[int]:
    """Extract the primary block index from a layer name.

    Handles patterns like:
      - ``transformer_blocks.5.ff.net.0``  → 5
      - ``up_blocks.1.attentions.0.ff.net.2`` → complex UNet mapping
      - ``single_transformer_blocks.3.ff.net.0`` → 3
      - ``double_blocks.7.ff.net.0`` → 7
    """
    # Match the first numeric segment after a block-type keyword
    patterns = [
        r"transformer_blocks\.(\d+)",
        r"single_transformer_blocks\.(\d+)",
        r"double_blocks\.(\d+)",
        r"single_blocks\.(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, layer_name)
        if m:
            return int(m.group(1))
    return None


def compute_layer_depths_unet(
    layer_names: List[str],
) -> Dict[str, float]:
    """Compute normalized depth for UNet-based architectures (SD1, SD2, SDXL).

    UNet depth ordering: down_blocks → mid_block → up_blocks (reversed).
    We linearize this into a [0, 1] depth scale.
    """
    # Categorize layers
    down_layers = []
    mid_layers = []
    up_layers = []
    for name in layer_names:
        if "down_blocks" in name:
            down_layers.append(name)
        elif "mid_block" in name:
            mid_layers.append(name)
        elif "up_blocks" in name:
            up_layers.append(name)
        else:
            mid_layers.append(name)

    # Sort by block index within each category
    def sort_key(n):
        nums = re.findall(r"\.(\d+)", n)
        return tuple(int(x) for x in nums)

    down_layers.sort(key=sort_key)
    mid_layers.sort(key=sort_key)
    up_layers.sort(key=sort_key)

    total = len(down_layers) + len(mid_layers) + len(up_layers)
    if total == 0:
        return {}

    depths = {}
    for i, name in enumerate(down_layers):
        depths[name] = i / total
    offset = len(down_layers)
    for i, name in enumerate(mid_layers):
        depths[name] = (offset + i) / total
    offset += len(mid_layers)
    for i, name in enumerate(up_layers):
        depths[name] = (offset + i) / total

    return depths


def compute_layer_depths_dit(
    layer_names: List[str],
) -> Dict[str, float]:
    """Compute normalized depth for DiT-based architectures (SD3, FLUX, CogVideoX, HunyuanVideo).

    For transformer stacks, depth is simply block_index / total_blocks.
    For dual-stream architectures (HunyuanVideo), double_blocks come before
    single_blocks in the depth ordering.
    """
    # Separate block types
    double_blocks = []
    single_blocks = []
    regular_blocks = []

    for name in layer_names:
        if "double_blocks" in name or "double_transformer_blocks" in name:
            double_blocks.append(name)
        elif "single_blocks" in name or "single_transformer_blocks" in name:
            single_blocks.append(name)
        else:
            regular_blocks.append(name)

    def sort_key(n):
        idx = _extract_block_index(n)
        return idx if idx is not None else 0

    double_blocks.sort(key=sort_key)
    single_blocks.sort(key=sort_key)
    regular_blocks.sort(key=sort_key)

    # Order: regular (or double) blocks first, then single blocks
    ordered = regular_blocks or double_blocks
    ordered = ordered + single_blocks

    total = len(ordered)
    if total == 0:
        return {}

    depths = {}
    for i, name in enumerate(ordered):
        depths[name] = (i + 0.5) / total

    return depths


def compute_layer_depths(
    layer_names: List[str],
    model_type: str,
) -> Dict[str, float]:
    """Compute normalized depth [0, 1] for each layer name.

    Dispatches to architecture-specific depth computation.
    """
    key = model_type.lower()
    if key == "sd1":
        return compute_layer_depths_unet(layer_names)
    else:
        return compute_layer_depths_dit(layer_names)


# ─────────────────────────────────────────────────────────────────────────────
# Prior weight computation
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_prior(depth: float, config: SemanticZoneConfig) -> float:
    """Compute the Gaussian prior weight for a given depth.

    Returns 0 outside [hard_min, hard_max], and a Gaussian-shaped
    weight within the valid range.
    """
    if depth < config.hard_min or depth > config.hard_max:
        return 0.0
    return math.exp(-0.5 * ((depth - config.center) / config.sigma) ** 2)


def compute_layer_prior_weights(
    layer_names: List[str],
    model_type: str,
    config: Optional[SemanticZoneConfig] = None,
) -> Dict[str, float]:
    """Compute prior weights for all layers based on architecture knowledge.

    Parameters
    ----------
    layer_names : list of str
        Names of FFN layers (e.g., ``"up_blocks.1.attentions.0.ff"``).
    model_type : str
        Architecture identifier (``"sd1"``, ``"sd3"``, ``"flux"``,
        ``"cogvideox"``, ``"hunyuanvideo"``).
    config : SemanticZoneConfig, optional
        Override the default zone configuration for the architecture.

    Returns
    -------
    Dict mapping layer names to prior weights in [0, 1].
    """
    if config is None:
        config = _ZONE_CONFIGS.get(
            model_type.lower(),
            SemanticZoneConfig(),  # fallback: uniform prior centered at 0.5
        )

    depths = compute_layer_depths(layer_names, model_type)
    weights = {}
    for name in layer_names:
        depth = depths.get(name, 0.5)
        weights[name] = gaussian_prior(depth, config)

    return weights


def apply_prior_to_scores(
    layer_scores: Dict[str, float],
    prior_weights: Dict[str, float],
    prior_strength: float = 1.0,
) -> Dict[str, float]:
    """Combine CAD/SSV layer scores with architecture prior weights.

    Parameters
    ----------
    layer_scores : dict
        Mapping of layer names to their CAD/SSV scores.
    prior_weights : dict
        Mapping of layer names to their prior weights (from
        :func:`compute_layer_prior_weights`).
    prior_strength : float
        Controls how strongly the prior influences selection:
        0.0 = ignore prior (pure data-driven), 1.0 = full prior weighting.

    Returns
    -------
    Dict with adjusted scores: ``score * (1 - α + α * prior_weight)``.
    """
    adjusted = {}
    for name, score in layer_scores.items():
        pw = prior_weights.get(name, 0.5)
        # Smooth blending: at prior_strength=0, multiplier=1 (no effect);
        # at prior_strength=1, multiplier=prior_weight
        multiplier = 1.0 - prior_strength + prior_strength * pw
        adjusted[name] = score * multiplier
    return adjusted
