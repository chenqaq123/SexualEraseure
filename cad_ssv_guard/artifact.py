from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


@dataclass
class GuardLayer:
    # ── Module identity ───────────────────────────────────────────────────────
    ff_name: str
    hidden_module_name: str
    projection_module_name: str
    # ── Channel selection ─────────────────────────────────────────────────────
    selected_channels: List[int]
    cad_candidate_channels: List[int]
    # ── Concept direction(s) ──────────────────────────────────────────────────
    # steering_vector: first (primary) PCA direction, shape (k,).
    # Kept for backward compatibility with older artifacts and hook-based steering.
    steering_vector: List[float]
    # concept_directions: all PCA directions, shape (d, k).
    # d=1 means rank-1 (same as steering_vector); d>1 means rank-d edit.
    # None for artifacts built before PCA support was added.
    concept_directions: Optional[List[List[float]]] = None
    # Explained variance ratio for each retained PCA direction.
    pca_explained_variance_ratios: Optional[List[float]] = None
    # ── Layer scoring / ranking ───────────────────────────────────────────────
    layer_score: float = 0.0           # adjusted score used for ranking (= cad × prior)
    cad_raw_score: float = 0.0         # raw CAD score before prior weighting
    layer_rank: int = 0                # 1-indexed rank across all candidate layers
    # ── Channel selection diagnostics ────────────────────────────────────────
    ssv_scores_selected: Optional[List[float]] = None   # SSV score per selected channel
    # ── PCA diagnostics ───────────────────────────────────────────────────────
    pca_n_samples: int = 0             # rows in the concept-diff matrix fed to PCA
    # ── Hook call counts ──────────────────────────────────────────────────────
    positive_hook_calls: int = 0
    negative_hook_calls: int = 0


@dataclass
class GuardArtifact:
    model_id: str
    target: str
    base_prompt: str
    layers: List[GuardLayer]
    alpha: float
    num_inference_steps: int
    guidance_scale: float
    cad_steps: int
    cad_num_samples: int
    positive_prompts: List[str]
    negative_prompts: List[str]
    metadata: Dict[str, Any]
    # Full layer ranking table (all candidates, including non-selected layers).
    # Each entry: {rank, ff_name, cad_score_raw, cad_score_adjusted,
    #              prior_weight, selected, channel_budget}
    layer_ranking: Optional[List[Dict[str, Any]]] = None


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_artifact(artifact: GuardArtifact, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(asdict(artifact), output_path)

    json_path = output_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(asdict(artifact), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_artifact(path: str | Path) -> GuardArtifact:
    data = torch.load(path, map_location="cpu", weights_only=False)

    if "layers" not in data:
        # ── Legacy single-layer format ────────────────────────────────────────
        legacy_layer = GuardLayer(
            ff_name=data.get("metadata", {}).get("selected_ffn", "legacy_ff"),
            hidden_module_name=data["hidden_module_name"],
            projection_module_name=data["projection_module_name"],
            selected_channels=data["selected_channels"],
            cad_candidate_channels=data["cad_candidate_channels"],
            steering_vector=data["steering_vector"],
            layer_score=data["layer_score"],
            positive_hook_calls=data.get("metadata", {}).get("positive_hook_calls", 0),
            negative_hook_calls=data.get("metadata", {}).get("negative_hook_calls", 0),
        )
        data = {
            "model_id": data["model_id"],
            "target": data["target"],
            "base_prompt": data["base_prompt"],
            "layers": [legacy_layer],
            "alpha": data["alpha"],
            "num_inference_steps": data["num_inference_steps"],
            "guidance_scale": data["guidance_scale"],
            "cad_steps": data["cad_steps"],
            "cad_num_samples": data["cad_num_samples"],
            "positive_prompts": data["positive_prompts"],
            "negative_prompts": data["negative_prompts"],
            "metadata": {
                **data.get("metadata", {}),
                "selected_ffn_layers": [legacy_layer.ff_name],
                "num_layers": 1,
                "layer_budgets": [len(legacy_layer.selected_channels)],
            },
        }
    else:
        # ── Current multi-layer format ────────────────────────────────────────
        # GuardLayer may have been saved without the optional PCA fields;
        # fill missing keys with None defaults before constructing.
        layers = []
        for layer_dict in data["layers"]:
            layer_dict.setdefault("concept_directions", None)
            layer_dict.setdefault("pca_explained_variance_ratios", None)
            layer_dict.setdefault("cad_raw_score", layer_dict.get("layer_score", 0.0))
            layer_dict.setdefault("layer_rank", 0)
            layer_dict.setdefault("ssv_scores_selected", None)
            layer_dict.setdefault("pca_n_samples", 0)
            layers.append(GuardLayer(**layer_dict))
        data["layers"] = layers
        data.setdefault("layer_ranking", None)

    return GuardArtifact(**data)
