from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List

import torch


@dataclass
class GuardLayer:
    ff_name: str
    hidden_module_name: str
    projection_module_name: str
    selected_channels: List[int]
    cad_candidate_channels: List[int]
    steering_vector: List[float]
    layer_score: float
    positive_hook_calls: int
    negative_hook_calls: int


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
    data = torch.load(path, map_location="cpu")
    if "layers" not in data:
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
        data["layers"] = [GuardLayer(**layer) for layer in data["layers"]]
    return GuardArtifact(**data)
