"""Unified builder script for image and video concept erasure.

Builds a guard artifact using CAD + SSV + layer prior for any supported model:
  - Image models: SD1, SD3, FLUX
  - Video models: CogVideoX, HunyuanVideo

Usage examples:
  # SD1.4 image model
  python scripts/build_unified_guard.py --model_type sd1 \
      --model_id CompVis/stable-diffusion-v1-4

  # CogVideoX video model
  python scripts/build_unified_guard.py --model_type cogvideox \
      --model_id THUDM/CogVideoX-2b

  # HunyuanVideo video model
  python scripts/build_unified_guard.py --model_type hunyuanvideo \
      --model_id tencent/HunyuanVideo
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cad_ssv_guard import build_nudity_guard, create_pipeline, get_backend, save_artifact
from cad_ssv_guard.prompts import load_prompt_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a concept-erasure guard artifact with CAD + SSV + layer prior.",
    )
    # Model selection
    parser.add_argument(
        "--model_type",
        type=str,
        default="sd1",
        choices=["sd1", "sd3", "flux", "cogvideox", "hunyuanvideo"],
        help="Model architecture type",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="HuggingFace model ID or local path (default depends on model_type)",
    )
    parser.add_argument("--device", type=str, default="cuda:0")

    # Prompt configuration
    parser.add_argument(
        "--positive_prompts",
        type=str,
        default="prompt_sets/nudity_positive.txt",
    )
    parser.add_argument(
        "--negative_prompts",
        type=str,
        default="prompt_sets/nudity_negative.txt",
    )

    # Output
    parser.add_argument(
        "--artifact_output",
        "--output",
        dest="artifact_output",
        type=str,
        default=None,
        help="Output artifact path (default: artifacts/{model_type}_nudity_cad_ssv.pt)",
    )

    # Concept configuration
    parser.add_argument("--target", type=str, default="nudity")
    parser.add_argument("--concept_prompt", type=str, default="naked")
    parser.add_argument("--base_prompt", type=str, default="")

    # CAD configuration
    parser.add_argument("--cad_steps", type=int, default=50)
    parser.add_argument("--cad_num_samples", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--cad_candidate_topk", type=int, default=256)
    parser.add_argument("--steering_topk", type=int, default=32)
    parser.add_argument("--total_steering_channels", type=int, default=None)
    parser.add_argument("--min_channels_per_layer", type=int, default=8)

    # Layer prior configuration
    parser.add_argument(
        "--use_layer_prior",
        action="store_true",
        default=True,
        help="Apply architecture-aware Gaussian prior to layer selection",
    )
    parser.add_argument(
        "--no_layer_prior",
        action="store_true",
        default=False,
        help="Disable layer prior (pure data-driven selection)",
    )
    parser.add_argument("--prior_strength", type=float, default=1.0)

    # Steering configuration
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)

    # Video-specific
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)

    return parser.parse_args()


# Default model IDs per type
_DEFAULT_MODEL_IDS = {
    "sd1": "CompVis/stable-diffusion-v1-4",
    "sd3": "stabilityai/stable-diffusion-3-medium-diffusers",
    "flux": "black-forest-labs/FLUX.1-dev",
    "cogvideox": "THUDM/CogVideoX-2b",
    "hunyuanvideo": "tencent/HunyuanVideo",
}


def main() -> None:
    args = parse_args()

    # Resolve defaults
    model_type = args.model_type.lower()
    model_id = args.model_id or _DEFAULT_MODEL_IDS[model_type]
    artifact_output = args.artifact_output or f"artifacts/{model_type}_nudity_cad_ssv.pt"

    positive_prompts = load_prompt_file(args.positive_prompts)
    negative_prompts = load_prompt_file(args.negative_prompts)

    backend = get_backend(model_type)

    pipe = create_pipeline(
        model_type=model_type,
        model_id=model_id,
        device=args.device,
    )

    artifact = build_nudity_guard(
        pipe=pipe,
        backend=backend,
        positive_prompts=positive_prompts,
        negative_prompts=negative_prompts,
        target=args.target,
        concept_prompt=args.concept_prompt,
        base_prompt=args.base_prompt,
        cad_steps=args.cad_steps,
        cad_num_samples=args.cad_num_samples,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_layers=args.num_layers,
        cad_candidate_topk=args.cad_candidate_topk,
        steering_topk=args.steering_topk,
        total_steering_channels=args.total_steering_channels,
        min_channels_per_layer=args.min_channels_per_layer,
        alpha=args.alpha,
        seed=args.seed,
        height=args.height,
        width=args.width,
        use_layer_prior=not args.no_layer_prior,
        prior_strength=args.prior_strength,
    )

    save_artifact(artifact, Path(artifact_output))
    print(f"Saved artifact to {artifact_output}")
    print(f"Model type: {model_type}")
    print(f"Selected layers: {artifact.metadata['selected_ffn_layers']}")
    print(f"Layer prior enabled: {artifact.metadata.get('use_layer_prior', False)}")
    print(f"Prior strength: {artifact.metadata.get('prior_strength', 1.0)}")
    for budget, layer in zip(artifact.metadata["layer_budgets"], artifact.layers):
        print(f"  {layer.ff_name}: budget={budget}, score={layer.layer_score:.6f}, channels={len(layer.selected_channels)}")


if __name__ == "__main__":
    main()
