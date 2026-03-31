from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cad_ssv_guard import build_nudity_guard, create_sd_pipeline, save_artifact
from cad_ssv_guard.prompts import load_prompt_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a nudity guard artifact with CAD localization and SSV-style steering.",
    )
    parser.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--device", type=str, default="cuda:0")
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
    parser.add_argument(
        "--artifact_output",
        "--output",
        dest="artifact_output",
        type=str,
        default="artifacts/sd14_nudity_cad_ssv.pt",
    )
    parser.add_argument("--concept_prompt", type=str, default="naked")
    parser.add_argument("--base_prompt", type=str, default="")
    parser.add_argument("--cad_steps", type=int, default=50)
    parser.add_argument("--cad_num_samples", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--cad_candidate_topk", type=int, default=256)
    parser.add_argument("--steering_topk", type=int, default=32)
    parser.add_argument("--total_steering_channels", type=int, default=None)
    parser.add_argument("--min_channels_per_layer", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    positive_prompts = load_prompt_file(args.positive_prompts)
    negative_prompts = load_prompt_file(args.negative_prompts)

    pipe = create_sd_pipeline(
        model_id=args.model_id,
        device=args.device,
    )

    artifact = build_nudity_guard(
        pipe=pipe,
        positive_prompts=positive_prompts,
        negative_prompts=negative_prompts,
        target="nudity",
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
    )

    save_artifact(artifact, Path(args.artifact_output))
    print(f"Saved artifact to {args.artifact_output}")
    print(f"Selected layers: {artifact.metadata['selected_ffn_layers']}")
    for budget, layer in zip(artifact.metadata["layer_budgets"], artifact.layers):
        print(f"{layer.ff_name}: budget={budget}, score={layer.layer_score:.6f}, channels={layer.selected_channels}")


if __name__ == "__main__":
    main()
