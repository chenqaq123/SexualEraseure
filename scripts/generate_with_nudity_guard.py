from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cad_ssv_guard import create_sd_pipeline, load_artifact, register_guard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a validation image with the nudity guard enabled.",
    )
    parser.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--artifact", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument(
        "--image_output",
        "--output",
        dest="image_output",
        type=str,
        default="outputs/guarded_sample.png",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--steer_step_end",
        type=float,
        default=0.7,
        help="Fraction of denoising steps to apply steering (0–1). "
             "Steps beyond this threshold are skipped to preserve fine-detail quality. "
             "Default 0.7 (steer first 70%% of steps, skip last 30%%).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    artifact = load_artifact(args.artifact)
    pipe = create_sd_pipeline(
        model_id=args.model_id,
        device=args.device,
    )
    guard = register_guard(
        pipe=pipe,
        artifact=artifact,
        device=torch.device(args.device),
        alpha=args.alpha,
    )

    output_path = Path(args.image_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Timestep scheduling: disable steering for the last (1 - steer_step_end) fraction
    # of steps so fine-detail passes are not disturbed.
    stop_at_step = max(1, int(args.steer_step_end * args.steps))

    def _step_callback(_pipe, step_index, _timestep, callback_kwargs):
        guard.set_enabled(step_index < stop_at_step)
        return callback_kwargs

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    image = pipe(
        args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        callback_on_step_end=_step_callback,
    ).images[0]
    image.save(output_path)
    guard.remove()

    print(f"Saved guarded image to {output_path}")


if __name__ == "__main__":
    main()
