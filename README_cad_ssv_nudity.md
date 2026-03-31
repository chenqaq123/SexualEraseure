# CAD + SSV Nudity Guard

This prototype combines:

- CAD-style localization on Stable Diffusion v1.4 UNet FFN projection weights
- SSV-style sparse activation steering at inference time

The current implementation only targets nudity suppression and only supports SD v1.4 in the intended workflow.

## What It Does

1. Compute CAD attribution scores on FFN output projection weights using a nudity-vs-base objective.
2. Rank FFN layers by CAD score and keep the top few layers instead of only one.
3. Collect mean FFN hidden activations on nudity prompts and benign prompts for each selected layer.
4. Build a sparse steering vector from channels that are strong under both CAD and SSV scoring.
5. Apply forward hooks during sampling to suppress aligned activations across multiple layers.

Compared with the earlier prototype, the CAD stage now:

- averages over multiple nudity/benign prompt pairs instead of a single `"naked"` prompt
- uses normalized channel scores `raw_score / mean_abs_weight_column`
- ranks layers with the mean of top-k normalized channel scores instead of raw layer sums
- allocates the total steering budget across layers in proportion to normalized CAD layer scores

## Build The Guard

```bash
python3 scripts/build_nudity_cad_ssv.py \
  --model_id CompVis/stable-diffusion-v1-4 \
  --device cuda:0 \
  --num_layers 3 \
  --total_steering_channels 96 \
  --artifact_output artifacts/sd14_nudity_cad_ssv.pt
```

Here `artifact_output` is the serialized guard artifact, not a generated image.

## Eval With Detection

```bash
python3 scripts/eval_nudity_guard.py \
  --model_id CompVis/stable-diffusion-v1-4 \
  --artifact artifacts/sd14_nudity_cad_ssv.pt \
  --prompts_file prompt_sets/nudity_positive.txt \
  --output_dir eval_outputs/nudity_guard_eval
```

This script generates both base and guarded images, runs NudeNet detection, and writes a `results.json`.

Shell wrapper:

```bash
bash scripts/run_eval_nudity_guard.sh
```

## Generate For Visual Validation

```bash
python3 scripts/generate_with_nudity_guard.py \
  --model_id CompVis/stable-diffusion-v1-4 \
  --artifact artifacts/sd14_nudity_cad_ssv.pt \
  --prompt "a woman standing on a beach" \
  --image_output outputs/guarded.png
```

Here `image_output` is just the generated validation image.

Shell wrapper:

```bash
bash scripts/run_generate_nudity_guard.sh
```

You can override defaults with environment variables, for example:

```bash
ARTIFACT=artifacts/sd14_nudity_cad_ssv.pt \
PROMPT="a portrait photo of a person in formal clothing" \
IMAGE_OUTPUT=outputs/formal_portrait.png \
bash scripts/run_generate_nudity_guard.sh
```

## Full Pipeline

Shell wrapper:

```bash
bash scripts/run_full_nudity_guard_pipeline.sh
```

This runs:

1. guard artifact building
2. detection-based evaluation
3. one validation image generation

You can override the main settings with environment variables, for example:

```bash
ARTIFACT_OUTPUT=artifacts/sd14_exp1.pt \
EVAL_OUTPUT_DIR=eval_outputs/exp1 \
VALIDATION_PROMPT="a realistic portrait of a person wearing a coat" \
VALIDATION_IMAGE_OUTPUT=outputs/exp1_validation.png \
bash scripts/run_full_nudity_guard_pipeline.sh
```

## Notes

- The CAD stage is constrained to FFN projection weights because they map cleanly to hidden channels.
- The runtime steering hooks act on the paired FFN hidden modules (`ff.net.0`) and only modify sparse channel subsets.
- This is a practical SD1.4 research prototype rather than a fully benchmarked reproduction of either paper.
