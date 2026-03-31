#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_ID="${MODEL_ID:-CompVis/stable-diffusion-v1-4}"
DEVICE="${DEVICE:-cuda:0}"
ARTIFACT="${ARTIFACT:-artifacts/sd14_nudity_cad_ssv.pt}"
PROMPT="${PROMPT:-a woman standing on a beach}"
IMAGE_OUTPUT="${IMAGE_OUTPUT:-outputs/guarded_sample.png}"
STEPS="${STEPS:-30}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.5}"
ALPHA="${ALPHA:-}"
SEED="${SEED:-0}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"

EXTRA=()
[[ -n "${ALPHA}" ]]           && EXTRA+=(--alpha           "${ALPHA}")
[[ -n "${NEGATIVE_PROMPT}" ]] && EXTRA+=(--negative_prompt "${NEGATIVE_PROMPT}")

python3 scripts/generate_with_nudity_guard.py \
  --model_id       "${MODEL_ID}"       \
  --artifact       "${ARTIFACT}"       \
  --prompt         "${PROMPT}"         \
  --image_output   "${IMAGE_OUTPUT}"   \
  --device         "${DEVICE}"         \
  --steps          "${STEPS}"          \
  --guidance_scale "${GUIDANCE_SCALE}" \
  --seed           "${SEED}"           \
  "${EXTRA[@]}" "$@"
