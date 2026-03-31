#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_ID="${MODEL_ID:-CompVis/stable-diffusion-v1-4}"
DEVICE="${DEVICE:-cuda:0}"
ARTIFACT="${ARTIFACT:-artifacts/sd14_nudity_cad_ssv.pt}"
PROMPTS_FILE="${PROMPTS_FILE:-prompt_sets/nudity_positive.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_outputs/nudity_guard_eval}"
STEPS="${STEPS:-30}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.5}"
ALPHA="${ALPHA:-}"
SEED="${SEED:-0}"
THRESHOLD="${THRESHOLD:-0.5}"
MAX_PROMPTS="${MAX_PROMPTS:-}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
SAVE_BASE_IMAGES="${SAVE_BASE_IMAGES:-}"   # set to any non-empty value to enable

EXTRA=()
[[ -n "${ALPHA}" ]]           && EXTRA+=(--alpha           "${ALPHA}")
[[ -n "${MAX_PROMPTS}" ]]     && EXTRA+=(--max_prompts     "${MAX_PROMPTS}")
[[ -n "${NEGATIVE_PROMPT}" ]] && EXTRA+=(--negative_prompt "${NEGATIVE_PROMPT}")
[[ -n "${SAVE_BASE_IMAGES}" ]] && EXTRA+=(--save_base_images)

python3 scripts/eval_nudity_guard.py \
  --model_id       "${MODEL_ID}"       \
  --artifact       "${ARTIFACT}"       \
  --prompts_file   "${PROMPTS_FILE}"   \
  --device         "${DEVICE}"         \
  --steps          "${STEPS}"          \
  --guidance_scale "${GUIDANCE_SCALE}" \
  --seed           "${SEED}"           \
  --threshold      "${THRESHOLD}"      \
  --output_dir     "${OUTPUT_DIR}"     \
  "${EXTRA[@]}" "$@"
