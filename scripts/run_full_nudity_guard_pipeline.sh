#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=/home/chenguanxu/common_model/huggingface/hub
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── shared ────────────────────────────────────────────────────────────────────
MODEL_ID="${MODEL_ID:-CompVis/stable-diffusion-v1-4}"
DEVICE="${DEVICE:-cuda:7}"
ARTIFACT="${ARTIFACT:-artifacts/sd14_nudity_cad_ssv.pt}"
STEPS="${STEPS:-30}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.5}"
ALPHA="${ALPHA:-2.0}"
SEED="${SEED:-0}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"

# ── build ─────────────────────────────────────────────────────────────────────
POSITIVE_PROMPTS="${POSITIVE_PROMPTS:-prompt_sets/nudity_positive.txt}"
NEGATIVE_PROMPTS="${NEGATIVE_PROMPTS:-prompt_sets/nudity_negative.txt}"
NUM_LAYERS="${NUM_LAYERS:-3}"
TOTAL_STEERING_CHANNELS="${TOTAL_STEERING_CHANNELS:-96}"
MIN_CHANNELS_PER_LAYER="${MIN_CHANNELS_PER_LAYER:-8}"
CAD_STEPS="${CAD_STEPS:-50}"
CAD_NUM_SAMPLES="${CAD_NUM_SAMPLES:-4}"

# ── eval ──────────────────────────────────────────────────────────────────────
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-eval_outputs/nudity_guard_eval}"
EVAL_MAX_PROMPTS="${EVAL_MAX_PROMPTS:-}"
THRESHOLD="${THRESHOLD:-0.5}"
SAVE_BASE_IMAGES="${SAVE_BASE_IMAGES:-}"   # set to any non-empty value to enable

# ── validation image ──────────────────────────────────────────────────────────
VALIDATION_PROMPT="${VALIDATION_PROMPT:-a woman standing on a beach}"
VALIDATION_IMAGE="${VALIDATION_IMAGE:-outputs/guarded_validation.png}"

# ── helpers ───────────────────────────────────────────────────────────────────
COMMON=(--model_id "${MODEL_ID}" --device "${DEVICE}"
        --guidance_scale "${GUIDANCE_SCALE}" --seed "${SEED}")
NP=(); [[ -n "${NEGATIVE_PROMPT}" ]] && NP=(--negative_prompt "${NEGATIVE_PROMPT}")

# ── 1/3 build ─────────────────────────────────────────────────────────────────
echo "[1/3] Building guard artifact"
python3 scripts/build_nudity_cad_ssv.py \
  "${COMMON[@]}" \
  --positive_prompts        "${POSITIVE_PROMPTS}"        \
  --negative_prompts        "${NEGATIVE_PROMPTS}"        \
  --artifact_output         "${ARTIFACT}"                \
  --num_inference_steps     "${STEPS}"                   \
  --num_layers              "${NUM_LAYERS}"              \
  --total_steering_channels "${TOTAL_STEERING_CHANNELS}" \
  --min_channels_per_layer  "${MIN_CHANNELS_PER_LAYER}"  \
  --cad_steps               "${CAD_STEPS}"               \
  --cad_num_samples         "${CAD_NUM_SAMPLES}"         \
  --alpha                   "${ALPHA}"                   \
  "$@"

# ── 2/3 eval ──────────────────────────────────────────────────────────────────
EVAL_EXTRA=()
[[ -n "${EVAL_MAX_PROMPTS}" ]] && EVAL_EXTRA+=(--max_prompts "${EVAL_MAX_PROMPTS}")
[[ -n "${SAVE_BASE_IMAGES}" ]] && EVAL_EXTRA+=(--save_base_images)

echo "[2/3] Running evaluation"
python3 scripts/eval_nudity_guard.py \
  "${COMMON[@]}" \
  --artifact     "${ARTIFACT}"          \
  --prompts_file "${POSITIVE_PROMPTS}"  \
  --steps        "${STEPS}"             \
  --alpha        "${ALPHA}"             \
  --threshold    "${THRESHOLD}"         \
  --output_dir   "${EVAL_OUTPUT_DIR}"   \
  "${NP[@]}" "${EVAL_EXTRA[@]}"

# ── 3/3 validate ──────────────────────────────────────────────────────────────
echo "[3/3] Generating validation image"
python3 scripts/generate_with_nudity_guard.py \
  "${COMMON[@]}" \
  --artifact     "${ARTIFACT}"          \
  --prompt       "${VALIDATION_PROMPT}" \
  --image_output "${VALIDATION_IMAGE}"  \
  --steps        "${STEPS}"             \
  --alpha        "${ALPHA}"             \
  "${NP[@]}"

echo "Done."
echo "  Artifact : ${ARTIFACT}"
echo "  Eval dir : ${EVAL_OUTPUT_DIR}"
echo "  Image    : ${VALIDATION_IMAGE}"
