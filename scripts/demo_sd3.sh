#!/bin/bash
# Demo script for SD3 Medium concept erasure using UniErase
#
# Usage:
#   Single GPU:    bash scripts/demo_sd3.sh cuda:0
#   Multi-GPU:     bash scripts/demo_sd3.sh "cuda:0,cuda:1,cuda:2,cuda:3"

export HF_HOME="/home/mlsnrs/common_model/huggingface"

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

MODEL_TYPE="sd3"
MODEL_ID="stabilityai/stable-diffusion-3-medium-diffusers"
DEVICE_INPUT="${1:-cuda:0}"
OUTPUT_DIR="outputs/sd3_unierase_demo_rankd_pca_suppression_ratio_2.0"

# Parse device string
if echo "$DEVICE_INPUT" | grep -q ','; then
    DEVICES_ARG="--cad_devices $DEVICE_INPUT"
    echo "Multi-GPU CAD: $DEVICE_INPUT"
else
    DEVICES_ARG=""
    echo "Single-GPU CAD: $DEVICE_INPUT"
fi

echo "========================================"
echo "SD3 Medium UniErase Demo"
echo "========================================"
echo "Model: $MODEL_ID"
echo "Device(s): $DEVICE_INPUT"
echo "Output: $OUTPUT_DIR"
echo "========================================"

python scripts/run_unierase.py \
    --model_type "$MODEL_TYPE" \
    --model_id "$MODEL_ID" \
    --device "${DEVICE_INPUT%%,*}" \
    --output_dir "$OUTPUT_DIR" \
    --num_layers 3 \
    --cad_steps 28 \
    --cad_num_samples 4 \
    --use_layer_prior \
    --prior_strength 1.0 \
    --skip_cpca \
    --ssv_num_seeds 4 \
    --pca_max_rank 8 \
    --pca_variance_threshold 0.80 \
    --suppression_strength 2.0 \
    --aggressive_mode \
    --generate_case \
    --seed 42 \
    $DEVICES_ARG

echo ""
echo "Demo complete! Check $OUTPUT_DIR for results."
