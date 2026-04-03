#!/bin/bash
# Demo script for SD1.4 concept erasure using UniErase
#
# Usage:
#   Single GPU:    bash scripts/demo_sd14.sh cuda:0
#   Multi-GPU:     bash scripts/demo_sd14.sh "cuda:0,cuda:1,cuda:2,cuda:3"

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

MODEL_TYPE="sd1"
MODEL_ID="CompVis/stable-diffusion-v1-4"
DEVICE_INPUT="${1:-cuda:0}"
OUTPUT_DIR="outputs/sd14_unierase_demo"

# Parse device string
if echo "$DEVICE_INPUT" | grep -q ','; then
    DEVICES_ARG="--cad_devices $DEVICE_INPUT"
    echo "Multi-GPU CAD: $DEVICE_INPUT"
else
    DEVICES_ARG=""
    echo "Single-GPU CAD: $DEVICE_INPUT"
fi

echo "========================================"
echo "SD1.4 UniErase Demo"
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
    --cad_steps 50 \
    --cad_num_samples 4 \
    --use_layer_prior \
    --prior_strength 1.0 \
    --cpca_alpha 1.0 \
    --cpca_rank 5 \
    --suppression_strength 1.0 \
    --generate_case \
    --seed 42 \
    $DEVICES_ARG

echo ""
echo "Demo complete! Check $OUTPUT_DIR for results."
