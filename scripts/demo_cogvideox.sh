#!/bin/bash
# Demo script for CogVideoX concept erasure using UniErase
#
# Usage:
#   Single GPU (case gen only):  bash scripts/demo_cogvideox.sh cuda:0
#   Multi-GPU (full CAD):        bash scripts/demo_cogvideox.sh "cuda:0,cuda:1,cuda:2,cuda:3"
#
# NOTE: CAD attribution on video models is memory-hungry even on 24GB GPUs.
# With 4x 4090 (24GB each), multi-GPU CAD distributes samples across cards.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

MODEL_TYPE="cogvideox"
MODEL_ID="THUDM/CogVideoX-2b"
DEVICE_INPUT="${1:-cuda:0}"
OUTPUT_DIR="outputs/cogvideox_unierase_demo"

if echo "$DEVICE_INPUT" | grep -q ','; then
    DEVICES_ARG="--cad_devices $DEVICE_INPUT"
    echo "Multi-GPU CAD: $DEVICE_INPUT"
else
    DEVICES_ARG=""
    echo "Single-GPU (case generation only, no CAD)"
fi

echo "========================================"
echo "CogVideoX UniErase Demo"
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
    --cad_steps 20 \
    --cad_num_samples 4 \
    --use_layer_prior \
    --prior_strength 1.0 \
    --cpca_alpha 1.0 \
    --cpca_rank 5 \
    --suppression_strength 1.0 \
    --generate_case \
    --num_frames 13 \
    --seed 42 \
    $DEVICES_ARG

echo ""
echo "Demo complete! Check $OUTPUT_DIR for results."
