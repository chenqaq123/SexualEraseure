#!/bin/bash
# Download HunyuanVideo 1.5 models via hf-mirror
# Usage: bash download_models.sh

# Note: no set -e here, to prevent early exit when a download fails

# ============================================================
# Set your HuggingFace token in the environment before running:
#   export HF_TOKEN="hf_xxx"
# ============================================================
export HF_TOKEN="${HF_TOKEN:?Please export HF_TOKEN before running this script}"

# Use hf-mirror for faster download in China
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/home/mlsnrs/common_model/huggingface"
export HF_HUB_CACHE="/home/mlsnrs/common_model/huggingface/hub"

CACHE_DIR="/home/mlsnrs/common_model/huggingface/hub"

MODELS=(
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v"
)

echo "=========================================="
echo "Downloading models via hf-mirror"
echo "HF_ENDPOINT: ${HF_ENDPOINT}"
echo "Cache dir:   ${CACHE_DIR}"
echo "=========================================="

# Disable symlinks to avoid OSError on certain filesystems
export HF_HUB_ENABLE_HF_TRANSFER=0

PIDS=()
for MODEL_ID in "${MODELS[@]}"; do
    echo ">>> Downloading: ${MODEL_ID}"
    huggingface-cli download \
        "${MODEL_ID}" \
        --cache-dir "${CACHE_DIR}" \
        --local-dir-use-symlinks False \
        --token "${HF_TOKEN}" \
        > "/tmp/dl_${MODEL_ID//\//_}.log" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "Waiting for all downloads to finish (PIDs: ${PIDS[*]})..."

FAIL=0
for i in "${!PIDS[@]}"; do
    wait ${PIDS[$i]}
    STATUS=$?
    if [ $STATUS -ne 0 ]; then
        echo "[FAIL] ${MODELS[$i]} (exit=$STATUS)"
        FAIL=$((FAIL+1))
    else
        echo "[DONE] ${MODELS[$i]}"
    fi
done

echo ""
echo "=========================================="
if [ $FAIL -eq 0 ]; then
    echo "All models downloaded successfully!"
else
    echo "${FAIL} model(s) failed. Check logs in /tmp/dl_*.log"
fi
echo "=========================================="
