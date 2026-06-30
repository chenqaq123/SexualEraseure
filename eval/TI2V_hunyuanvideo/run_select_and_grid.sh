#!/bin/bash
# TI2V pipeline: select channels then grid search
# Step 1: select_channels.py -> channel_data_ti2v.json
# Step 2: gridsearch.py (reads channel_data_ti2v.json)
#
# Usage: bash run_select_and_grid.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

PYTHON="/home/chenguanxu/miniconda3/envs/latentGuard/bin/python"

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/home/chenguanxu/common_model/huggingface"
export HF_HUB_CACHE="/home/chenguanxu/common_model/huggingface/hub"

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# ============================================================
# Step 1: Select channels
# ============================================================
echo "=========================================="
echo "[Step 1] Running select_channels.py ..."
echo "=========================================="

CUDA_VISIBLE_DEVICES=4,5,6,7 ${PYTHON} "${SCRIPT_DIR}/select_channels.py" 2>&1 | tee "${LOG_DIR}/select_channels.log"

if [ ! -f "${SCRIPT_DIR}/channel_data_ti2v.json" ]; then
    echo "[ERROR] channel_data_ti2v.json not found after select_channels. Aborting."
    exit 1
fi

echo ""
echo "[Step 1] Done. channel_data_ti2v.json generated."

# ============================================================
# Step 2: Grid search
# ============================================================
echo "=========================================="
echo "[Step 2] Running gridsearch.py ..."
echo "=========================================="

CUDA_VISIBLE_DEVICES=4,5,6,7 ${PYTHON} "${SCRIPT_DIR}/gridsearch.py" 2>&1 | tee "${LOG_DIR}/gridsearch.log"

echo ""
echo "=========================================="
echo "All done!"
echo "=========================================="
