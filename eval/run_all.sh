#!/bin/bash
# Run HunyuanVideo 1.5 eval scripts in parallel
# GPU allocation:
#   T2V_hunyuanvideo  (HunyuanVideo-1.5-480p_t2v) -> GPU 0,1,2,3
#   TI2V_hunyuanvideo (HunyuanVideo-1.5-480p_i2v) -> GPU 4,5,6,7
#
# Usage: bash run_all.sh

# Note: no set -e, to prevent early exit when a process fails

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Python from conda environment
PYTHON="/home/mlsnrs/data/miniconda3/envs/latentGuard/bin/python"

# HF mirror for model loading
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/home/mlsnrs/common_model/huggingface"
export HF_HUB_CACHE="/home/mlsnrs/common_model/huggingface/hub"

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "Starting parallel eval (2 processes)"
echo "Logs: ${LOG_DIR}"
echo "=========================================="

# T2V_hunyuanvideo -> GPU 0,1,2,3
CUDA_VISIBLE_DEVICES=0,1,2,3 ${PYTHON} "${SCRIPT_DIR}/T2V_hunyuanvideo/test.py" > "${LOG_DIR}/T2V_hunyuanvideo.log" 2>&1 &
PID1=$!
echo "[PID ${PID1}] T2V_hunyuanvideo  (GPU 0,1,2,3) started"

# TI2V_hunyuanvideo -> GPU 4,5,6,7
CUDA_VISIBLE_DEVICES=4,5,6,7 ${PYTHON} "${SCRIPT_DIR}/TI2V_hunyuanvideo/test.py" > "${LOG_DIR}/TI2V_hunyuanvideo.log" 2>&1 &
PID2=$!
echo "[PID ${PID2}] TI2V_hunyuanvideo (GPU 4,5,6,7) started"

echo ""
echo "Waiting for all processes to finish..."

FAIL=0
wait $PID1; S1=$?; [ $S1 -ne 0 ] && echo "[FAIL] T2V_hunyuanvideo  (exit=$S1)" && FAIL=$((FAIL+1)) || echo "[DONE] T2V_hunyuanvideo"
wait $PID2; S2=$?; [ $S2 -ne 0 ] && echo "[FAIL] TI2V_hunyuanvideo (exit=$S2)" && FAIL=$((FAIL+1)) || echo "[DONE] TI2V_hunyuanvideo"

echo ""
echo "=========================================="
if [ $FAIL -eq 0 ]; then
    echo "All processes completed successfully!"
else
    echo "${FAIL} process(es) failed. Check logs in ${LOG_DIR}"
fi
echo "=========================================="
