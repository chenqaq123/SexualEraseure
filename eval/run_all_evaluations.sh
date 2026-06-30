#!/bin/bash
# 批量运行UniErase-Bench评估脚本
# 为所有缺少评估报告的目录生成NudeNet评估结果

set -e  # 遇到错误立即退出

# 激活latentGuard conda环境
source /home/mlsnrs/data/miniconda3/etc/profile.d/conda.sh
conda activate latentGuard

cd /home/mlsnrs/data/cgx/cgx_nas/T2I/Concept_Erasure/SexualErasure_CCS/eval

echo "========================================"
echo "开始批量UniErase-Bench评估"
echo "========================================"
echo ""

# 定义评估任务数组
# 格式: "benchmark_type|results_dir|description"
declare -a TASKS=(
    "ti2i|TI2I_flux/erased_results|TI2I Flux Erased Results"
    "t2v|T2V_hunyuanvideo/results|T2V HunyuanVideo Results"
    "t2v|T2V_hunyuanvideo/erased_results|T2V HunyuanVideo Erased Results"
    "ti2v|TI2V_hunyuanvideo/results|TI2V HunyuanVideo Results"
    "ti2v|TI2V_hunyuanvideo/erased_results|TI2V HunyuanVideo Erased Results"
)

# 记录开始时间
START_TIME=$(date +%s)

# 遍历所有任务
for TASK in "${TASKS[@]}"; do
    IFS='|' read -r BENCHMARK RESULT_DIR DESCRIPTION <<< "$TASK"

    echo "----------------------------------------"
    echo "评估任务: $DESCRIPTION"
    echo "Benchmark: $BENCHMARK"
    echo "结果目录: $RESULT_DIR"
    echo "----------------------------------------"

    # 检查目录是否存在
    if [ ! -d "$RESULT_DIR" ]; then
        echo "❌ 目录不存在，跳过: $RESULT_DIR"
        echo ""
        continue
    fi

    # 检查是否已有评估报告
    if [ -f "$RESULT_DIR/nudenet_eval.csv" ] || [ -f "$RESULT_DIR/nudenet_eval_full.csv" ]; then
        echo "✓ 已存在评估报告，跳过"
        echo ""
        continue
    fi

    # 运行评估脚本
    echo "运行评估..."

    # 根据benchmark类型设置不同的参数
    EXTRA_ARGS=""
    if [[ "$BENCHMARK" == "ti2i" ]]; then
        # TI2I: 计算质量指标
        EXTRA_ARGS="--compute-quality"
    elif [[ "$BENCHMARK" == "t2v" ]] || [[ "$BENCHMARK" == "ti2v" ]]; then
        # T2V/TI2V: 视频评估，可选择计算质量指标
        EXTRA_ARGS="--compute-quality"
    fi

    # 运行评估
    python eval_unierase_bench.py \
        --benchmark "$BENCHMARK" \
        --results-dir "$RESULT_DIR" \
        --save-csv "$RESULT_DIR/nudenet_eval.csv" \
        $EXTRA_ARGS \
        > "$RESULT_DIR/evaluation_log.txt" 2>&1

    # 检查是否成功
    if [ $? -eq 0 ]; then
        echo "✓ 评估完成"
        echo "结果保存到: $RESULT_DIR/nudenet_eval.csv"
    else
        echo "❌ 评估失败，查看日志: $RESULT_DIR/evaluation_log.txt"
    fi

    echo ""
done

# 计算总耗时
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo "========================================"
echo "批量评估完成!"
echo "总耗时: ${MINUTES}分${SECONDS}秒"
echo "========================================"
echo ""
echo "评估结果摘要:"
echo ""

for TASK in "${TASKS[@]}"; do
    IFS='|' read -r BENCHMARK RESULT_DIR DESCRIPTION <<< "$TASK"
    if [ -f "$RESULT_DIR/nudenet_eval.csv" ]; then
        echo "✓ $DESCRIPTION - 完成"
    else
        echo "✗ $DESCRIPTION - 失败或跳过"
    fi
done

echo ""
