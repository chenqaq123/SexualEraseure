#!/bin/bash
# 自动监控评估进度并继续运行剩余任务的脚本

source /home/mlsnrs/data/miniconda3/etc/profile.d/conda.sh
conda activate latentGuard

cd /home/mlsnrs/data/cgx/cgx_nas/T2I/Concept_Erasure/SexualErasure_CCS/eval

echo "========================================"
echo "自动评估监控和继续脚本"
echo "启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""

# 定义剩余需要运行的任务
declare -a REMAINING_TASKS=(
    "t2v|T2V_hunyuanvideo/erased_results|T2V HunyuanVideo Erased Results"
    "ti2v|TI2V_hunyuanvideo/results|TI2V HunyuanVideo Results"
)

# 检查指定任务的评估状态
check_task_status() {
    local RESULT_DIR=$1
    if [ -f "$RESULT_DIR/nudenet_eval.csv" ]; then
        return 0  # 已完成
    else
        return 1  # 未完成
    fi
}

# 运行单个评估任务
run_single_task() {
    local BENCHMARK=$1
    local RESULT_DIR=$2
    local DESCRIPTION=$3

    echo "----------------------------------------"
    echo "开始评估: $DESCRIPTION"
    echo "Benchmark: $BENCHMARK"
    echo "结果目录: $RESULT_DIR"
    echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "----------------------------------------"

    # 检查目录是否存在
    if [ ! -d "$RESULT_DIR" ]; then
        echo "❌ 目录不存在，跳过: $RESULT_DIR"
        return 1
    fi

    # 检查是否已有评估报告
    if [ -f "$RESULT_DIR/nudenet_eval.csv" ]; then
        echo "✓ 已存在评估报告，跳过"
        return 0
    fi

    # 运行评估
    python eval_unierase_bench.py \
        --benchmark "$BENCHMARK" \
        --results-dir "$RESULT_DIR" \
        --save-csv "$RESULT_DIR/nudenet_eval.csv" \
        --compute-quality \
        > "$RESULT_DIR/evaluation_log.txt" 2>&1

    # 检查是否成功
    if [ $? -eq 0 ] && [ -f "$RESULT_DIR/nudenet_eval.csv" ]; then
        echo "✓ 评估完成"
        echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
        return 0
    else
        echo "❌ 评估失败，查看日志: $RESULT_DIR/evaluation_log.txt"
        return 1
    fi
}

# 监控当前运行的评估任务
monitor_current_task() {
    local MAX_WAIT=7200  # 最长等待2小时
    local CHECK_INTERVAL=60  # 每60秒检查一次
    local waited=0

    echo "⏳ 监控当前评估任务..."
    echo "最长等待时间: $MAX_WAIT秒"
    echo ""

    while [ $waited -lt $MAX_WAIT ]; do
        # 检查是否还有评估进程在运行
        RUNNING_PROCS=$(ps aux | grep -v grep | grep "eval_unierase_bench.py" | wc -l)

        if [ $RUNNING_PROCS -eq 0 ]; then
            echo "✅ 当前评估任务已完成"
            return 0
        fi

        # 显示进度信息
        local CURRENT_TIME=$(date '+%Y-%m-%d %H:%M:%S')
        echo "[$CURRENT_TIME] 评估仍在运行中... (已等待: ${waited}s)"

        # 检查当前正在运行的任务
        ps aux | grep -v grep | grep "eval_unierase_bench.py" | head -1 | while read line; do
            local PID=$(echo "$line" | awk '{print $2}')
            local CPU=$(echo "$line" | awk '{print $3}')
            local MEM=$(echo "$line" | awk '{print $4}')
            echo "  进程 PID: $PID, CPU: ${CPU}%, MEM: ${MEM}%"
        done

        echo ""
        sleep $CHECK_INTERVAL
        waited=$((waited + CHECK_INTERVAL))
    done

    echo "⚠️  等待超时，继续下一个任务"
    return 1
}

# 主执行流程
main() {
    echo "1️⃣  等待当前评估任务完成..."
    monitor_current_task

    echo ""
    echo "2️⃣  检查并运行剩余任务..."
    echo ""

    for TASK in "${REMAINING_TASKS[@]}"; do
        IFS='|' read -r BENCHMARK RESULT_DIR DESCRIPTION <<< "$TASK"

        if check_task_status "$RESULT_DIR"; then
            echo "✓ 跳过已完成的任务: $DESCRIPTION"
            echo ""
            continue
        fi

        run_single_task "$BENCHMARK" "$RESULT_DIR" "$DESCRIPTION"
        echo ""
    done

    echo "========================================"
    echo "所有任务完成!"
    echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    echo ""

    # 生成最终报告
    echo "📊 最终评估报告:"
    echo ""

    for DIR in TI2I_flux/results TI2I_flux/erased_results T2V_hunyuanvideo/results T2V_hunyuanvideo/erased_results TI2V_hunyuanvideo/results TI2V_hunyuanvideo/erased_results; do
        if [ -f "$DIR/nudenet_eval.csv" ]; then
            CASES=$(tail -n +2 "$DIR/nudenet_eval.csv" | wc -l)
            SIZE=$(du -h "$DIR/nudenet_eval.csv" | awk '{print $1}')
            echo "✅ $DIR: $CASES 案例 (文件大小: $SIZE)"
        else
            echo "❌ $DIR: 评估未完成"
        fi
    done
}

# 运行主流程
main

echo ""
