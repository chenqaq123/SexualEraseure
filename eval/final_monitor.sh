#!/bin/bash
# 最终的评估完成状态监控脚本

echo "========================================"
echo "UniErase-Bench 最终状态监控"
echo "检查时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""

cd /home/mlsnrs/data/cgx/cgx_nas/T2I/Concept_Erasure/SexualErasure_CCS/eval

# 定义所有需要监控的目录
declare -a DIRS=(
    "TI2I_flux/results"
    "TI2I_flux/erased_results"
    "T2V_hunyuanvideo/results"
    "T2V_hunyuanvideo/erased_results"
    "TI2V_hunyuanvideo/results"
    "TI2V_hunyuanvideo/erased_results"
)

# 检查所有评估是否完成
ALL_COMPLETED=true
COMPLETED_COUNT=0

echo "📊 评估完成状态:"
echo ""

for DIR in "${DIRS[@]}"; do
    if [ -f "$DIR/nudenet_eval.csv" ]; then
        COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
        CASES=$(tail -n +2 "$DIR/nudenet_eval.csv" | wc -l)
        FLAGGED=$(tail -n +2 "$DIR/nudenet_eval.csv" | awk -F',' '$8=="True" {count++} END {print count+0}')
        MTIME=$(stat -c %y "$DIR/nudenet_eval.csv" | cut -d'.' -f1)

        printf "✅ %-40s 完成 (%3d 案例, %3d 标记, %s)\n" "$DIR" "$CASES" "$FLAGGED" "$MTIME"
    else
        ALL_COMPLETED=false
        TOTAL_FILES=$(find "$DIR" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.mp4" \) | wc -l)
        printf "❌ %-40s 待评估 (%3d 文件)\n" "$DIR" "$TOTAL_FILES"
    fi
done

echo ""
echo "完成进度: $COMPLETED_COUNT/${#DIRS[@]} ($(awk "BEGIN {printf \"%.1f\", $COMPLETED_COUNT/${#DIRS[@]}*100}")%)"
echo ""

# 检查是否有评估进程在运行
RUNNING_PROCS=$(ps aux | grep -v grep | grep "eval_unierase_bench.py" | wc -l)

if [ $RUNNING_PROCS -gt 0 ]; then
    echo "🔄 评估进程正在运行:"
    ps aux | grep -v grep | grep "eval_unierase_bench.py" | while read line; do
        PID=$(echo "$line" | awk '{print $2}')
        CPU=$(echo "$line" | awk '{print $3}')
        MEM=$(echo "$line" | awk '{print $4}')
        CMD=$(echo "$line" | awk '{for(i=11;i<=NF;i++)printf "%s ", $i; print ""}')

        # 提取results目录
        CURRENT_DIR=$(echo "$CMD" | grep -oP ' --results-dir [^ ]+' | awk '{print $2}')
        ELAPSED=$(ps -p $PID -o etime= | tr -d ' ')

        echo "  PID: $PID, CPU: ${CPU}%, MEM: ${MEM}%, 运行时间: $ELAPSED"
        echo "  目录: $CURRENT_DIR"
    done
else
    echo "✅ 没有评估进程在运行"
fi

echo ""
echo "========================================"
echo "GPU使用情况"
echo "========================================"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader | awk -F',' '{printf "GPU %s: %s - 利用率: %s%%, 显存: %sMB / %sMB\n", $1, $2, $3, $4, $5}'

echo ""

# 检查自动继续脚本的状态
if ps aux | grep -v grep | grep "auto_continue_eval.sh" > /dev/null; then
    echo "🤖 自动继续脚本: 🔄 运行中"
else
    echo "🤖 自动继续脚本: ⏸️  未运行"
fi

echo ""

# 如果所有评估都完成了，显示最终消息
if [ "$ALL_COMPLETED" = true ]; then
    echo "🎉 ========================================"
    echo "🎉 所有评估任务已完成!"
    echo "🎉 ========================================"
    echo ""
    echo "您可以运行以下命令生成最终汇总报告:"
    echo "  bash generate_summary_report.sh"
    echo ""
    exit 0
else
    echo "⏳ 评估仍在进行中，请耐心等待..."
    echo ""
    echo "您可以使用以下命令查看实时进度:"
    echo "  bash detailed_monitor.sh"
    echo ""
    exit 1
fi
