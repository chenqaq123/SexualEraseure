#!/bin/bash
# 监控评估进度的脚本

echo "========================================"
echo "UniErase-Bench 评估进度监控"
echo "========================================"
echo ""

# 定义所有需要监控的目录
declare -a DIRS=(
    "TI2I_flux/erased_results"
    "T2V_hunyuanvideo/results"
    "T2V_hunyuanvideo/erased_results"
    "TI2V_hunyuanvideo/results"
    "TI2V_hunyuanvideo/erased_results"
)

cd /home/mlsnrs/data/cgx/cgx_nas/T2I/Concept_Erasure/SexualErasure_CCS/eval

for DIR in "${DIRS[@]}"; do
    echo "📁 $DIR"

    # 检查是否有评估报告
    if [ -f "$DIR/nudenet_eval.csv" ]; then
        echo "  ✅ 已完成评估"

        # 显示评估报告的行数（不包括header）
        LINES=$(tail -n +2 "$DIR/nudenet_eval.csv" | wc -l)
        echo "  📊 评估案例数: $LINES"

    elif [ -f "$DIR/evaluation_log.txt" ]; then
        echo "  🔄 正在评估中..."

        # 检查日志文件的最后修改时间
        MTIME=$(stat -c %Y "$DIR/evaluation_log.txt")
        NOW=$(date +%s)
        DIFF=$((NOW - MTIME))

        if [ $DIFF -lt 60 ]; then
            echo "  ⏱️  日志更新: ${DIFF}秒前 (活跃)"
        else
            echo "  ⏱️  日志更新: ${DIFF}秒前 (可能卡住)"
        fi

        # 显示日志的最后几行
        echo "  📝 最近日志:"
        tail -3 "$DIR/evaluation_log.txt" | sed 's/^/     /'

    else
        echo "  ❌ 未开始评估"
    fi

    # 显示目录中的文件数量
    TOTAL_FILES=$(find "$DIR" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.mp4" -o -name "*.avi" \) | wc -l)
    echo "  📁 结果文件数: $TOTAL_FILES"
    echo ""
done

echo "========================================"
echo "进程检查"
echo "========================================"

# 检查是否有评估脚本在运行
PYTHON_PROCS=$(ps aux | grep -v grep | grep "eval_unierase_bench.py" | wc -l)
if [ $PYTHON_PROCS -gt 0 ]; then
    echo "🔄 评估进程正在运行 (数量: $PYTHON_PROCS)"
    ps aux | grep -v grep | grep "eval_unierase_bench.py" | awk '{print "  PID:", $2, "CPU:", $3"%", "MEM:", $4"%", "CMD:", $11, $12, $13}'
else
    echo "✅ 没有评估进程在运行"
fi

echo ""
echo "========================================"
echo "GPU使用情况"
echo "========================================"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | awk -F',' '{print "  GPU "$1": "$2" - 利用率: "$3"% - 显存: "$4"MB / "$5"MB"}'

echo ""
