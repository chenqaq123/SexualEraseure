#!/bin/bash
# 详细的评估进度监控脚本

echo "========================================"
echo "UniErase-Bench 详细评估进度监控"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""

# 检查是否有正在运行的评估进程
RUNNING_PROCS=$(ps aux | grep -v grep | grep "eval_unierase_bench.py" | wc -l)

if [ $RUNNING_PROCS -gt 0 ]; then
    echo "🔄 正在运行的评估进程:"

    ps aux | grep -v grep | grep "eval_unierase_bench.py" | while read line; do
        PID=$(echo "$line" | awk '{print $2}')
        CMD=$(echo "$line" | awk '{for(i=11;i<=NF;i++)printf "%s ", $i; print ""}')

        echo "  PID: $PID"
        echo "  命令: $CMD"

        # 提取results目录
        RESULTS_DIR=$(echo "$CMD" | grep -oP ' --results-dir [^ ]+' | awk '{print $2}')
        if [ -n "$RESULTS_DIR" ]; then
            echo "  目标目录: $RESULTS_DIR"

            # 检查日志文件大小和修改时间
            if [ -f "$RESULTS_DIR/evaluation_log.txt" ]; then
                LOG_SIZE=$(du -h "$RESULTS_DIR/evaluation_log.txt" | awk '{print $1}')
                LOG_MTIME=$(stat -c %y "$RESULTS_DIR/evaluation_log.txt" | cut -d'.' -f1)
                echo "  日志大小: $LOG_SIZE, 最后更新: $LOG_MTIME"

                # 尝试从日志中提取进度信息
                PROGRESS=$(grep -oP '\[\d+/\d+\]' "$RESULTS_DIR/evaluation_log.txt" | tail -1)
                if [ -n "$PROGRESS" ]; then
                    echo "  进度: $PROGRESS"
                fi
            fi
        fi
        echo ""
    done
else
    echo "✅ 没有评估进程在运行"
    echo ""
fi

echo "========================================"
echo "已完成评估统计"
echo "========================================"

for DIR in TI2I_flux/results TI2I_flux/erased_results T2V_hunyuanvideo/results T2V_hunyuanvideo/erased_results TI2V_hunyuanvideo/results TI2V_hunyuanvideo/erased_results; do
    if [ -f "$DIR/nudenet_eval.csv" ]; then
        # 计算文件大小
        SIZE=$(du -h "$DIR/nudenet_eval.csv" | awk '{print $1}')
        # 计算评估案例数
        CASES=$(tail -n +2 "$DIR/nudenet_eval.csv" | wc -l)
        # 最后修改时间
        MTIME=$(stat -c %y "$DIR/nudenet_eval.csv" | cut -d'.' -f1)

        echo "✅ $DIR"
        echo "   案例: $CASES, 大小: $SIZE, 完成: $MTIME"
    fi
done

echo ""
echo "========================================"
echo "GPU状态"
echo "========================================"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader | awk -F',' '{printf "GPU %s: %s\n  利用率: %s%%, 显存: %sMB / %sMB, 温度: %s°C\n\n", $1, $2, $3, $4, $5, $6}'

echo ""
echo "========================================"
echo "系统资源"
echo "========================================"
echo "CPU负载: $(uptime | awk -F'load average:' '{print $2}')"
echo "内存使用: $(free -h | grep Mem | awk '{printf "总计: %s, 已用: %s, 可用: %s (%.1f%%)\n", $2, $3, $7, ($3/$2)*100}')"

echo ""
