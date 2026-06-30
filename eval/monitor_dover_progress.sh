#!/bin/bash

echo "DOVER批量评估进度监控"
echo "===================="
echo ""

# 检查进程状态
echo "进程状态:"
ps aux | grep "python.*batch_dover_evaluation_optimized" | grep -v grep | awk '{print "进程ID:", $2, "内存:", $6/1024"MB", CPU:", $3"%", "运行时间:", $10}'

echo ""

# 检查进度文件
PROGRESS_FILE="/home/mlsnrs/data/cgx/cgx_nas/T2I/Concept_Erasure/SexualErasure_CCS/eval/dover_batch_results_optimized_progress.json"

if [ -f "$PROGRESS_FILE" ]; then
    echo "进度文件内容:"
    cat "$PROGRESS_FILE" | python3 -c "
import json
import sys
data = json.load(sys.stdin)
print(f\"已处理: {data.get('total_count', 0)} 个视频\")
print(f\"成功: {data.get('success_count', 0)} 个\")
print(f\"失败: {data.get('fail_count', 0)} 个\")
"
else
    echo "进度文件尚未生成，评估可能还在初始化..."
fi

echo ""
echo "最新结果文件:"
ls -la /home/mlsnrs/data/cgx/cgx_nas/T2I/Concept_Erasure/SexualErasure_CCS/eval/dover_batch_optimized* 2>/dev/null