#!/bin/bash

echo "DOVER批处理版本4 (batch_size=4) 监控报告"
echo "========================================"
echo ""

# 检查进程状态
echo "进程状态:"
PROCESS_INFO=$(ps aux | grep "python.*batch_dover_evaluation_optimized" | grep -v grep | head -1)
if [ -n "$PROCESS_INFO" ]; then
    echo "$PROCESS_INFO" | awk '{print "✓ 进程ID:", $2, "状态:", $8, "CPU:", $3"%", "内存:", int($6/1024)"MB", "运行时间:", $10}'
else
    echo "✗ DOVER批处理进程未找到"
fi

echo ""

# 检查进度文件
PROGRESS_FILE="/home/mlsnrs/data/cgx/cgx_nas/T2I/Concept_Erasure/SexualErasure_CCS/eval/dover_batch_results_batch4_progress.json"

if [ -f "$PROGRESS_FILE" ]; then
    echo "评估进度:"
    echo "----------------------------------------"

    python3 << 'EOF'
import json
import sys

try:
    with open('/home/mlsnrs/data/cgx/cgx_nas/T2I/Concept_Erasure/SexualErasure_CCS/eval/dover_batch_results_batch4_progress.json', 'r') as f:
        data = json.load(f)

    total = data.get('total_count', 0)
    success = data.get('success_count', 0)
    fail = data.get('fail_count', 0)

    print(f"已处理: {total} 个视频")
    print(f"成功:   {success} 个 ({success/total*100:.1f}%)")
    print(f"失败:   {fail} 个 ({fail/total*100:.1f}%)")
    print(f"成功率: {success/total*100:.1f}%")

    # 与之前版本对比
    print(f"\n对比之前版本 (batch_size=8):")
    print(f"之前成功率: 25.0%")
    new_rate = success/total*100
    print(f"当前成功率: {new_rate:.1f}%")

    if new_rate > 25.0:
        improvement = ((new_rate - 25.0) / 25.0) * 100
        print(f"✓ 成功率提升: {improvement:.1f}%")
    elif new_rate < 25.0:
        decline = ((25.0 - new_rate) / 25.0) * 100
        print(f"✗ 成功率下降: {decline:.1f}%")
    else:
        print(f"= 成功率相同")

except Exception as e:
    print(f"无法读取进度文件: {e}")
    print("DOVER评估可能还在初始化中...")

EOF

else
    echo "进度文件尚未生成"
    echo "DOVER评估可能还在初始化中..."
fi

echo ""
echo "========================================"
echo "预计完成时间: 45-60分钟"
echo "========================================"