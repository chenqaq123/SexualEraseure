#!/bin/bash

echo "DOVER多GPU并行评估监控"
echo "======================"
echo ""

# 检查进程状态
echo "运行中的进程:"
PROCESS_COUNT=$(ps aux | grep "multi_gpu_dover_fixed" | grep -v grep | wc -l)
echo "活跃进程数: $PROCESS_COUNT"
echo ""

# 检查GPU状态
echo "GPU状态:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits | awk '{printf "  GPU %s: 利用率=%s%%, 显存=%sMB, 功率=%sW\n", $1, $2, $3, $4}'
echo ""

# 检查结果目录
OUTPUT_DIR="/home/mlsnrs/data/cgx/cgx_nas/T2I/Concept_Erasure/SexualErasure_CCS/multi_gpu_results"

if [ -d "$OUTPUT_DIR" ]; then
    echo "进度统计:"
    echo "----------"

    python3 << 'EOF'
import json
import os
import glob

output_dir = "/home/mlsnrs/data/cgx/cgx_nas/T2I/Concept_Erasure/SexualErasure_CCS/multi_gpu_results"

total_processed = 0
total_success = 0
total_fail = 0

# 检查临时文件
temp_files = glob.glob(os.path.join(output_dir, "gpu_*_temp.json"))
if temp_files:
    print(f"找到 {len(temp_files)} 个临时进度文件:")
    for temp_file in temp_files:
        try:
            with open(temp_file, 'r') as f:
                data = json.load(f)
                gpu_id = data.get('gpu_id', '?')
                processed = data.get('processed', 0)
                success = data.get('success_count', 0)
                fail = data.get('fail_count', 0)
                total = data.get('total', 0)

                total_processed += processed
                total_success += success
                total_fail += fail

                progress_pct = (processed / total * 100) if total > 0 else 0
                print(f"  GPU {gpu_id}: {processed}/{total} ({progress_pct:.1f}%) | 成功:{success}, 失败:{fail}")
        except Exception as e:
            print(f"  无法读取 {os.path.basename(temp_file)}: {e}")

    print(f"\n总体进度: {total_processed} 个视频已处理")
    print(f"成功: {total_success}, 失败: {total_fail}")

    if total_processed > 0:
        success_rate = (total_success / total_processed) * 100
        print(f"成功率: {success_rate:.1f}%")

else:
    print("尚未生成临时进度文件")
    print("DOVER模型可能还在初始化中...")

# 检查最终结果文件
final_files = glob.glob(os.path.join(output_dir, "gpu_*_results.json"))
if final_files:
    print(f"\n已完成: {len(final_files)} 个GPU")

EOF

else
    echo "输出目录尚未创建"
    echo "DOVER模型可能还在初始化中..."
fi

echo ""
echo "======================"
echo "预计完成时间: 30-45分钟"
echo "======================"
