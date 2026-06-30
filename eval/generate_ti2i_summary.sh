#!/bin/bash

# TI2I Flux Image-Condition 综合汇总报告生成脚本
# 针对 image-condition 的情况，分别汇报 s1 s2 s3 和总体的色情监测指标和质量指标

BASE_DIR="/home/mlsnrs/data/cgx/cgx_nas/T2I/Concept_Erasure/SexualErasure_CCS/eval"
OUTPUT_FILE="$BASE_DIR/ti2i_image_condition_summary_report.txt"

echo "========================================" > "$OUTPUT_FILE"
echo "TI2I Flux Image-Condition 综合评估报告" >> "$OUTPUT_FILE"
echo "生成时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# 处理原始结果和消除后的结果
for RESULT_TYPE in "results" "erased_results"; do
    RESULT_DIR="$BASE_DIR/TI2I_flux/$RESULT_TYPE"
    CSV_FILE="$RESULT_DIR/nudenet_eval.csv"

    if [ ! -f "$CSV_FILE" ]; then
        continue
    fi

    echo "========================================" >> "$OUTPUT_FILE"
    if [ "$RESULT_TYPE" == "results" ]; then
        echo "原始模型结果 (未处理)" >> "$OUTPUT_FILE"
    else
        echo "消除后模型结果 (处理后)" >> "$OUTPUT_FILE"
    fi
    echo "========================================" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"

    # 使用Python来处理数据
    python3 << EOF >> "$OUTPUT_FILE"
import csv
import sys

csv_file = "$CSV_FILE"

# 读取数据
data = []
with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

# 分类统计
s1_data = [row for row in data if row['attack_subset'] == 'S1']
s2_data = [row for row in data if row['attack_subset'] == 'S2']
s3_data = [row for row in data if row['attack_subset'] == 'S3']
all_data = data

# 计算统计函数
def calc_stats(subset_data):
    if not subset_data:
        return {
            'count': 0,
            'flagged': 0,
            'flag_rate': 0.0,
            'avg_clip_score': 0.0,
            'avg_clip_i': 0.0
        }

    total = len(subset_data)
    flagged = sum(1 for row in subset_data if row['flagged'] == 'True')

    # 色情监测指标
    flag_rate = (flagged / total) * 100 if total > 0 else 0.0

    # 质量指标
    clip_scores = [float(row.get('clip_score', 0) or 0) for row in subset_data if row.get('clip_score')]
    clip_i_scores = [float(row.get('clip_i', 0) or 0) for row in subset_data if row.get('clip_i')]

    avg_clip_score = sum(clip_scores) / len(clip_scores) if clip_scores else 0.0
    avg_clip_i = sum(clip_i_scores) / len(clip_i_scores) if clip_i_scores else 0.0

    return {
        'count': total,
        'flagged': flagged,
        'flag_rate': flag_rate,
        'avg_clip_score': avg_clip_score,
        'avg_clip_i': avg_clip_i
    }

# 计算各组统计数据
s1_stats = calc_stats(s1_data)
s2_stats = calc_stats(s2_data)
s3_stats = calc_stats(s3_data)
total_stats = calc_stats(all_data)

# 打印色情监测指标
print("=" * 60)
print("一、色情监测指标")
print("=" * 60)
print()
print("| 子集 | 总数 | 被标记数 | 标记率 |")
print("|------|------|----------|--------|")
print(f"| S1   | {s1_stats['count']:4d} | {s1_stats['flagged']:8d} | {s1_stats['flag_rate']:6.2f}% |")
print(f"| S2   | {s2_stats['count']:4d} | {s2_stats['flagged']:8d} | {s2_stats['flag_rate']:6.2f}% |")
print(f"| S3   | {s3_stats['count']:4d} | {s3_stats['flagged']:8d} | {s3_stats['flag_rate']:6.2f}% |")
print(f"| 总体 | {total_stats['count']:4d} | {total_stats['flagged']:8d} | {total_stats['flag_rate']:6.2f}% |")
print()

# 打印质量指标
print("=" * 60)
print("二、质量指标")
print("=" * 60)
print()
print("| 子集 | 平均CLIP Score | 平均CLIP-I |")
print("|------|----------------|------------|")
print(f"| S1   | {s1_stats['avg_clip_score']:14.4f} | {s1_stats['avg_clip_i']:10.4f} |")
print(f"| S2   | {s2_stats['avg_clip_score']:14.4f} | {s2_stats['avg_clip_i']:10.4f} |")
print(f"| S3   | {s3_stats['avg_clip_score']:14.4f} | {s3_stats['avg_clip_i']:10.4f} |")
print(f"| 总体 | {total_stats['avg_clip_score']:14.4f} | {total_stats['avg_clip_i']:10.4f} |")
print()

# S1/S2/S3 子集说明
print("=" * 60)
print("S1/S2/S3 子集说明")
print("=" * 60)
print()
print("S1: Text-Led Attack (文本引导攻击)")
print("    - 恶意指令 + 良性参考图像")
print()
print("S2: Visual Jailbreak (视觉越狱)")
print("    - 良性指令 + 恶意参考图像")
print()
print("S3: Dual Reinforcement (双重强化)")
print("    - 恶意指令 + 恶意参考图像")
print()

EOF
    echo "" >> "$OUTPUT_FILE"
done

echo "========================================" >> "$OUTPUT_FILE"
echo "三、消音效果分析" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# 计算消音效果（如果有两个结果）
python3 << 'EOF' >> "$OUTPUT_FILE"
import csv
import os

base_dir = "/home/mlsnrs/data/cgx/cgx_nas/T2I/Concept_Erasure/SexualErasure_CCS/eval"
original_csv = os.path.join(base_dir, "TI2I_flux/results/nudenet_eval.csv")
erased_csv = os.path.join(base_dir, "TI2I_flux/erased_results/nudenet_eval.csv")

if os.path.exists(original_csv) and os.path.exists(erased_csv):
    # 读取数据
    original_data = []
    with open(original_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            original_data.append(row)

    erased_data = []
    with open(erased_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            erased_data.append(row)

    # 分类统计
    def get_subset_stats(data, subset):
        subset_data = [row for row in data if row['attack_subset'] == subset]
        if not subset_data:
            return {'count': 0, 'flagged': 0, 'flag_rate': 0.0}

        total = len(subset_data)
        flagged = sum(1 for row in subset_data if row['flagged'] == 'True')
        flag_rate = (flagged / total) * 100 if total > 0 else 0.0

        return {'count': total, 'flagged': flagged, 'flag_rate': flag_rate}

    # 对比分析
    subsets = ['S1', 'S2', 'S3']

    print("| 子集 | 原始标记率 | 消除后标记率 | 下降幅度 | 下降百分比 |")
    print("|------|------------|--------------|----------|------------|")

    for subset in subsets:
        orig_stats = get_subset_stats(original_data, subset)
        erased_stats = get_subset_stats(erased_data, subset)

        if orig_stats['count'] > 0 and erased_stats['count'] > 0:
            reduction = orig_stats['flag_rate'] - erased_stats['flag_rate']
            reduction_pct = (reduction / orig_stats['flag_rate'] * 100) if orig_stats['flag_rate'] > 0 else 0

            print(f"| {subset}   | {orig_stats['flag_rate']:10.2f}% | {erased_stats['flag_rate']:12.2f}% | {reduction:8.2f}% | {reduction_pct:10.1f}% |")

    # 总体对比
    orig_total = get_subset_stats(original_data, 'ALL')
    erased_total = get_subset_stats(erased_data, 'ALL')

    # 计算总体
    orig_total_count = len(original_data)
    orig_total_flagged = sum(1 for row in original_data if row['flagged'] == 'True')
    orig_total_rate = (orig_total_flagged / orig_total_count * 100) if orig_total_count > 0 else 0

    erased_total_count = len(erased_data)
    erased_total_flagged = sum(1 for row in erased_data if row['flagged'] == 'True')
    erased_total_rate = (erased_total_flagged / erased_total_count * 100) if erased_total_count > 0 else 0

    total_reduction = orig_total_rate - erased_total_rate
    total_reduction_pct = (total_reduction / orig_total_rate * 100) if orig_total_rate > 0 else 0

    print(f"| 总体 | {orig_total_rate:10.2f}% | {erased_total_rate:12.2f}% | {total_reduction:8.2f}% | {total_reduction_pct:10.1f}% |")
    print()

else:
    print("无法进行消音效果分析 - 缺少原始或消除后的数据文件")
    print()

EOF

echo "" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "报告生成完成: $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"

cat "$OUTPUT_FILE"
echo ""
echo "========================================"
echo "报告已保存至: $OUTPUT_FILE"
echo "========================================"