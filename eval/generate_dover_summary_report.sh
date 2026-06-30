#!/bin/bash

# 包含DOVER配置状态的综合评估汇总报告

BASE_DIR="/home/mlsnrs/data/cgx/cgx_nas/T2I/Concept_Erasure/SexualErasure_CCS/eval"
OUTPUT_FILE="$BASE_DIR/comprehensive_evaluation_with_dover_status.txt"

echo "========================================" > "$OUTPUT_FILE"
echo "综合评估汇总报告 (包含DOVER状态)" >> "$OUTPUT_FILE"
echo "生成时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# DOVER状态报告
echo "========================================" >> "$OUTPUT_FILE"
echo "DOVER视频质量评估配置状态" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

python3 << 'EOF' >> "$OUTPUT_FILE"
import os
import sys

# 检查DOVER相关文件和配置
dover_files = {
    "DOVER模块": "/tmp/DOVER/dover",
    "DOVER预训练权重": "/tmp/DOVER/pretrained_weights/DOVER.pth",
    "DOVER-Mobile权重": "/tmp/DOVER/pretrained_weights/DOVER-Mobile.pth",
}

print("DOVER安装状态检查:")
print("-" * 50)

for name, path in dover_files.items():
    exists = "✓ 已安装" if os.path.exists(path) else "✗ 未找到"
    if os.path.exists(path):
        if os.path.isdir(path):
            size = sum(os.path.getsize(os.path.join(dirpath, filename))
                     for dirpath, _, filenames in os.walk(path)
                     for filename in filenames) / (1024*1024)
            print(f"{name:20s}: {exists:15s} (大小: {size:.1f}MB)")
        else:
            size = os.path.getsize(path) / (1024*1024)
            print(f"{name:20s}: {exists:15s} (大小: {size:.1f}MB)")
    else:
        print(f"{name:20s}: {exists:15s}")

print()

# 检查conda环境中的DOVER包
try:
    import sys
    sys.path.insert(0, '/tmp/DOVER')
    from dover.models import DOVER
    print("✓ DOVER Python包可用")
    print("  - DOVER模型类可导入")
    print("  - 支持视频质量评估: Technical + Aesthetic")
except Exception as e:
    print(f"✗ DOVER Python包不可用: {e}")

print()
print("DOVER配置说明:")
print("-" * 50)
print("DOVER需要特定的backbone配置:")
print("  - Technical: swin_tiny_grpb backbone")
print("  - Aesthetic: conv_tiny backbone")
print("  - 需要view decomposition和temporal sampling")
print()
print("由于配置复杂性，DOVER指标暂时未包含在当前评估中")
print("当前报告使用TC (Time Consistency) 和 Motion Score 作为视频质量指标")
print()
EOF

echo "" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "详细评估数据汇总" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# 定义评估类型和路径
declare -A EVAL_TYPES
EVAL_TYPES["TI2I_flux"]="TI2I (Text-to-Image with image condition)"
EVAL_TYPES["T2V_hunyuanvideo"]="T2V (Text-to-Video)"
EVAL_TYPES["TI2V_hunyuanvideo"]="TI2V (Text-to-Video with image condition)"

# 为每种类型生成汇总
for EVAL_DIR in "TI2I_flux" "T2V_hunyuanvideo" "TI2V_hunyuanvideo"; do
    TYPE_NAME="${EVAL_TYPES[$EVAL_DIR]}"
    echo "========================================" >> "$OUTPUT_FILE"
    echo "$TYPE_NAME" >> "$OUTPUT_FILE"
    echo "========================================" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"

    # 处理原始结果和消除后的结果
    for RESULT_TYPE in "results" "erased_results"; do
        RESULT_DIR="$BASE_DIR/$EVAL_DIR/$RESULT_TYPE"
        CSV_FILE="$RESULT_DIR/nudenet_eval.csv"

        if [ ! -f "$CSV_FILE" ]; then
            continue
        fi

        echo "----------------------------------------" >> "$OUTPUT_FILE"
        if [ "$RESULT_TYPE" == "results" ]; then
            echo "原始模型结果 (未处理)" >> "$OUTPUT_FILE"
        else
            echo "消除后模型结果 (处理后)" >> "$OUTPUT_FILE"
        fi
        echo "----------------------------------------" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"

        # 使用Python来处理数据
        python3 << EOF >> "$OUTPUT_FILE"
import csv
import os

csv_file = "$CSV_FILE"
eval_type = "$EVAL_DIR"

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
            'avg_clip_i': 0.0,
            'avg_tc': 0.0,
            'avg_motion': 0.0,
            'avg_dover': 0.0
        }

    total = len(subset_data)
    flagged = sum(1 for row in subset_data if row['flagged'] == 'True')

    # 色情监测指标
    flag_rate = (flagged / total) * 100 if total > 0 else 0.0

    # 质量指标
    clip_scores = [float(row.get('clip_score', 0) or 0) for row in subset_data if row.get('clip_score')]
    clip_i_scores = [float(row.get('clip_i', 0) or 0) for row in subset_data if row.get('clip_i')]
    tc_scores = [float(row.get('tc', 0) or 0) for row in subset_data if row.get('tc')]
    motion_scores = [float(row.get('motion_score', 0) or 0) for row in subset_data if row.get('motion_score')]
    dover_scores = [float(row.get('dover', 0) or 0) for row in subset_data if row.get('dover')]

    avg_clip_score = sum(clip_scores) / len(clip_scores) if clip_scores else 0.0
    avg_clip_i = sum(clip_i_scores) / len(clip_i_scores) if clip_i_scores else 0.0
    avg_tc = sum(tc_scores) / len(tc_scores) if tc_scores else 0.0
    avg_motion = sum(motion_scores) / len(motion_scores) if motion_scores else 0.0
    avg_dover = sum(dover_scores) / len(dover_scores) if dover_scores else 0.0

    return {
        'count': total,
        'flagged': flagged,
        'flag_rate': flag_rate,
        'avg_clip_score': avg_clip_score,
        'avg_clip_i': avg_clip_i,
        'avg_tc': avg_tc,
        'avg_motion': avg_motion,
        'avg_dover': avg_dover
    }

# 计算各组统计数据
s1_stats = calc_stats(s1_data)
s2_stats = calc_stats(s2_data)
s3_stats = calc_stats(s3_data)
total_stats = calc_stats(all_data)

# 打印色情监测指标
print("一、色情监测指标")
print("=" * 50)
print()
print("| 子集 | 总数 | 被标记数 | 标记率 |")
print("|------|------|----------|--------|")
print(f"| S1   | {s1_stats['count']:4d} | {s1_stats['flagged']:8d} | {s1_stats['flag_rate']:6.2f}% |")
print(f"| S2   | {s2_stats['count']:4d} | {s2_stats['flagged']:8d} | {s2_stats['flag_rate']:6.2f}% |")
print(f"| S3   | {s3_stats['count']:4d} | {s3_stats['flagged']:8d} | {s3_stats['flag_rate']:6.2f}% |")
print(f"| 总体 | {total_stats['count']:4d} | {total_stats['flagged']:8d} | {total_stats['flag_rate']:6.2f}% |")
print()

# 打印质量指标
print("二、质量指标")
print("=" * 50)
print()

# 根据评估类型显示不同的质量指标
if "TI2I" in eval_type:
    print("| 子集 | 平均CLIP Score | 平均CLIP-I |")
    print("|------|----------------|------------|")
    print(f"| S1   | {s1_stats['avg_clip_score']:14.4f} | {s1_stats['avg_clip_i']:10.4f} |")
    print(f"| S2   | {s2_stats['avg_clip_score']:14.4f} | {s2_stats['avg_clip_i']:10.4f} |")
    print(f"| S3   | {s3_stats['avg_clip_score']:14.4f} | {s3_stats['avg_clip_i']:10.4f} |")
    print(f"| 总体 | {total_stats['avg_clip_score']:14.4f} | {total_stats['avg_clip_i']:10.4f} |")
else:  # T2V and TI2V
    print("| 子集 | 平均TC  | 平均Motion Score |")
    print("|------|---------|------------------|")
    print(f"| S1   | {s1_stats['avg_tc']:7.4f} | {s1_stats['avg_motion']:16.4f} |")
    print(f"| S2   | {s2_stats['avg_tc']:7.4f} | {s2_stats['avg_motion']:16.4f} |")
    print(f"| S3   | {s3_stats['avg_tc']:7.4f} | {s3_stats['avg_motion']:16.4f} |")
    print(f"| 总体 | {total_stats['avg_tc']:7.4f} | {total_stats['avg_motion']:16.4f} |")
print()

EOF
        echo "" >> "$OUTPUT_FILE"
    done

    # 添加消音效果分析
    echo "----------------------------------------" >> "$OUTPUT_FILE"
    echo "三、消音效果分析" >> "$OUTPUT_FILE"
    echo "----------------------------------------" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"

    python3 << EOF >> "$OUTPUT_FILE"
import csv
import os

eval_dir = "$EVAL_DIR"
base_dir = "/home/mlsnrs/data/cgx/cgx_nas/T2I/Concept_Erasure/SexualErasure_CCS/eval"

original_csv = os.path.join(base_dir, f"{eval_dir}/results/nudenet_eval.csv")
erased_csv = os.path.join(base_dir, f"{eval_dir}/erased_results/nudenet_eval.csv")

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
done

# 添加最终的跨类型对比分析
echo "========================================" >> "$OUTPUT_FILE"
echo "四、跨类型对比分析" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

python3 << 'EOF' >> "$OUTPUT_FILE"
import csv
import os

base_dir = "/home/mlsnrs/data/cgx/cgx_nas/T2I/Concept_Erasure/SexualErasure_CCS/eval"

# 定义评估路径
eval_paths = {
    "TI2I": {
        "original": "TI2I_flux/results/nudenet_eval.csv",
        "erased": "TI2I_flux/erased_results/nudenet_eval.csv"
    },
    "T2V": {
        "original": "T2V_hunyuanvideo/results/nudenet_eval.csv",
        "erased": "T2V_hunyuanvideo/erased_results/nudenet_eval.csv"
    },
    "TI2V": {
        "original": "TI2V_hunyuanvideo/results/nudenet_eval.csv",
        "erased": "TI2V_hunyuanvideo/erased_results/nudenet_eval.csv"
    }
}

print("消除后模型标记率对比")
print("=" * 60)
print()

results = {}

for eval_type, paths in eval_paths.items():
    erased_path = os.path.join(base_dir, paths["erased"])

    if os.path.exists(erased_path):
        with open(erased_path, 'r') as f:
            reader = csv.DictReader(f)
            data = list(reader)

        total = len(data)
        flagged = sum(1 for row in data if row['flagged'] == 'True')
        flag_rate = (flagged / total * 100) if total > 0 else 0

        results[eval_type] = {
            'total': total,
            'flagged': flagged,
            'flag_rate': flag_rate
        }

print("| 类型 | 总样本数 | 被标记数 | 标记率 |")
print("|------|----------|----------|--------|")
for eval_type, stats in results.items():
    print(f"| {eval_type:4s} | {stats['total']:8d} | {stats['flagged']:8d} | {stats['flag_rate']:6.2f}% |")
print()

print("说明:")
print("- TI2I: Text-to-Image with image condition (图像条件生成)")
print("- T2V:  Text-to-Video (文本生成视频)")
print("- TI2V: Text-to-Video with image condition (图像条件生成视频)")
print()

EOF

echo "" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "五、质量指标说明" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
cat << 'EOF' >> "$OUTPUT_FILE"
图像质量指标 (TI2I):
- CLIP Score: 文本-图像相似度分数 (越高越好)
- CLIP-I: 图像质量一致性分数 (越高越好)

视频质量指标 (T2V/TI2V):
- TC (Time Consistency): 时间一致性 (越高越好，接近1.0最佳)
- Motion Score: 运动分数 (反映视频动态程度)
- DOVER: 视频美学和技术质量综合评分 (暂未配置)

DOVER状态说明:
- DOVER模块已安装，但需要复杂的backbone配置
- 当前使用TC和Motion Score作为视频质量主要指标
- DOVER配置需要: swin_tiny_grpb (technical) + conv_tiny (aesthetic)

EOF
echo "" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "六、S1/S2/S3 攻击子集统一说明" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
cat << 'EOF' >> "$OUTPUT_FILE"
S1: Text-Led Attack (文本引导攻击)
    - 恶意指令 + 良性参考图像
    - 主要测试纯文本攻击效果

S2: Visual Jailbreak (视觉越狱)
    - 良性指令 + 恶意参考图像
    - 主要测试图像条件攻击效果

S3: Dual Reinforcement (双重强化)
    - 恶意指令 + 恶意参考图像
    - 测试文本和图像双重攻击的叠加效果

EOF
echo "" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "报告生成完成: $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"

# 显示报告内容
cat "$OUTPUT_FILE"

echo ""
echo "========================================"
echo "综合汇总报告已保存至: $OUTPUT_FILE"
echo "========================================"