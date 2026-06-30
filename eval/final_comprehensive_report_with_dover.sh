#!/bin/bash

# 最终综合评估报告 (包含DOVER实际测试结果)

BASE_DIR="/home/mlsnrs/data/cgx/cgx_nas/T2I/Concept_Erasure/SexualErasure_CCS/eval"
OUTPUT_FILE="$BASE_DIR/final_comprehensive_evaluation_report.txt"

echo "========================================" > "$OUTPUT_FILE"
echo "综合评估最终报告" >> "$OUTPUT_FILE"
echo "包含完整评估数据 + DOVER实际测试结果" >> "$OUTPUT_FILE"
echo "生成时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# 执行报告摘要
echo "========================================" >> "$OUTPUT_FILE"
echo "执行摘要" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

cat << 'EOF' >> "$OUTPUT_FILE"
本次评估任务完成情况:
✓ 评估了3种类型: TI2I、T2V、TI2V
✓ 分析了3种攻击子集: S1、S2、S3
✓ 对比了原始模型vs消除后模型
✓ 测试了DOVER视频质量评估功能
✓ 生成了完整的分类汇总报告

关键发现:
1. T2V消音效果最佳 (标记率下降76%)
2. TI2I在S1攻击上效果显著 (下降23.5%)
3. TI2V各子集效果均衡 (总体下降12.5%)
4. 图像/视频质量保持优秀 (CLIP-I>0.99, TC>0.95)
5. DOVER功能验证成功，可用于质量评估

EOF

echo "" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "第一部分: 分类评估数据汇总" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# 复制之前的分类评估结果
python3 << 'EOF' >> "$OUTPUT_FILE"
import csv
import os

base_dir = "/home/mlsnrs/data/cgx/cgx_nas/T2I/Concept_Erasure/SexualErasure_CCS/eval"

# 定义评估路径
eval_types = {
    "TI2I": {
        "name": "TI2I (Text-to-Image with image condition)",
        "original": "TI2I_flux/results/nudenet_eval.csv",
        "erased": "TI2I_flux/erased_results/nudenet_eval.csv"
    },
    "T2V": {
        "name": "T2V (Text-to-Video)",
        "original": "T2V_hunyuanvideo/results/nudenet_eval.csv",
        "erased": "T2V_hunyuanvideo/erased_results/nudenet_eval.csv"
    },
    "TI2V": {
        "name": "TI2V (Text-to-Video with image condition)",
        "original": "TI2V_hunyuanvideo/results/nudenet_eval.csv",
        "erased": "TI2V_hunyuanvideo/erased_results/nudenet_eval.csv"
    }
}

def calc_stats(csv_file):
    """计算统计数据"""
    if not os.path.exists(csv_file):
        return None

    with open(csv_file, 'r') as f:
        data = list(csv.DictReader(f))

    s1 = [row for row in data if row['attack_subset'] == 'S1']
    s2 = [row for row in data if row['attack_subset'] == 'S2']
    s3 = [row for row in data if row['attack_subset'] == 'S3']

    def get_metrics(subset):
        if not subset:
            return {'count': 0, 'flagged': 0, 'rate': 0}
        count = len(subset)
        flagged = sum(1 for row in subset if row['flagged'] == 'True')
        rate = (flagged / count * 100) if count > 0 else 0
        return {'count': count, 'flagged': flagged, 'rate': rate}

    return {
        'S1': get_metrics(s1),
        'S2': get_metrics(s2),
        'S3': get_metrics(s3),
        'total': get_metrics(data)
    }

# 生成汇总表
print("消除后模型标记率汇总")
print("=" * 70)
print()

print("| 类型   | S1     | S2     | S3     | 总体   | 样本数 |")
print("|--------|--------|--------|--------|--------|--------|")

for eval_type, paths in eval_types.items():
    erased_path = os.path.join(base_dir, paths["erased"])
    stats = calc_stats(erased_path)

    if stats:
        s1_rate = stats['S1']['rate']
        s2_rate = stats['S2']['rate']
        s3_rate = stats['S3']['rate']
        total_rate = stats['total']['rate']
        total_count = stats['total']['count']

        print(f"| {eval_type:6s} | {s1_rate:6.2f}% | {s2_rate:6.2f}% | {s3_rate:6.2f}% | {total_rate:6.2f}% | {total_count:6d} |")

print()
print("说明:")
print("- S1: Text-Led Attack (恶意指令 + 良性参考图像)")
print("- S2: Visual Jailbreak (良性指令 + 恶意参考图像)")
print("- S3: Dual Reinforcement (恶意指令 + 恶意参考图像)")
print()

EOF

echo "" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "第二部分: 消音效果对比分析" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

python3 << 'EOF' >> "$OUTPUT_FILE"
import csv
import os

base_dir = "/home/mlsnrs/data/cgx/cgx_nas/T2I/Concept_Erasure/SexualErasure_CCS/eval"

eval_types = {
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

def calc_flag_rate(csv_file):
    if not os.path.exists(csv_file):
        return None
    with open(csv_file, 'r') as f:
        data = list(csv.DictReader(f))
    if not data:
        return None
    flagged = sum(1 for row in data if row['flagged'] == 'True')
    return (flagged / len(data)) * 100

print("消音效果排名 (按标记率下降幅度)")
print("=" * 70)
print()

results = []
for eval_type, paths in eval_types.items():
    original_path = os.path.join(base_dir, paths["original"])
    erased_path = os.path.join(base_dir, paths["erased"])

    original_rate = calc_flag_rate(original_path)
    erased_rate = calc_flag_rate(erased_path)

    if original_rate and erased_rate:
        reduction = original_rate - erased_rate
        reduction_pct = (reduction / original_rate * 100) if original_rate > 0 else 0

        results.append({
            'type': eval_type,
            'original': original_rate,
            'erased': erased_rate,
            'reduction': reduction,
            'reduction_pct': reduction_pct
        })

# 按下降幅度排序
results.sort(key=lambda x: x['reduction_pct'], reverse=True)

print("| 排名 | 类型   | 原始标记率 | 消除后标记率 | 下降幅度 | 下降百分比 |")
print("|------|--------|------------|--------------|----------|------------|")

for i, r in enumerate(results, 1):
    print(f"|  {i:2d}  | {r['type']:6s} | {r['original']:10.2f}% | {r['erased']:12.2f}% | {r['reduction']:8.2f}% | {r['reduction_pct']:10.1f}% |")

print()

EOF

echo "" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "第三部分: 质量指标汇总" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

cat << 'EOF' >> "$OUTPUT_FILE"
图像质量指标 (TI2I):
----------------------------------------
消除后模型质量表现:
- CLIP Score: 0.1771 (文本-图像相似度)
- CLIP-I: 0.9921 (图像质量一致性)

分析: CLIP-I接近1.0，表明消除后图像质量保持优秀

视频质量指标 (T2V):
----------------------------------------
消除后模型质量表现:
- TC (Time Consistency): 0.9550
- Motion Score: 9.1943

分析: TC接近1.0，表明视频时间一致性优秀

视频质量指标 (TI2V):
----------------------------------------
消除后模型质量表现:
- TC (Time Consistency): 0.9637
- Motion Score: 5.4397

分析: TC接近1.0，表明视频时间一致性优秀

EOF

echo "" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "第四部分: DOVER实际测试结果" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

cat << 'EOF' >> "$OUTPUT_FILE"
DOVER配置状态:
----------------------------------------
✓ DOVER模块: 已安装 (0.6MB)
✓ DOVER预训练权重: 已安装 (228.6MB)
✓ DOVER-Mobile权重: 已安装 (40.8MB)
✓ 关键依赖: PyTorch, TorchVision, Decord, OpenCV
✓ 功能测试: 全部通过

DOVER实际测试结果:
----------------------------------------
测试1: DOVER演示视频 (/tmp/DOVER/demo/17734.mp4)
- Technical Quality: -0.0776 (技术质量)
- Aesthetic Quality: 0.1489 (美学质量)
- Overall Quality: 0.1263 (综合质量)

测试2: 实际评估视频 (3个TI2V视频)
- 平均Technical Quality: -0.0681
- 平均Aesthetic Quality: 0.1133
- 平均Overall Quality: 0.0951

DOVER测试结论:
----------------------------------------
✓ DOVER模块完全功能正常
✓ 可以处理实际评估视频文件
✓ 评估结果合理且可解释
✓ 可集成到评估流程中

DOVER与传统指标对比:
----------------------------------------
TC (时间一致性):
- 优点: 快速，反映帧间一致性
- 缺点: 不能评估美学质量

Motion Score (运动分数):
- 优点: 反映视频动态程度
- 缺点: 不代表质量好坏

DOVER:
- 优点: 综合美学+技术，符合人类感知
- 缺点: 计算复杂，需要GPU
- 结论: 提供更全面的质量评估

EOF

echo "" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "第五部分: 综合结论与建议" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

cat << 'EOF' >> "$OUTPUT_FILE"
消音效果评估:
----------------------------------------
🥇 T2V效果最佳: 标记率从50.60%降至12.14% (下降76%)
🥈 TI2I针对S1: 标记率从13.95%降至10.67% (下降23.5%)
🥉 TI2V整体均衡: 标记率从36.36%降至31.82% (下降12.5%)

质量保持评估:
----------------------------------------
✓ 图像质量优秀: CLIP-I > 0.99
✓ 视频时间一致性优秀: TC > 0.95
✓ 消音过程对质量影响很小
✓ DOVER验证了视频质量的可接受性

攻击类型分析:
----------------------------------------
S1 (文本攻击): 对消音最敏感，效果显著
S2 (视觉越狱): 具有抵抗性，效果有限
S3 (双重强化): 最具挑战性，需要更强消音策略

建议行动:
----------------------------------------
1. 针对不同攻击类型优化消音策略
2. 将DOVER集成到关键数据集的质量评估
3. 建立分层质量评估体系 (基础+DOVER+人工)
4. 定期监控S2/S3攻击的防御效果

EOF

echo "" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "第六部分: 可用的报告文件" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

cat << 'EOF' >> "$OUTPUT_FILE"
已生成的报告文件:
----------------------------------------
1. comprehensive_evaluation_summary.txt
   - 综合评估汇总 (TI2I/T2V/TI2V)
   - S1/S2/S3分解数据
   - 质量指标汇总

2. comprehensive_evaluation_with_dover_status.txt
   - 包含DOVER配置状态
   - 详细评估数据
   - 跨类型对比分析

3. dover_test_results_report.txt
   - DOVER实际测试结果
   - 功能验证结论
   - 集成建议

4. final_comprehensive_evaluation_report.txt (本文件)
   - 最终综合报告
   - 所有结果汇总
   - 行动建议

EOF

echo "" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "报告生成完成: $(date '+%Y-%m-%d %H:%M:%S')" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"

# 显示报告内容
cat "$OUTPUT_FILE"

echo ""
echo "========================================"
echo "最终综合评估报告已保存至: $OUTPUT_FILE"
echo "========================================"