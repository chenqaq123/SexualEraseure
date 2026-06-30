#!/bin/bash
# 生成所有评估结果的汇总报告

echo "========================================"
echo "UniErase-Bench 评估结果汇总报告"
echo "生成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""

cd /home/mlsnrs/data/cgx/cgx_nas/T2I/Concept_Erasure/SexualErasure_CCS/eval

# 定义所有需要汇总的目录
declare -a DIRS=(
    "TI2I_flux/results"
    "TI2I_flux/erased_results"
    "T2V_hunyuanvideo/results"
    "T2V_hunyuanvideo/erased_results"
    "TI2V_hunyuanvideo/results"
    "TI2V_hunyuanvideo/erased_results"
)

# 创建汇总文件
SUMMARY_FILE="eval_summary_report.txt"
echo "UniErase-Bench 评估结果汇总报告" > "$SUMMARY_FILE"
echo "生成时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "========================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# 遍历每个目录
for DIR in "${DIRS[@]}"; do
    echo "📁 处理目录: $DIR"

    if [ ! -d "$DIR" ]; then
        echo "❌ 目录不存在: $DIR" | tee -a "$SUMMARY_FILE"
        echo "" | tee -a "$SUMMARY_FILE"
        continue
    fi

    echo "目录: $DIR" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"

    # 检查是否有评估报告
    if [ ! -f "$DIR/nudenet_eval.csv" ]; then
        echo "状态: ❌ 评估未完成" | tee -a "$SUMMARY_FILE"
        echo "结果文件数: $(find "$DIR" -maxdepth 1 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.mp4" \) | wc -l)" | tee -a "$SUMMARY_FILE"
        echo "" | tee -a "$SUMMARY_FILE"
        continue
    fi

    echo "状态: ✅ 评估已完成" | tee -a "$SUMMARY_FILE"

    # 提取评估结果信息
    CASES=$(tail -n +2 "$DIR/nudenet_eval.csv" | wc -l)
    echo "评估案例数: $CASES" | tee -a "$SUMMARY_FILE"

    # 提取关键统计信息
    FLAGGED=$(tail -n +2 "$DIR/nudenet_eval.csv" | awk -F',' '$8=="True" {count++} END {print count+0}')
    echo "被标记的案例数: $FLAGGED" | tee -a "$SUMMARY_FILE"

    if [ $CASES -gt 0 ]; then
        FLAGGED_PCT=$(awk "BEGIN {printf \"%.1f\", $FLAGGED/$CASES*100}")
        echo "标记率: ${FLAGGED_PCT}%" | tee -a "$SUMMARY_FILE"
    fi

    # 提取质量指标（如果有）
    # 判断是图像还是视频任务
    if [[ "$DIR" == *"TI2I"* ]]; then
        # 图像质量指标 (TI2I)
        AVG_CLIP_SCORE=$(tail -n +2 "$DIR/nudenet_eval.csv" | awk -F',' '$10!="" {sum+=$10; count++} END {if(count>0) printf "%.4f", sum/count; else print "N/A"}')
        if [ "$AVG_CLIP_SCORE" != "N/A" ]; then
            echo "平均CLIP Score: $AVG_CLIP_SCORE" | tee -a "$SUMMARY_FILE"
        fi

        AVG_CLIP_I=$(tail -n +2 "$DIR/nudenet_eval.csv" | awk -F',' '$11!="" {sum+=$11; count++} END {if(count>0) printf "%.4f", sum/count; else print "N/A"}')
        if [ "$AVG_CLIP_I" != "N/A" ]; then
            echo "平均CLIP-I: $AVG_CLIP_I" | tee -a "$SUMMARY_FILE"
        fi
    elif [[ "$DIR" == *"T2V"* ]] || [[ "$DIR" == *"TI2V"* ]]; then
        # 视频质量指标 (T2V/TI2V)
        AVG_TC=$(tail -n +2 "$DIR/nudenet_eval.csv" | awk -F',' '$12!="" {sum+=$12; count++} END {if(count>0) printf "%.4f", sum/count; else print "N/A"}')
        if [ "$AVG_TC" != "N/A" ]; then
            echo "平均TC (时间一致性): $AVG_TC" | tee -a "$SUMMARY_FILE"
        fi

        AVG_MOTION=$(tail -n +2 "$DIR/nudenet_eval.csv" | awk -F',' '$13!="" {sum+=$13; count++} END {if(count>0) printf "%.4f", sum/count; else print "N/A"}')
        if [ "$AVG_MOTION" != "N/A" ]; then
            echo "平均Motion Score: $AVG_MOTION" | tee -a "$SUMMARY_FILE"
        fi
    fi

    # 文件信息
    SIZE=$(du -h "$DIR/nudenet_eval.csv" | awk '{print $1}')
    MTIME=$(stat -c %y "$DIR/nudenet_eval.csv" | cut -d'.' -f1)
    echo "报告文件大小: $SIZE" | tee -a "$SUMMARY_FILE"
    echo "评估完成时间: $MTIME" | tee -a "$SUMMARY_FILE"

    echo "" | tee -a "$SUMMARY_FILE"
done

echo "========================================" >> "$SUMMARY_FILE"
echo "详细数据位置" >> "$SUMMARY_FILE"
echo "========================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

for DIR in "${DIRS[@]}"; do
    if [ -f "$DIR/nudenet_eval.csv" ]; then
        echo "$DIR: nudenet_eval.csv" >> "$SUMMARY_FILE"
        echo "$DIR: evaluation_log.txt" >> "$SUMMARY_FILE"
    fi
done

echo "" >> "$SUMMARY_FILE"
echo "========================================" >> "$SUMMARY_FILE"
echo "评估完成情况统计" >> "$SUMMARY_FILE"
echo "========================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

TOTAL_DIRS=0
COMPLETED_DIRS=0
for DIR in "${DIRS[@]}"; do
    TOTAL_DIRS=$((TOTAL_DIRS + 1))
    if [ -f "$DIR/nudenet_eval.csv" ]; then
        COMPLETED_DIRS=$((COMPLETED_DIRS + 1))
    fi
done

echo "总目录数: $TOTAL_DIRS" >> "$SUMMARY_FILE"
echo "已完成评估: $COMPLETED_DIRS" >> "$SUMMARY_FILE"
echo "完成率: $(awk "BEGIN {printf \"%.1f\", $COMPLETED_DIRS/$TOTAL_DIRS*100}")%" >> "$SUMMARY_FILE"

echo "" | tee -a "$SUMMARY_FILE"
echo "========================================" | tee -a "$SUMMARY_FILE"
echo "汇总报告已生成: $SUMMARY_FILE" | tee -a "$SUMMARY_FILE"
echo "========================================" | tee -a "$SUMMARY_FILE"

# 显示汇总报告内容
cat "$SUMMARY_FILE"

echo ""
echo "========================================"
echo "CSV文件详细位置"
echo "========================================"
for DIR in "${DIRS[@]}"; do
    if [ -f "$DIR/nudenet_eval.csv" ]; then
        REAL_PATH=$(realpath "$DIR/nudenet_eval.csv")
        echo "✅ $REAL_PATH"
    else
        echo "❌ $DIR/nudenet_eval.csv (不存在)"
    fi
done

echo ""
