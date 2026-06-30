#!/bin/bash
# 生成包含S1/S2/S3细分类别质量指标的详细报告

echo "========================================"
echo "UniErase-Bench 质量指标S1/S2/S3细分报告"
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
SUMMARY_FILE="eval_quality_by_subset_report.txt"
echo "UniErase-Bench 质量指标S1/S2/S3细分报告" > "$SUMMARY_FILE"
echo "生成时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "========================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# 函数：计算并提取S1/S2/S3质量指标
extract_quality_by_subset() {
    local CSV_FILE=$1
    local DIR_TYPE=$2  # "image" or "video"

    if [ ! -f "$CSV_FILE" ]; then
        return
    fi

    # 跳过表头，按attack_subset分组计算质量指标
    if [[ "$DIR_TYPE" == "image" ]]; then
        # 图像任务：提取clip_score和clip_i
        echo "" >> "$SUMMARY_FILE"
        echo "┌─ 质量指标按S1/S2/S3细分 (图像) ──────────┐" >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"
        printf "%-10s %15s %15s %15s\n" "子集" "案例数" "CLIP Score" "CLIP-I" >> "$SUMMARY_FILE"
        echo "---------- --------------- --------------- ---------------" >> "$SUMMARY_FILE"

        for subset in "S1" "S2" "S3"; do
            # 统计该子集的案例数和质量指标
            local count=$(tail -n +2 "$CSV_FILE" | awk -F',' -v subset="$subset" '$4==subset {print $0}' | wc -l)
            if [ $count -eq 0 ]; then
                continue
            fi

            # 计算CLIP Score平均值
            local clip_score=$(tail -n +2 "$CSV_FILE" | awk -F',' -v subset="$subset" '$4==subset && $10!="" {sum+=$10; count++} END {if(count>0) printf "%.4f", sum/count; else print "--"}')

            # 计算CLIP-I平均值
            local clip_i=$(tail -n +2 "$CSV_FILE" | awk -F',' -v subset="$subset" '$4==subset && $11!="" {sum+=$11; count++} END {if(count>0) printf "%.4f", sum/count; else print "--"}')

            printf "%-10s %15d %15s %15s\n" "$subset" "$count" "$clip_score" "$clip_i" >> "$SUMMARY_FILE"
        done

    else
        # 视频任务：提取tc和motion_score
        echo "" >> "$SUMMARY_FILE"
        echo "┌─ 质量指标按S1/S2/S3细分 (视频) ──────────┐" >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"
        printf "%-10s %15s %15s %15s\n" "子集" "案例数" "TC (时间一致性)" "Motion Score" >> "$SUMMARY_FILE"
        echo "---------- --------------- --------------- ---------------" >> "$SUMMARY_FILE"

        for subset in "S1" "S2" "S3"; do
            # 统计该子集的案例数和质量指标
            local count=$(tail -n +2 "$CSV_FILE" | awk -F',' -v subset="$subset" '$4==subset {print $0}' | wc -l)
            if [ $count -eq 0 ]; then
                continue
            fi

            # 计算TC平均值
            local tc=$(tail -n +2 "$CSV_FILE" | awk -F',' -v subset="$subset" '$4==subset && $12!="" {sum+=$12; count++} END {if(count>0) printf "%.4f", sum/count; else print "--"}')

            # 计算Motion Score平均值
            local motion=$(tail -n +2 "$CSV_FILE" | awk -F',' -v subset="$subset" '$4==subset && $13!="" {sum+=$13; count++} END {if(count>0) printf "%.4f", sum/count; else print "--"}')

            printf "%-10s %15d %15s %15s\n" "$subset" "$count" "$tc" "$motion" >> "$SUMMARY_FILE"
        done
    fi

    echo "" >> "$SUMMARY_FILE"
    echo "└────────────────────────────────────────────┘" >> "$SUMMARY_FILE"
}

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

    # 判断任务类型并提取质量指标
    if [[ "$DIR" == *"TI2I"* ]]; then
        extract_quality_by_subset "$DIR/nudenet_eval.csv" "image"
    elif [[ "$DIR" == *"T2V"* ]] || [[ "$DIR" == *"TI2V"* ]]; then
        extract_quality_by_subset "$DIR/nudenet_eval.csv" "video"
    fi

    echo "" | tee -a "$SUMMARY_FILE"
    echo "----------------------------------------" | tee -a "$SUMMARY_FILE"
    echo "" | tee -a "$SUMMARY_FILE"
done

echo "" | tee -a "$SUMMARY_FILE"
echo "========================================" | tee -a "$SUMMARY_FILE"
echo "汇总报告已生成: $SUMMARY_FILE" | tee -a "$SUMMARY_FILE"
echo "========================================" | tee -a "$SUMMARY_FILE"

# 显示汇总报告内容
cat "$SUMMARY_FILE"

echo ""
