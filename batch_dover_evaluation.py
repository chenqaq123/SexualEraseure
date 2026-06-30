#!/usr/bin/env python3
"""
DOVER批量视频质量评估脚本
按照擦除前后、任务类型、S1/S2/S3子集进行分类统计
"""

import torch
import os
import sys
import json
import csv
from datetime import datetime
from collections import defaultdict
import glob

# 添加DOVER到路径
sys.path.insert(0, '/tmp/DOVER')

def setup_dover():
    """初始化DOVER模型"""
    print("初始化DOVER模型...")
    from dover.models import DOVER
    from dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # DOVER配置
    dover_config = {
        "backbone_size": "divided",
        "backbone_preserve_keys": "technical,aesthetic",
        "divide_head": True,
        "vqa_head": {"in_channels": 768, "hidden_channels": 64},
        "backbone": {
            "technical": {"type": "swin_tiny_grpb", "checkpoint": True, "pretrained": True},
            "aesthetic": {"type": "conv_tiny"}
        }
    }

    dover_model = DOVER(**dover_config)

    # 加载权重
    weights_path = "/tmp/DOVER/pretrained_weights/DOVER.pth"
    print(f"加载权重: {weights_path}")
    state_dict = torch.load(weights_path, map_location=device)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    dover_model.load_state_dict(state_dict, strict=False)
    dover_model.to(device)
    dover_model.eval()

    # DOVER预处理参数
    mean = torch.FloatTensor([123.675, 116.28, 103.53])
    std = torch.FloatTensor([58.395, 57.12, 57.375])

    # 配置采样参数
    dopt = {
        "technical": {
            "fragments_h": 7, "fragments_w": 7, "fsize_h": 32, "fsize_w": 32,
            "aligned": 32, "clip_len": 32, "frame_interval": 2, "num_clips": 1
        },
        "aesthetic": {
            "size_h": 224, "size_w": 224, "clip_len": 32, "frame_interval": 2, "t_frag": 32, "num_clips": 1
        }
    }

    temporal_samplers = {}
    for stype, sopt in dopt.items():
        if "t_frag" not in sopt:
            temporal_samplers[stype] = UnifiedFrameSampler(
                sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"]
            )
        else:
            temporal_samplers[stype] = UnifiedFrameSampler(
                sopt["clip_len"] // sopt["t_frag"],
                sopt["t_frag"], sopt["frame_interval"], sopt["num_clips"]
            )

    print("✓ DOVER模型初始化完成")
    return dover_model, device, mean, std, dopt, temporal_samplers

def evaluate_single_video(dover_model, video_path, device, mean, std, dopt, temporal_samplers):
    """评估单个视频的DOVER质量"""
    try:
        from dover.datasets import spatial_temporal_view_decomposition

        # View decomposition
        views, _ = spatial_temporal_view_decomposition(video_path, dopt, temporal_samplers)

        # 预处理
        for k, v in views.items():
            num_clips = dopt[k].get("num_clips", 1)
            views[k] = (
                ((v.permute(1, 2, 3, 0) - mean) / std)
                .permute(3, 0, 1, 2)
                .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
                .transpose(0, 1)
                .to(device)
            )

        # 运行DOVER评估
        with torch.no_grad():
            results = [r.mean().item() for r in dover_model(views)]

        return {
            'technical': results[0],
            'aesthetic': results[1],
            'overall': 0.1 * results[0] + 0.9 * results[1],
            'success': True
        }

    except Exception as e:
        return {
            'technical': None,
            'aesthetic': None,
            'overall': None,
            'success': False,
            'error': str(e)
        }

def get_attack_subset_from_filename(filename):
    """从文件名推断攻击子集"""
    # 查找对应的CSV文件来获取attack_subset信息
    base_dir = os.path.dirname(filename)
    csv_files = glob.glob(os.path.join(os.path.dirname(base_dir), "nudenet_eval.csv"))
    csv_files.extend(glob.glob(os.path.join(base_dir, "nudenet_eval.csv")))

    for csv_file in csv_files:
        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['filename'] == os.path.basename(filename):
                        return row['attack_subset']
        except:
            continue

    # 如果找不到CSV，尝试从文件名推断
    if '_sid' in filename:
        return 'unknown'
    return 'unknown'

def scan_video_files(base_dir):
    """扫描所有视频文件并分类"""
    print("扫描视频文件...")

    video_categories = {
        'TI2I': {'original': [], 'erased': []},
        'T2V': {'original': [], 'erased': []},
        'TI2V': {'original': [], 'erased': []}
    }

    # 定义目录映射
    dir_mappings = {
        'TI2I': ['TI2I_flux/results', 'TI2I_flux/erased_results'],
        'T2V': ['T2V_hunyuanvideo/results', 'T2V_hunyuanvideo/erased_results'],
        'TI2V': ['TI2V_hunyuanvideo/results', 'TI2V_hunyuanvideo/erased_results']
    }

    total_videos = 0
    for task_type, dir_list in dir_mappings.items():
        for i, dir_name in enumerate(dir_list):
            result_type = 'original' if i == 0 else 'erased'
            full_path = os.path.join(base_dir, dir_name)

            if os.path.exists(full_path):
                mp4_files = glob.glob(os.path.join(full_path, "*.mp4"))
                for video_file in mp4_files:
                    # 读取CSV获取attack_subset信息
                    attack_subset = get_attack_subset_from_filename(video_file)
                    video_categories[task_type][result_type].append({
                        'path': video_file,
                        'attack_subset': attack_subset,
                        'filename': os.path.basename(video_file)
                    })
                    total_videos += 1

    print(f"✓ 找到 {total_videos} 个视频文件")

    # 打印统计信息
    for task_type in ['TI2I', 'T2V', 'TI2V']:
        print(f"{task_type}:")
        print(f"  - 原始: {len(video_categories[task_type]['original'])} 个视频")
        print(f"  - 消除: {len(video_categories[task_type]['erased'])} 个视频")

    return video_categories

def batch_evaluate_videos(video_categories, dover_model, device, mean, std, dopt, temporal_samplers, output_file):
    """批量评估视频并分类统计"""

    # 初始化结果存储
    results = {
        'TI2I': {'original': defaultdict(list), 'erased': defaultdict(list)},
        'T2V': {'original': defaultdict(list), 'erased': defaultdict(list)},
        'TI2V': {'original': defaultdict(list), 'erased': defaultdict(list)}
    }

    # 临时文件用于保存进度
    progress_file = output_file.replace('.json', '_progress.json')
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress_data = json.load(f)
            processed_files = set(progress_data.get('processed_files', []))
        print(f"从进度文件恢复，已处理 {len(processed_files)} 个文件")
    else:
        processed_files = set()

    total_count = 0
    success_count = 0
    fail_count = 0

    start_time = datetime.now()

    # 遍历所有类别
    for task_type in ['TI2I', 'T2V', 'TI2V']:
        for result_type in ['original', 'erased']:
            video_list = video_categories[task_type][result_type]

            print(f"\n处理 {task_type} - {result_type} ({len(video_list)} 个视频)")

            for i, video_info in enumerate(video_list):
                video_path = video_info['path']
                attack_subset = video_info['attack_subset']
                filename = video_info['filename']

                # 跳过已处理的文件
                if video_path in processed_files:
                    total_count += 1
                    success_count += 1
                    continue

                try:
                    print(f"  [{total_count+1}] {filename} ({attack_subset})", end=" ")

                    # 评估视频
                    dover_result = evaluate_single_video(
                        dover_model, video_path, device, mean, std, dopt, temporal_samplers
                    )

                    if dover_result['success']:
                        results[task_type][result_type][attack_subset].append({
                            'filename': filename,
                            'technical': dover_result['technical'],
                            'aesthetic': dover_result['aesthetic'],
                            'overall': dover_result['overall']
                        })
                        print(f"✓ T={dover_result['technical']:.4f}, A={dover_result['aesthetic']:.4f}, O={dover_result['overall']:.4f}")
                        success_count += 1
                    else:
                        print(f"✗ 失败: {dover_result.get('error', 'Unknown error')}")
                        fail_count += 1

                    # 保存进度
                    processed_files.add(video_path)
                    if total_count % 10 == 0:
                        progress_data = {
                            'processed_files': list(processed_files),
                            'total_count': total_count,
                            'success_count': success_count,
                            'fail_count': fail_count
                        }
                        with open(progress_file, 'w') as f:
                            json.dump(progress_data, f)

                    total_count += 1

                except Exception as e:
                    print(f"✗ 异常: {e}")
                    fail_count += 1
                    total_count += 1

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"\n评估完成！")
    print(f"总计: {total_count} 个视频")
    print(f"成功: {success_count} 个")
    print(f"失败: {fail_count} 个")
    print(f"耗时: {duration/60:.1f} 分钟")

    # 保存最终结果
    with open(output_file, 'w') as f:
        json.dump({
            'results': results,
            'summary': {
                'total_count': total_count,
                'success_count': success_count,
                'fail_count': fail_count,
                'duration_seconds': duration,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat()
            }
        }, f, indent=2)

    print(f"结果已保存至: {output_file}")

    # 删除进度文件
    if os.path.exists(progress_file):
        os.remove(progress_file)

    return results

def generate_statistics_report(results, output_txt):
    """生成统计报告"""

    with open(output_txt, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("DOVER批量视频质量评估报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        # 对每个任务类型生成报告
        for task_type in ['TI2I', 'T2V', 'TI2V']:
            f.write("=" * 70 + "\n")
            f.write(f"{task_type} (Text-to-{'Image' if task_type == 'TI2I' else 'Video'}")
            if task_type == 'TI2V':
                f.write(" with image condition")
            f.write(")\n")
            f.write("=" * 70 + "\n\n")

            for result_type in ['original', 'erased']:
                result_type_name = "原始模型 (未处理)" if result_type == 'original' else "消除后模型 (处理后)"
                f.write("-" * 70 + "\n")
                f.write(f"{result_type_name}\n")
                f.write("-" * 70 + "\n\n")

                subset_data = results[task_type][result_type]

                if not subset_data:
                    f.write("无数据\n\n")
                    continue

                # 按S1/S2/S3统计
                for subset in ['S1', 'S2', 'S3']:
                    if subset not in subset_data or not subset_data[subset]:
                        continue

                    data = subset_data[subset]
                    technical_scores = [d['technical'] for d in data]
                    aesthetic_scores = [d['aesthetic'] for d in data]
                    overall_scores = [d['overall'] for d in data]

                    f.write(f"{subset} 子集 ({len(data)} 个视频):\n")
                    f.write(f"  Technical Quality:  平均={sum(technical_scores)/len(technical_scores):.4f}\n")
                    f.write(f"  Aesthetic Quality:  平均={sum(aesthetic_scores)/len(aesthetic_scores):.4f}\n")
                    f.write(f"  Overall Quality:    平均={sum(overall_scores)/len(overall_scores):.4f}\n")
                    f.write("\n")

                # 总体统计
                all_data = []
                for subset_data in subset_data.values():
                    all_data.extend(subset_data)

                if all_data:
                    all_technical = [d['technical'] for d in all_data]
                    all_aesthetic = [d['aesthetic'] for d in all_data]
                    all_overall = [d['overall'] for d in all_data]

                    f.write(f"总体统计 ({len(all_data)} 个视频):\n")
                    f.write(f"  Technical Quality:  平均={sum(all_technical)/len(all_technical):.4f}\n")
                    f.write(f"  Aesthetic Quality:  平均={sum(all_aesthetic)/len(all_aesthetic):.4f}\n")
                    f.write(f"  Overall Quality:    平均={sum(all_overall)/len(all_overall):.4f}\n")
                    f.write("\n")

    print(f"统计报告已保存至: {output_txt}")

def main():
    """主函数"""
    base_dir = "/home/mlsnrs/data/cgx/cgx_nas/T2I/Concept_Erasure/SexualErasure_CCS/eval"
    output_json = os.path.join(base_dir, "dover_batch_results.json")
    output_txt = os.path.join(base_dir, "dover_batch_statistics_report.txt")

    print("=" * 70)
    print("DOVER批量视频质量评估")
    print("=" * 70)
    print()

    try:
        # 1. 初始化DOVER
        dover_model, device, mean, std, dopt, temporal_samplers = setup_dover()

        # 2. 扫描视频文件
        video_categories = scan_video_files(base_dir)

        # 3. 批量评估
        print(f"\n开始批量评估...")
        print(f"结果将保存至: {output_json}")
        print(f"预计需要 2-3 小时完成所有 {sum(len(v['erased']) for v in video_categories.values())} 个消除后视频的评估")

        results = batch_evaluate_videos(
            video_categories, dover_model, device, mean, std, dopt, temporal_samplers, output_json
        )

        # 4. 生成统计报告
        print(f"\n生成统计报告...")
        generate_statistics_report(results, output_txt)

        print("\n" + "=" * 70)
        print("DOVER批量评估完成！")
        print("=" * 70)

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())