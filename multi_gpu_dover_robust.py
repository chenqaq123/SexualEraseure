#!/usr/bin/env python3
"""
DOVER多GPU并行评估脚本 - 稳健版本
每个GPU独立处理，避免tensor维度匹配问题
"""

import torch
import os
import sys
import json
import csv
from datetime import datetime
from collections import defaultdict
import glob
import traceback

# 添加DOVER到路径
sys.path.insert(0, '/tmp/DOVER')

def setup_dover_on_gpu(gpu_id):
    """在指定GPU上初始化DOVER模型"""
    try:
        from dover.models import DOVER
        from dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition

        device = torch.device(f"cuda:{gpu_id}")

        print(f"[GPU {gpu_id}] 初始化DOVER模型...")

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

        with torch.cuda.device(device):
            dover_model = DOVER(**dover_config)

            # 加载权重
            weights_path = "/tmp/DOVER/pretrained_weights/DOVER.pth"
            state_dict = torch.load(weights_path, map_location=device)
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            dover_model.load_state_dict(state_dict, strict=False)
            dover_model.to(device)
            dover_model.eval()

        print(f"[GPU {gpu_id}] ✓ DOVER模型初始化完成")
        return dover_model, device

    except Exception as e:
        print(f"[GPU {gpu_id}] ✗ 初始化失败: {e}")
        traceback.print_exc()
        return None, None

def evaluate_video_on_gpu(dover_model, video_path, device, mean, std, gpu_id):
    """在指定GPU上评估单个视频"""
    try:
        from dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition

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
            'success': True,
            'gpu_id': gpu_id,
            'technical': results[0],
            'aesthetic': results[1],
            'overall': 0.1 * results[0] + 0.9 * results[1]
        }

    except Exception as e:
        return {
            'success': False,
            'gpu_id': gpu_id,
            'error': str(e)
        }

def process_video_list(gpu_id, video_list, dover_model, device, mean, std, output_dir):
    """在指定GPU上处理视频列表"""
    results = []
    success_count = 0
    fail_count = 0

    for i, video_info in enumerate(video_list):
        video_path = video_info['path']
        filename = video_info['filename']

        try:
            result = evaluate_video_on_gpu(
                dover_model, video_path, device, mean, std, gpu_id
            )

            if result['success']:
                results.append({
                    'filename': filename,
                    'task_type': video_info['task_type'],
                    'result_type': video_info['result_type'],
                    'attack_subset': video_info['attack_subset'],
                    'technical': result['technical'],
                    'aesthetic': result['aesthetic'],
                    'overall': result['overall'],
                    'gpu_id': gpu_id
                })
                success_count += 1
                print(f"[GPU {gpu_id}] {i+1}/{len(video_list)} {filename} ✓ T={result['technical']:.4f}, A={result['aesthetic']:.4f}")
            else:
                fail_count += 1
                print(f"[GPU {gpu_id}] {i+1}/{len(video_list)} {filename} ✗")

        except Exception as e:
            fail_count += 1
            print(f"[GPU {gpu_id}] {i+1}/{len(video_list)} {filename} ✗ 异常: {e}")

        # 定期保存进度
        if (i + 1) % 10 == 0:
            temp_output = os.path.join(output_dir, f"gpu_{gpu_id}_temp.json")
            with open(temp_output, 'w') as f:
                json.dump({
                    'results': results,
                    'success_count': success_count,
                    'fail_count': fail_count,
                    'processed': i + 1,
                    'total': len(video_list)
                }, f, indent=2)

    print(f"[GPU {gpu_id}] 处理完成: {len(results)} 成功, {fail_count} 失败")

    # 保存最终结果
    gpu_output = os.path.join(output_dir, f"gpu_{gpu_id}_results.json")
    with open(gpu_output, 'w') as f:
        json.dump({
            'results': results,
            'success_count': success_count,
            'fail_count': fail_count,
            'gpu_id': gpu_id
        }, f, indent=2)

    return results

def scan_all_videos(base_dir):
    """扫描所有需要处理的视频文件"""
    print("扫描视频文件...")

    all_videos = []

    dir_mappings = {
        'T2V': ['T2V_hunyuanvideo/results', 'T2V_hunyuanvideo/erased_results'],
        'TI2V': ['TI2V_hunyuanvideo/results', 'TI2V_hunyuanvideo/erased_results']
    }

    for task_type, dir_list in dir_mappings.items():
        for i, dir_name in enumerate(dir_list):
            result_type = 'original' if i == 0 else 'erased'
            full_path = os.path.join(base_dir, dir_name)

            if os.path.exists(full_path):
                # 读取CSV获取attack_subset信息
                csv_file = os.path.join(full_path, "nudenet_eval.csv")
                subset_mapping = {}

                if os.path.exists(csv_file):
                    with open(csv_file, 'r') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            subset_mapping[row['filename']] = row['attack_subset']

                # 扫描视频文件
                mp4_files = glob.glob(os.path.join(full_path, "*.mp4"))
                for video_file in mp4_files:
                    filename = os.path.basename(video_file)
                    attack_subset = subset_mapping.get(filename, 'unknown')

                    all_videos.append({
                        'path': video_file,
                        'task_type': task_type,
                        'result_type': result_type,
                        'attack_subset': attack_subset,
                        'filename': filename
                    })

    print(f"✓ 找到 {len(all_videos)} 个视频文件")
    return all_videos

def distribute_videos_to_gpus(all_videos, num_gpus=8):
    """将视频均匀分配给所有GPU"""
    print(f"将 {len(all_videos)} 个视频分配给 {num_gpus} 个GPU")

    gpu_videos = [[] for _ in range(num_gpus)]

    for i, video_info in enumerate(all_videos):
        gpu_id = i % num_gpus
        gpu_videos[gpu_id].append(video_info)

    for gpu_id, videos in enumerate(gpu_videos):
        print(f"  GPU {gpu_id}: {len(videos)} 个视频")

    return gpu_videos

def merge_all_gpu_results(output_dir):
    """合并所有GPU的结果"""
    print("合并所有GPU的结果...")

    all_results = {
        'T2V': {'original': defaultdict(list), 'erased': defaultdict(list)},
        'TI2V': {'original': defaultdict(list), 'erased': defaultdict(list)}
    }

    total_success = 0

    # 读取每个GPU的结果文件
    for gpu_id in range(8):
        gpu_file = os.path.join(output_dir, f"gpu_{gpu_id}_results.json")
        if os.path.exists(gpu_file):
            with open(gpu_file, 'r') as f:
                gpu_data = json.load(f)
                gpu_results = gpu_data['results']

                for result in gpu_results:
                    task_type = result['task_type']
                    result_type = result['result_type']
                    attack_subset = result['attack_subset']

                    all_results[task_type][result_type][attack_subset].append({
                        'filename': result['filename'],
                        'technical': result['technical'],
                        'aesthetic': result['aesthetic'],
                        'overall': result['overall']
                    })
                    total_success += 1

    print(f"✓ 合并完成，总计 {total_success} 个成功结果")

    # 保存合并后的结果
    merged_output = os.path.join(output_dir, "multi_gpu_merged_results.json")
    with open(merged_output, 'w') as f:
        json.dump(all_results, f, indent=2)

    return all_results

def main():
    """主函数"""
    base_dir = "/home/mlsnrs/data/cgx/cgx_nas/T2I/Concept_Erasure/SexualErasure_CCS/eval"
    output_dir = os.path.join(base_dir, "multi_gpu_results")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("DOVER多GPU并行评估 - 稳健版本")
    print("使用8张RTX 4090并行处理")
    print("=" * 70)
    print()

    try:
        # 1. 初始化所有GPU的DOVER模型
        print("步骤1: 初始化8个GPU的DOVER模型")
        print("-" * 70)

        dover_models = []
        devices = []
        num_gpus = 8

        for gpu_id in range(num_gpus):
            model, device = setup_dover_on_gpu(gpu_id)
            if model is not None:
                dover_models.append(model)
                devices.append(device)

        if not dover_models:
            print("✗ 没有成功初始化任何GPU模型")
            return 1

        print(f"✓ 成功初始化 {len(dover_models)} 个GPU模型")

        # 2. 扫描所有视频文件
        print()
        print("步骤2: 扫描视频文件")
        print("-" * 70)
        all_videos = scan_all_videos(base_dir)

        # 3. 分配视频到GPU
        print()
        print("步骤3: 分配视频到各个GPU")
        print("-" * 70)
        gpu_video_lists = distribute_videos_to_gpus(all_videos, len(dover_models))

        # 4. 并行处理
        print()
        print("步骤4: 启动多GPU并行处理")
        print("-" * 70)
        print(f"预计处理速度: {len(dover_models)}x 提升")
        print(f"预计完成时间: 30-45分钟")
        print()

        # 使用多进程并行处理
        from multiprocessing import Pool, Manager

        def process_gpu(gpu_id):
            """处理单个GPU的所有视频"""
            if gpu_id >= len(dover_models):
                return []

            print(f"\n[GPU {gpu_id}] 开始处理...")

            try:
                results = process_video_list(
                    gpu_id,
                    gpu_video_lists[gpu_id],
                    dover_models[gpu_id],
                    devices[gpu_id],
                    torch.FloatTensor([123.675, 116.28, 103.53]),
                    torch.FloatTensor([58.395, 57.12, 57.375]),
                    gpu_id,
                    output_dir
                )

                print(f"[GPU {gpu_id}] 完成: {len(results)} 个成功结果")
                return results

            except Exception as e:
                print(f"[GPU {gpu_id}] 处理失败: {e}")
                traceback.print_exc()
                return []

        # 启动多进程处理
        with Pool(processes=len(dover_models)) as pool:
            all_results_list = pool.map(process_gpu, range(len(dover_models)))

        # 5. 合并结果
        print()
        print("步骤5: 合并所有GPU结果")
        print("-" * 70)
        final_results = merge_all_gpu_results(output_dir)

        # 6. 生成统计报告
        print()
        print("步骤6: 生成统计报告")
        print("-" * 70)

        generate_final_report(final_results, output_dir)

        print()
        print("=" * 70)
        print("多GPU DOVER评估完成！")
        print("=" * 70)

    except Exception as e:
        print(f"错误: {e}")
        traceback.print_exc()
        return 1

    return 0

def generate_final_report(results, output_dir):
    """生成最终统计报告"""

    output_txt = os.path.join(output_dir, "multi_gpu_final_report.txt")

    with open(output_txt, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("DOVER多GPU并行评估 - 最终报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        # 对每个任务类型生成报告
        for task_type in ['T2V', 'TI2V']:
            f.write("=" * 70 + "\n")
            f.write(f"{task_type} (Text-to-Video")
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

                    import statistics
                    n = len(data)

                    f.write(f"{subset} 子集 ({n} 个视频):\n")
                    if n > 0:
                        f.write(f"  Technical Quality:  平均={sum(technical_scores)/n:.4f}, 标准差={statistics.stdev(technical_scores):.4f}\n")
                        f.write(f"  Aesthetic Quality:  平均={sum(aesthetic_scores)/n:.4f}, 标准差={statistics.stdev(aesthetic_scores):.4f}\n")
                        f.write(f"  Overall Quality:    平均={sum(overall_scores)/n:.4f}, 标准差={statistics.stdev(overall_scores):.4f}\n")
                    f.write("\n")

                # 总体统计
                all_data = []
                for subset_data in subset_data.values():
                    all_data.extend(subset_data)

                if all_data:
                    all_technical = [d['technical'] for d in all_data]
                    all_aesthetic = [d['aesthetic'] for d in all_data]
                    all_overall = [d['overall'] for d in all_data]

                    import statistics
                    n = len(all_data)

                    f.write(f"总体统计 ({n} 个视频):\n")
                    f.write(f"  Technical Quality:  平均={sum(all_technical)/n:.4f}, 标准差={statistics.stdev(all_technical):.4f}\n")
                    f.write(f"  Aesthetic Quality:  平均={sum(all_aesthetic)/n:.4f}, 标准差={statistics.stdev(all_aesthetic):.4f}\n")
                    f.write(f"  Overall Quality:    平均={sum(all_overall)/n:.4f}, 标准差={statistics.stdev(all_overall):.4f}\n")
                    f.write("\n")

    print(f"✓ 统计报告已保存至: {output_txt}")

if __name__ == "__main__":
    sys.exit(main())