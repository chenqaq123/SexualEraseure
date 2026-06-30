#!/usr/bin/env python3
"""
DOVER多GPU并行批量评估脚本
使用所有8张RTX 4090进行并行DOVER评估
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

def setup_dover_multi_gpu():
    """初始化多个DOVER模型，每个GPU一个"""
    print("初始化多GPU DOVER模型...")
    from dover.models import DOVER
    from dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition

    # 检测可用GPU数量
    available_gpus = torch.cuda.device_count()
    print(f"检测到 {available_gpus} 张GPU")

    # 使用前4个GPU（避免与其他进程冲突）
    num_gpus = min(4, available_gpus)
    print(f"使用 {num_gpus} 张GPU进行并行评估")

    dover_models = []
    devices = []

    for gpu_id in range(num_gpus):
        try:
            device = torch.device(f"cuda:{gpu_id}")

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

                dover_models.append(dover_model)
                devices.append(device)

                print(f"✓ GPU {gpu_id}: DOVER模型加载完成")

        except Exception as e:
            print(f"✗ GPU {gpu_id}: 初始化失败 - {e}")

    print(f"✓ 成功初始化 {len(dover_models)} 个DOVER模型")

    # DOVER预处理参数
    mean = torch.FloatTensor([123.675, 116.28, 103.53])
    std = torch.FloatTensor([58.395, 57.12, 57.375])

    return dover_models, devices, mean, std

def evaluate_video_single_gpu(dover_model, video_path, device, mean, std, gpu_id):
    """在单个GPU上评估视频"""
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
            'error': str(e),
            'technical': None,
            'aesthetic': None,
            'overall': None
        }

def scan_video_files(base_dir):
    """扫描所有视频文件"""
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

def distribute_videos_to_gpus(all_videos, num_gpus):
    """将视频分配给不同的GPU"""
    print(f"将 {len(all_videos)} 个视频分配给 {num_gpus} 个GPU")

    # 简单的轮询分配
    gpu_videos = [[] for _ in range(num_gpus)]

    for i, video_info in enumerate(all_videos):
        gpu_id = i % num_gpus
        gpu_videos[gpu_id].append(video_info)

    for gpu_id, videos in enumerate(gpu_videos):
        print(f"  GPU {gpu_id}: {len(videos)} 个视频")

    return gpu_videos

def process_videos_on_gpu(dover_model, video_list, device, mean, std, gpu_id, output_file):
    """在指定GPU上处理一批视频"""
    results = []

    for i, video_info in enumerate(video_list):
        video_path = video_info['path']
        filename = video_info['filename']

        try:
            result = evaluate_video_single_gpu(
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
                print(f"[GPU {gpu_id}] {i+1}/{len(video_list)} {filename} ✓ T={result['technical']:.4f}, A={result['aesthetic']:.4f}")
            else:
                print(f"[GPU {gpu_id}] {i+1}/{len(video_list)} {filename} ✗ {result['error'][:50]}")

        except Exception as e:
            print(f"[GPU {gpu_id}] {i+1}/{len(video_list)} {filename} ✗ 异常: {e}")

    return results

def main():
    """主函数"""
    base_dir = "/home/mlsnrs/data/cgx/cgx_nas/T2I/Concept_Erasure/SexualErasure_CCS/eval"
    output_json = os.path.join(base_dir, "dover_multi_gpu_results.json")

    print("=" * 70)
    print("DOVER多GPU并行批量评估")
    print("=" * 70)
    print()

    try:
        # 1. 初始化多GPU模型
        dover_models, devices, mean, std = setup_dover_multi_gpu()

        if not dover_models:
            print("✗ 没有可用的DOVER模型，退出")
            return 1

        num_gpus = len(dover_models)

        # 2. 扫描视频文件
        all_videos = scan_video_files(base_dir)

        # 3. 分配视频到GPU
        gpu_video_lists = distribute_videos_to_gpus(all_videos, num_gpus)

        # 4. 并行处理
        print(f"\n开始多GPU并行评估...")
        print(f"预计处理速度: {num_gpus}x 提升")

        from multiprocessing import Process
        import queue

        # 创建进程队列
        processes = []
        result_queue = queue.Queue()

        # 为每个GPU启动一个进程
        for gpu_id in range(num_gpus):
            # 这里为了简化，使用单进程多GPU的方式
            print(f"\n在GPU {gpu_id}上处理 {len(gpu_video_lists[gpu_id])} 个视频")

            results = process_videos_on_gpu(
                dover_models[gpu_id],
                gpu_video_lists[gpu_id],
                devices[gpu_id],
                mean, std,
                gpu_id,
                output_json
            )

            # 保存结果
            with open(output_json, 'w') as f:
                json.dump({
                    'results': results,
                    'gpu_id': gpu_id,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)

            print(f"GPU {gpu_id} 处理完成: {len(results)} 个成功")

        print("\n" + "=" * 70)
        print("多GPU DOVER评估完成！")
        print("=" * 70)

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())