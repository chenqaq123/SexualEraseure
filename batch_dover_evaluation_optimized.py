#!/usr/bin/env python3
"""
DOVER批量视频质量评估脚本 - 优化版
支持批量处理以提高GPU利用率
"""

import torch
import os
import sys
import json
import csv
from datetime import datetime
from collections import defaultdict
import glob
from torch.utils.data import Dataset, DataLoader

# 添加DOVER到路径
sys.path.insert(0, '/tmp/DOVER')

class VideoDataset(Dataset):
    """视频数据集，用于批量处理"""
    def __init__(self, video_info_list, base_dir):
        self.video_info_list = video_info_list
        self.base_dir = base_dir

    def __len__(self):
        return len(self.video_info_list)

    def __getitem__(self, idx):
        video_info = self.video_info_list[idx]
        video_path = video_info['path']

        try:
            from dover.datasets import spatial_temporal_view_decomposition

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

            from dover.datasets import UnifiedFrameSampler

            # 创建temporal samplers
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
            processed_views = {}
            for k, v in views.items():
                num_clips = dopt[k].get("num_clips", 1)
                processed_views[k] = (
                    ((v.permute(1, 2, 3, 0) - mean) / std)
                    .permute(3, 0, 1, 2)
                    .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
                    .transpose(0, 1)
                )

            return {
                'filename': video_info['filename'],
                'attack_subset': video_info['attack_subset'],
                'path': video_path,
                'views': processed_views,
                'success': True
            }

        except Exception as e:
            return {
                'filename': video_info['filename'],
                'attack_subset': video_info['attack_subset'],
                'path': video_path,
                'views': None,
                'success': False,
                'error': str(e)
            }

def collate_fn(batch):
    """自定义collate函数处理不同大小的视频"""
    filenames = [item['filename'] for item in batch]
    attack_subsets = [item['attack_subset'] for item in batch]
    paths = [item['path'] for item in batch]
    success_flags = [item['success'] for item in batch]

    # 分离成功和失败的样本
    success_indices = [i for i, flag in enumerate(success_flags) if flag]
    fail_indices = [i for i, flag in enumerate(success_flags) if not flag]

    batch_views = {}
    if success_indices:
        # 只处理成功的样本
        for key in ['technical', 'aesthetic']:
            views_list = [batch[i]['views'][key] for i in success_indices if key in batch[i]['views']]
            if views_list:
                # 将views堆叠为batch
                try:
                    batch_views[key] = torch.cat(views_list, dim=0)
                except:
                    # 如果无法堆叠，保持为列表
                    batch_views[key] = views_list

    return {
        'filenames': filenames,
        'attack_subsets': attack_subsets,
        'paths': paths,
        'success_flags': success_flags,
        'success_indices': success_indices,
        'fail_indices': fail_indices,
        'batch_views': batch_views
    }

def setup_dover():
    """初始化DOVER模型"""
    print("初始化DOVER模型...")
    from dover.models import DOVER

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

    print("✓ DOVER模型初始化完成")
    return dover_model, device

def process_batch(dover_model, batch_data, device):
    """处理一个batch的数据"""
    results = []

    if not batch_data['success_indices']:
        # 所有样本都失败了
        for i in range(len(batch_data['filenames'])):
            results.append({
                'filename': batch_data['filenames'][i],
                'attack_subset': batch_data['attack_subsets'][i],
                'technical': None,
                'aesthetic': None,
                'overall': None,
                'success': False
            })
        return results

    # 将成功的batch数据移到GPU
    batch_views_gpu = {}
    for key, views in batch_data['batch_views'].items():
        if isinstance(views, torch.Tensor):
            batch_views_gpu[key] = views.to(device)

    # 运行DOVER评估
    with torch.no_grad():
        try:
            if batch_views_gpu:
                model_output = dover_model(batch_views_gpu)
                # 处理输出
                results_list = []
                for i in range(len(model_output)):
                    scores = [model_output[i][j].mean().item() for j in range(len(model_output[i]))]
                    results_list.append({
                        'technical': scores[0],
                        'aesthetic': scores[1],
                        'overall': 0.1 * scores[0] + 0.9 * scores[1]
                    })
            else:
                results_list = []
        except Exception as e:
            print(f"Batch处理错误: {e}")
            results_list = []

    # 组装完整结果
    result_idx = 0
    for i in range(len(batch_data['filenames'])):
        if i in batch_data['success_indices'] and result_idx < len(results_list):
            results.append({
                'filename': batch_data['filenames'][i],
                'attack_subset': batch_data['attack_subsets'][i],
                'technical': results_list[result_idx]['technical'],
                'aesthetic': results_list[result_idx]['aesthetic'],
                'overall': results_list[result_idx]['overall'],
                'success': True
            })
            result_idx += 1
        else:
            results.append({
                'filename': batch_data['filenames'][i],
                'attack_subset': batch_data['attack_subsets'][i],
                'technical': None,
                'aesthetic': None,
                'overall': None,
                'success': False
            })

    return results

def scan_video_files(base_dir):
    """扫描所有视频文件并分类"""
    print("扫描视频文件...")

    video_categories = {
        'TI2I': {'original': [], 'erased': []},
        'T2V': {'original': [], 'erased': []},
        'TI2V': {'original': [], 'erased': []}
    }

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
                # 查找对应的CSV文件
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

                    video_categories[task_type][result_type].append({
                        'path': video_file,
                        'attack_subset': attack_subset,
                        'filename': filename
                    })
                    total_videos += 1

    print(f"✓ 找到 {total_videos} 个视频文件")

    # 打印统计信息
    for task_type in ['TI2I', 'T2V', 'TI2V']:
        print(f"{task_type}:")
        print(f"  - 原始: {len(video_categories[task_type]['original'])} 个视频")
        print(f"  - 消除: {len(video_categories[task_type]['erased'])} 个视频")

    return video_categories

def batch_evaluate_videos(video_categories, dover_model, device, output_file, batch_size=8):
    """批量评估视频"""

    # 初始化结果存储
    results = {
        'TI2I': {'original': defaultdict(list), 'erased': defaultdict(list)},
        'T2V': {'original': defaultdict(list), 'erased': defaultdict(list)},
        'TI2V': {'original': defaultdict(list), 'erased': defaultdict(list)}
    }

    # 进度文件
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

            if not video_list:
                continue

            print(f"\n处理 {task_type} - {result_type} ({len(video_list)} 个视频)")
            print(f"批处理大小: {batch_size}")

            # 过滤已处理的文件
            remaining_videos = [v for v in video_list if v['path'] not in processed_files]
            print(f"待处理: {len(remaining_videos)} 个视频")

            if not remaining_videos:
                # 统计已处理的结果
                # 这里需要从之前保存的结果中恢复
                total_count += len(video_list)
                continue

            # 创建数据集和dataloader
            dataset = VideoDataset(remaining_videos, None)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0,  # 避免多进程问题
                pin_memory=True
            )

            # 批量处理
            for batch_idx, batch_data in enumerate(dataloader):
                print(f"  处理批次 {batch_idx + 1}/{len(dataloader)} ({len(batch_data['filenames'])} 个视频)")

                # 处理批次
                batch_results = process_batch(dover_model, batch_data, device)

                # 保存结果
                for idx, result in enumerate(batch_results):
                    video_path = batch_data['paths'][idx]

                    if result['success']:
                        results[task_type][result_type][result['attack_subset']].append({
                            'filename': result['filename'],
                            'technical': result['technical'],
                            'aesthetic': result['aesthetic'],
                            'overall': result['overall']
                        })
                        success_count += 1
                    else:
                        fail_count += 1

                    processed_files.add(video_path)
                    total_count += 1

                # 打印进度
                for i, filename in enumerate(batch_data['filenames']):
                    if batch_results[i]['success']:
                        r = batch_results[i]
                        print(f"    [{total_count - len(batch_data['filenames']) + i + 1}] {filename} ✓ T={r['technical']:.4f}, A={r['aesthetic']:.4f}, O={r['overall']:.4f}")
                    else:
                        print(f"    [{total_count - len(batch_data['filenames']) + i + 1}] {filename} ✗")

                # 定期保存进度
                if total_count % 50 == 0:
                    progress_data = {
                        'processed_files': list(processed_files),
                        'total_count': total_count,
                        'success_count': success_count,
                        'fail_count': fail_count
                    }
                    with open(progress_file, 'w') as f:
                        json.dump(progress_data, f)

                    # 保存中间结果
                    with open(output_file, 'w') as f:
                        json.dump({
                            'results': results,
                            'summary': {
                                'total_count': total_count,
                                'success_count': success_count,
                                'fail_count': fail_count,
                                'duration_seconds': (datetime.now() - start_time).total_seconds()
                            }
                        }, f, indent=2)

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
        f.write("DOVER批量视频质量评估报告 (批量处理优化版)\n")
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
    output_json = os.path.join(base_dir, "dover_batch_results_batch4.json")
    output_txt = os.path.join(base_dir, "dover_batch_statistics_report_batch4.txt")

    # 批处理大小 - 可以根据GPU内存调整
    batch_size = 4  # 降低批处理大小以提高成功率

    print("=" * 70)
    print("DOVER批量视频质量评估 (批量处理优化版)")
    print(f"批处理大小: {batch_size}")
    print("=" * 70)
    print()

    try:
        # 1. 初始化DOVER
        dover_model, device = setup_dover()

        # 2. 扫描视频文件
        video_categories = scan_video_files(base_dir)

        # 3. 批量评估
        print(f"\n开始批量评估...")
        print(f"批处理大小: {batch_size}")
        print(f"结果将保存至: {output_json}")

        results = batch_evaluate_videos(
            video_categories, dover_model, device, output_json, batch_size
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