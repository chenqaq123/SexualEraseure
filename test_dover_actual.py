#!/usr/bin/env python3
"""
DOVER实际功能测试脚本
实际运行DOVER评估并获取结果
"""

import torch
import os
import sys
import yaml

# 添加DOVER到路径
sys.path.insert(0, '/tmp/DOVER')

def test_dover_with_video():
    print("=" * 60)
    print("DOVER实际功能测试")
    print("=" * 60)

    try:
        from dover.models import DOVER
        from dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition
        import decord
        print("✓ DOVER导入成功")

        # 配置参数
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")

        # DOVER配置
        dover_config = {
            "backbone_size": "divided",
            "backbone_preserve_keys": "technical,aesthetic",
            "divide_head": True,
            "vqa_head": {
                "in_channels": 768,
                "hidden_channels": 64
            },
            "backbone": {
                "technical": {
                    "type": "swin_tiny_grpb",
                    "checkpoint": True,
                    "pretrained": True
                },
                "aesthetic": {
                    "type": "conv_tiny"
                }
            }
        }

        print("创建DOVER模型...")
        dover_model = DOVER(**dover_config)
        print("✓ DOVER模型创建成功")

        # 加载权重
        weights_path = "/tmp/DOVER/pretrained_weights/DOVER.pth"
        print(f"加载权重: {weights_path}")

        state_dict = torch.load(weights_path, map_location=device)
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        dover_model.load_state_dict(state_dict, strict=False)
        dover_model.to(device)
        dover_model.eval()
        print("✓ 权重加载成功")

        # 测试视频文件
        test_video = "/tmp/DOVER/demo/17734.mp4"
        if not os.path.exists(test_video):
            print(f"测试视频不存在: {test_video}")
            return False

        print(f"测试视频: {test_video}")

        # DOVER预处理参数
        mean = torch.FloatTensor([123.675, 116.28, 103.53])
        std = torch.FloatTensor([58.395, 57.12, 57.375])

        # 配置采样参数
        dopt = {
            "technical": {
                "fragments_h": 7,
                "fragments_w": 7,
                "fsize_h": 32,
                "fsize_w": 32,
                "aligned": 32,
                "clip_len": 32,
                "frame_interval": 2,
                "num_clips": 1  # 减少到1个clip加快测试
            },
            "aesthetic": {
                "size_h": 224,
                "size_w": 224,
                "clip_len": 32,
                "frame_interval": 2,
                "t_frag": 32,
                "num_clips": 1
            }
        }

        print("创建temporal samplers...")
        temporal_samplers = {}
        for stype, sopt in dopt.items():
            if "t_frag" not in sopt:
                temporal_samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"]
                )
            else:
                temporal_samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"] // sopt["t_frag"],
                    sopt["t_frag"],
                    sopt["frame_interval"],
                    sopt["num_clips"],
                )
        print("✓ Temporal samplers创建成功")

        print("进行view decomposition...")
        try:
            views, _ = spatial_temporal_view_decomposition(
                test_video, dopt, temporal_samplers
            )
            print("✓ View decomposition成功")
            print(f"  - Technical views: {views['technical'].shape if 'technical' in views else 'N/A'}")
            print(f"  - Aesthetic views: {views['aesthetic'].shape if 'aesthetic' in views else 'N/A'}")
        except Exception as e:
            print(f"✗ View decomposition失败: {e}")
            return False

        # 预处理views
        print("预处理views...")
        for k, v in views.items():
            num_clips = dopt[k].get("num_clips", 1)
            views[k] = (
                ((v.permute(1, 2, 3, 0) - mean) / std)
                .permute(3, 0, 1, 2)
                .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
                .transpose(0, 1)
                .to(device)
            )
        print("✓ Views预处理成功")

        # 运行DOVER评估
        print("运行DOVER评估...")
        with torch.no_grad():
            results = [r.mean().item() for r in dover_model(views)]

        print("✓ DOVER评估成功！")
        print()
        print("=" * 60)
        print("DOVER评估结果")
        print("=" * 60)
        print(f"Technical Quality (技术质量):  {results[0]:.4f}")
        print(f"Aesthetic Quality (美学质量):  {results[1]:.4f}")

        # 计算综合分数
        overall = 0.1 * results[0] + 0.9 * results[1]
        print(f"Overall Quality (综合质量):    {overall:.4f}")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"✗ DOVER测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dover_with_eval_videos():
    """测试实际评估视频文件"""
    print()
    print("=" * 60)
    print("测试实际评估视频文件")
    print("=" * 60)

    # 查找实际评估视频
    eval_dirs = [
        "/home/mlsnrs/data/cgx/cgx_nas/T2I/Concept_Erasure/SexualErasure_CCS/eval/T2V_hunyuanvideo/results",
        "/home/mlsnrs/data/cgx/cgx_nas/T2I/Concept_Erasure/SexualErasure_CCS/eval/TI2V_hunyuanvideo/results"
    ]

    test_videos = []
    for eval_dir in eval_dirs:
        if os.path.exists(eval_dir):
            import glob
            videos = glob.glob(os.path.join(eval_dir, "*.mp4"))
            if videos:
                test_videos.extend(videos[:2])  # 取前2个视频

    if not test_videos:
        print("没有找到测试视频文件")
        return False

    print(f"找到 {len(test_videos)} 个测试视频")

    try:
        from dover.models import DOVER
        from dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 创建DOVER模型
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

        weights_path = "/tmp/DOVER/pretrained_weights/DOVER.pth"
        state_dict = torch.load(weights_path, map_location=device)
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        dover_model.load_state_dict(state_dict, strict=False)
        dover_model.to(device)
        dover_model.eval()

        mean = torch.FloatTensor([123.675, 116.28, 103.53])
        std = torch.FloatTensor([58.395, 57.12, 57.375])

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

        print()
        print("实际评估视频DOVER测试结果:")
        print("-" * 60)

        results_summary = []
        for i, video_path in enumerate(test_videos[:3], 1):  # 最多测试3个视频
            try:
                print(f"\n[{i}] 视频: {os.path.basename(video_path)}")

                views, _ = spatial_temporal_view_decomposition(video_path, dopt, temporal_samplers)

                for k, v in views.items():
                    num_clips = dopt[k].get("num_clips", 1)
                    views[k] = (
                        ((v.permute(1, 2, 3, 0) - mean) / std)
                        .permute(3, 0, 1, 2)
                        .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
                        .transpose(0, 1)
                        .to(device)
                    )

                with torch.no_grad():
                    results = [r.mean().item() for r in dover_model(views)]

                technical = results[0]
                aesthetic = results[1]
                overall = 0.1 * technical + 0.9 * aesthetic

                print(f"  Technical: {technical:.4f}, Aesthetic: {aesthetic:.4f}, Overall: {overall:.4f}")

                results_summary.append({
                    'video': os.path.basename(video_path),
                    'technical': technical,
                    'aesthetic': aesthetic,
                    'overall': overall
                })

            except Exception as e:
                print(f"  ✗ 评估失败: {e}")
                continue

        if results_summary:
            print()
            print("-" * 60)
            print("DOVER测试汇总:")
            avg_technical = sum(r['technical'] for r in results_summary) / len(results_summary)
            avg_aesthetic = sum(r['aesthetic'] for r in results_summary) / len(results_summary)
            avg_overall = sum(r['overall'] for r in results_summary) / len(results_summary)

            print(f"平均Technical Quality:  {avg_technical:.4f}")
            print(f"平均Aesthetic Quality:  {avg_aesthetic:.4f}")
            print(f"平均Overall Quality:    {avg_overall:.4f}")
            print("-" * 60)

        return True

    except Exception as e:
        print(f"✗ 实际视频评估失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始DOVER功能测试...")

    # 测试1: 使用DOVER自带演示视频
    success1 = test_dover_with_video()

    # 测试2: 使用实际评估视频
    success2 = test_dover_with_eval_videos()

    print()
    if success1 or success2:
        print("✓ DOVER测试完成！DOVER功能正常工作。")
        sys.exit(0)
    else:
        print("✗ DOVER测试失败")
        sys.exit(1)