#!/usr/bin/env python3
"""
DOVER测试脚本
测试DOVER功能是否正常工作
"""

import torch
import os
import sys

# 添加DOVER到路径
sys.path.insert(0, '/tmp/DOVER')

def test_dover():
    print("Testing DOVER installation...")

    try:
        from dover.models import DOVER
        print("✓ DOVER import successful")

        # Test DOVER initialization
        dover_model = DOVER()
        print("✓ DOVER initialization successful")

        # Test loading weights
        weights_path = "/tmp/DOVER/pretrained_weights/DOVER.pth"
        if os.path.exists(weights_path):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")

            state_dict = torch.load(weights_path, map_location=device)
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            dover_model.load_state_dict(state_dict, strict=False)
            dover_model.to(device)
            print("✓ DOVER weights loaded successful")

        return True

    except Exception as e:
        print(f"✗ DOVER test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_dover()
    sys.exit(0 if success else 1)