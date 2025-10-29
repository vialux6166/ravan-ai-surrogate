#!/usr/bin/env python3
"""
GPU Verification Script for Ravan Quantum-ML System
Verifies CUDA availability and RTX 3090 detection
"""

import sys

def check_cuda():
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            print(f"GPU name: {torch.cuda.get_device_name(0)}")
            
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024**3)
            print(f"VRAM: {vram_gb:.2f} GB")
            
            if 'RTX 3090' in torch.cuda.get_device_name(0):
                print("✓ RTX 3090 detected successfully!")
                return True
            else:
                print("⚠ Warning: Expected RTX 3090 but found different GPU")
                return False
        else:
            print("✗ CUDA not available")
            return False
    except ImportError:
        print("✗ PyTorch not installed")
        return False

if __name__ == '__main__':
    success = check_cuda()
    sys.exit(0 if success else 1)
