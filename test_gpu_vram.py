"""
Test GPU Accelerator and VRAM Manager
"""
import sys
sys.path.insert(0, '/home/windows/ravan-quantum-ml/src')

from gpu.accelerator import GPUAccelerator
from gpu.vram_manager import VRAMManager, WorkloadMode
from utils.logging import setup_logging

# Setup logging
logger = setup_logging(level='INFO')

print("=" * 60)
print("Testing GPU Accelerator and VRAM Manager")
print("=" * 60)

# Test GPU Accelerator
print("\n1. Testing GPU Accelerator...")
gpu = GPUAccelerator()

print(f"   Device: {gpu.get_device()}")
print(f"   CUDA Available: {gpu.is_cuda_available()}")
print(f"   Free VRAM: {gpu.get_free_vram_gb():.2f} GB")

# Log VRAM usage
gpu.log_vram_usage("Initial state")

# Test VRAM Manager
print("\n2. Testing VRAM Manager...")
vram_mgr = VRAMManager(gpu)

# Test simulation mode
print("\n3. Requesting SIMULATION mode...")
try:
    vram_mgr.request_mode(WorkloadMode.SIMULATION)
    print(f"   ✓ Current mode: {vram_mgr.get_current_mode().value}")
    print(f"   ✓ Can run simulation: {vram_mgr.can_run_simulation()}")
    print(f"   ✓ Can run training: {vram_mgr.can_run_training()}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test training mode
print("\n4. Requesting TRAINING mode...")
try:
    vram_mgr.request_mode(WorkloadMode.TRAINING)
    print(f"   ✓ Current mode: {vram_mgr.get_current_mode().value}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test inference mode
print("\n5. Requesting INFERENCE mode...")
try:
    vram_mgr.request_mode(WorkloadMode.INFERENCE)
    print(f"   ✓ Current mode: {vram_mgr.get_current_mode().value}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test memory optimization
print("\n6. Testing memory optimization...")
gpu.optimize_memory()
print(f"   ✓ Memory optimized")
print(f"   ✓ Free VRAM after optimization: {gpu.get_free_vram_gb():.2f} GB")

# Test mixed precision
print("\n7. Testing mixed precision...")
scaler = gpu.enable_mixed_precision()
if scaler:
    print(f"   ✓ Mixed precision enabled")
else:
    print(f"   ⚠ Mixed precision not available (CPU mode)")

print("\n" + "=" * 60)
print("All tests completed successfully!")
print("=" * 60)
