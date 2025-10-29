"""
Test XGBoost GPU support
"""

import sys
sys.path.insert(0, '/home/windows/ravan-quantum-ml/src')

import numpy as np
import xgboost as xgb
import time

print("=" * 80)
print("Testing XGBoost GPU Support")
print("=" * 80)

# Check XGBoost version
print(f"\nXGBoost version: {xgb.__version__}")

# Test 1: Basic GPU availability
print("\n1. Testing basic GPU availability...")
try:
    import torch
    cuda_available = torch.cuda.is_available()
    print(f"   PyTorch CUDA available: {cuda_available}")
    if cuda_available:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
except Exception as e:
    print(f"   Error checking PyTorch: {e}")

# Test 2: XGBoost GPU training
print("\n2. Testing XGBoost GPU training...")

# Generate synthetic data
n_samples = 10000
n_features = 20
X_train = np.random.rand(n_samples, n_features).astype(np.float32)
y_train = np.random.rand(n_samples).astype(np.float32)

X_val = np.random.rand(1000, n_features).astype(np.float32)
y_val = np.random.rand(1000).astype(np.float32)

print(f"   Data: {n_samples} samples, {n_features} features")

# Test GPU training (XGBoost 3.x style)
try:
    print("\n   Testing tree_method='hist' with device='cuda' (XGBoost 3.x)...")
    start_time = time.time()
    
    model_gpu = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=7,
        learning_rate=0.1,
        tree_method='hist',
        device='cuda',
        objective='reg:squarederror'
    )
    
    model_gpu.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    gpu_time = time.time() - start_time
    gpu_pred = model_gpu.predict(X_val)
    
    print(f"   ✓ GPU training successful!")
    print(f"   Training time: {gpu_time:.2f}s")
    print(f"   Predictions shape: {gpu_pred.shape}")
    
except Exception as e:
    print(f"   ✗ GPU training failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Compare GPU vs CPU
print("\n3. Comparing GPU vs CPU performance...")

try:
    # CPU training
    print("   Training on CPU...")
    start_time = time.time()
    
    model_cpu = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=7,
        learning_rate=0.1,
        tree_method='hist',
        objective='reg:squarederror',
        n_jobs=-1
    )
    
    model_cpu.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    cpu_time = time.time() - start_time
    cpu_pred = model_cpu.predict(X_val)
    
    print(f"   CPU training time: {cpu_time:.2f}s")
    
    if 'gpu_time' in locals():
        speedup = cpu_time / gpu_time
        print(f"\n   GPU Speedup: {speedup:.2f}x faster than CPU")
        
        if speedup > 1.5:
            print("   ✓ GPU acceleration is working well!")
        elif speedup > 1.0:
            print("   ⚠ GPU is faster but speedup is modest")
        else:
            print("   ⚠ GPU is slower than CPU (may indicate GPU not being used)")
    
except Exception as e:
    print(f"   Error in CPU training: {e}")

# Test 4: Test with our XGBoostRegressor class
print("\n4. Testing XGBoostRegressor class...")

try:
    from models.base import ModelConfig
    from models.xgboost_model import XGBoostRegressor
    
    config = ModelConfig(
        model_type='xgboost',
        input_dim=n_features,
        output_dim=1,
        hyperparameters={
            'n_estimators': 100,
            'max_depth': 7,
            'learning_rate': 0.1
        },
        save_path='./test_xgboost_model'
    )
    
    model = XGBoostRegressor(config)
    
    start_time = time.time()
    metrics = model.train(X_train, y_train, X_val, y_val)
    training_time = time.time() - start_time
    
    print(f"   ✓ XGBoostRegressor training successful!")
    print(f"   Training time: {training_time:.2f}s")
    print(f"   R² score: {metrics.r2_score:.4f}")
    print(f"   MSE: {metrics.val_mse:.6f}")
    
except Exception as e:
    print(f"   ✗ XGBoostRegressor test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("XGBoost GPU Test Complete")
print("=" * 80)
