#!/usr/bin/env python3
"""
ML Training Optimization
Task 20.2: Optimize ML training

Implements:
- Persistent DataLoader workers
- Optimal batch size tuning
- TorchScript compilation for inference
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
from pathlib import Path

from mlp_regressor import MLPRegressor
from model_base import ModelConfig


class OptimizedDataset(Dataset):
    """Optimized PyTorch Dataset with persistent workers"""
    
    def __init__(self, X, y):
        # Convert to tensors once
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def find_optimal_batch_size(model, X_train, y_train, device):
    """Find optimal batch size for GPU utilization with REAL training loop"""
    print("\n--- Finding Optimal Batch Size (Real Training Loop) ---")
    
    batch_sizes = [32, 64, 128, 256, 512, 1024]
    results = {}
    
    # Use full dataset for realistic test
    n_samples = len(X_train)
    print(f"Testing with {n_samples} samples (full dataset)")
    
    for batch_size in batch_sizes:
        try:
            # Create dataset and loader
            dataset = OptimizedDataset(X_train, y_train)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
                persistent_workers=True
            )
            
            # Setup optimizer and loss (REAL training)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            # Time one full epoch with backward pass
            model.train()
            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            
            start = time.time()
            epoch_loss = 0.0
            
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                # Full training step
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            elapsed = time.time() - start
            throughput = n_samples / elapsed
            
            # Check GPU utilization
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.max_memory_allocated() / 1e9
            else:
                gpu_mem = 0
            
            results[batch_size] = {
                'time': elapsed,
                'throughput': throughput,
                'gpu_mem': gpu_mem,
                'loss': epoch_loss / len(loader)
            }
            
            print(f"Batch {batch_size:4d}: {elapsed:.3f}s, {throughput:.1f} samples/s, {gpu_mem:.2f}GB GPU")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch {batch_size:4d}: OOM - Maximum batch size reached")
                break
            else:
                raise
    
    # Find best batch size (highest throughput without OOM)
    if results:
        best_batch = max(results.keys(), key=lambda k: results[k]['throughput'])
        print(f"\n✓ Optimal batch size: {best_batch}")
        print(f"  Peak throughput: {results[best_batch]['throughput']:.1f} samples/s")
        print(f"  GPU memory: {results[best_batch]['gpu_mem']:.2f}GB")
    else:
        best_batch = 128  # Safe default
        print(f"\n⚠ Using default batch size: {best_batch}")
    
    return best_batch, results


def test_persistent_workers():
    """Test persistent DataLoader workers"""
    print("\n--- Testing Persistent Workers ---")
    
    # Generate dummy data
    X = np.random.randn(1000, 5).astype(np.float32)
    y = np.random.randn(1000, 4).astype(np.float32)
    
    dataset = OptimizedDataset(X, y)
    
    # Test without persistent workers
    loader_normal = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=False
    )
    
    start = time.time()
    for _ in range(10):
        for batch in loader_normal:
            pass
    time_normal = time.time() - start
    
    # Test with persistent workers
    loader_persistent = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    start = time.time()
    for _ in range(10):
        for batch in loader_persistent:
            pass
    time_persistent = time.time() - start
    
    print(f"Normal workers: {time_normal:.3f}s")
    print(f"Persistent workers: {time_persistent:.3f}s")
    print(f"Speedup: {time_normal/time_persistent:.2f}x")
    
    return time_persistent < time_normal


def compile_model_torchscript(model, input_shape):
    """Compile model with TorchScript for faster inference"""
    print("\n--- Compiling Model with TorchScript ---")
    
    model.eval()
    
    # Create example input
    example_input = torch.randn(1, input_shape)
    if torch.cuda.is_available():
        example_input = example_input.cuda()
        model = model.cuda()
    
    # Trace model
    try:
        traced_model = torch.jit.trace(model, example_input)
        print("✓ Model compiled successfully")
        
        # Test inference speed
        n_tests = 1000
        
        # Original model
        start = time.time()
        with torch.no_grad():
            for _ in range(n_tests):
                _ = model(example_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time_original = time.time() - start
        
        # Compiled model
        start = time.time()
        with torch.no_grad():
            for _ in range(n_tests):
                _ = traced_model(example_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time_compiled = time.time() - start
        
        print(f"Original model: {time_original/n_tests*1000:.3f}ms per inference")
        print(f"Compiled model: {time_compiled/n_tests*1000:.3f}ms per inference")
        print(f"Speedup: {time_original/time_compiled:.2f}x")
        
        return traced_model, time_compiled < time_original
        
    except Exception as e:
        print(f"✗ Compilation failed: {e}")
        return None, False


def optimize_training():
    """Run all ML training optimizations"""
    print("="*70)
    print("ML TRAINING OPTIMIZATION")
    print("="*70)
    
    # Generate synthetic data
    print("\nGenerating synthetic training data...")
    n_samples = 8000
    n_features = 5
    n_outputs = 4
    
    X_train = np.random.randn(n_samples, n_features).astype(np.float32)
    y_train = np.random.randn(n_samples, n_outputs).astype(np.float32)
    
    # Normalize
    from sklearn.preprocessing import StandardScaler
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X_train = X_scaler.fit_transform(X_train)
    y_train = y_scaler.fit_transform(y_train)
    
    print(f"✓ Generated {n_samples} samples")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    from mlp_regressor import MLPNetwork
    model = MLPNetwork(
        input_dim=n_features,
        output_dim=n_outputs,
        hidden_layers=[256, 128, 64],
        dropout=0.2
    ).to(device)
    
    # Test 1: Find optimal batch size
    optimal_batch, batch_results = find_optimal_batch_size(
        model, X_train, y_train, device
    )
    
    # Test 2: Persistent workers
    persistent_faster = test_persistent_workers()
    
    # Test 3: TorchScript compilation
    compiled_model, compilation_faster = compile_model_torchscript(
        model, n_features
    )
    
    # Summary
    print("\n" + "="*70)
    print("OPTIMIZATION SUMMARY")
    print("="*70)
    
    print(f"\n1. Optimal Batch Size: {optimal_batch}")
    print(f"   - Throughput: {batch_results[optimal_batch]['throughput']:.1f} samples/s")
    print(f"   - GPU Memory: {batch_results[optimal_batch]['gpu_mem']:.2f}GB")
    
    print(f"\n2. Persistent Workers: {'✓ Faster' if persistent_faster else '✗ No improvement'}")
    
    print(f"\n3. TorchScript Compilation: {'✓ Faster' if compilation_faster else '✗ No improvement'}")
    
    # Save recommendations
    recommendations = {
        'optimal_batch_size': int(optimal_batch),
        'use_persistent_workers': persistent_faster,
        'use_torchscript': compilation_faster,
        'batch_results': {k: {
            'throughput': float(v['throughput']),
            'gpu_mem': float(v['gpu_mem'])
        } for k, v in batch_results.items()}
    }
    
    report_path = Path('performance_reports')
    report_path.mkdir(exist_ok=True)
    
    import json
    with open(report_path / 'ml_training_optimization.json', 'w') as f:
        json.dump(recommendations, f, indent=2)
    
    print(f"\n✓ Recommendations saved to: {report_path / 'ml_training_optimization.json'}")
    
    return True


if __name__ == '__main__':
    success = optimize_training()
    exit(0 if success else 1)
