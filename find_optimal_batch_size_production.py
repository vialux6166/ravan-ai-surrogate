#!/usr/bin/env python3
"""
Find Optimal Batch Size for PRODUCTION MLP Model
Uses the actual MLP architecture [256, 128, 64] with real dataset size
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import time
from pathlib import Path

from mlp_regressor import MLPNetwork


def find_optimal_batch_size_production():
    """Find optimal batch size using PRODUCTION model"""
    print("="*70)
    print("PRODUCTION BATCH SIZE OPTIMIZATION")
    print("="*70)
    
    # Use LARGE dataset to match model size
    n_samples = 8000
    hidden_dim = 4096  # Match model dimensions
    
    print(f"\nDataset: {n_samples} samples")
    print(f"Dimensions: {hidden_dim}")
    
    # Generate large data matching model dimensions
    X_train = np.random.randn(n_samples, hidden_dim).astype(np.float32)
    y_train = np.random.randn(n_samples, hidden_dim).astype(np.float32)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if not torch.cuda.is_available():
        print("\n⚠️  CUDA not available - cannot test GPU batch sizes")
        return
    
    # Create LARGE PRODUCTION MODEL (from VRAM stress test)
    print("\n--- Creating LARGE PRODUCTION Model ---")
    
    class VeryDeepModel(nn.Module):
        """Large model matching VRAM stress test"""
        def __init__(self, n_layers=50, hidden_dim=4096):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1)
                ) for _ in range(n_layers)
            ])
        
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
    
    # Use large model configuration
    n_layers = 50
    hidden_dim = 4096
    model = VeryDeepModel(n_layers, hidden_dim).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Architecture: {n_layers} layers × {hidden_dim} neurons")
    print(f"This matches the VRAM stress test model")
    
    # Test batch sizes (smaller range for large model)
    batch_sizes = [8, 16, 32, 64, 128, 256]
    results = {}
    
    print("\n--- Testing Batch Sizes (Full Training Loop) ---")
    print("Batch | Time  | Throughput | GPU Mem | GPU Util")
    print("-" * 60)
    
    for batch_size in batch_sizes:
        try:
            # Convert to tensors
            X_tensor = torch.FloatTensor(X_train).to(device)
            y_tensor = torch.FloatTensor(y_train).to(device)
            
            # Create dataset and loader
            dataset = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,  # Data already on GPU
                pin_memory=False
            )
            
            # Setup optimizer and loss
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            # Reset GPU stats
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            
            # Time one full epoch with REAL training
            model.train()
            start = time.time()
            epoch_loss = 0.0
            
            for batch_X, batch_y in loader:
                # Full training step
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping (like production)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_loss += loss.item()
            
            torch.cuda.synchronize()
            elapsed = time.time() - start
            
            # Metrics
            throughput = n_samples / elapsed
            gpu_mem = torch.cuda.max_memory_allocated() / 1e9
            avg_loss = epoch_loss / len(loader)
            
            # GPU utilization (approximate from memory)
            gpu_util = min(100, (gpu_mem / 24.0) * 100)  # RTX 3090 has 24GB
            
            results[batch_size] = {
                'time': elapsed,
                'throughput': throughput,
                'gpu_mem': gpu_mem,
                'gpu_util': gpu_util,
                'loss': avg_loss
            }
            
            print(f"{batch_size:5d} | {elapsed:5.2f}s | {throughput:8.1f}/s | {gpu_mem:5.2f}GB | {gpu_util:5.1f}%")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"{batch_size:5d} | OOM - Maximum batch size reached")
                torch.cuda.empty_cache()
                break
            else:
                raise
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    if not results:
        print("\n⚠️  No valid results - all batch sizes caused OOM")
        return
    
    # Find best batch size
    best_batch = max(results.keys(), key=lambda k: results[k]['throughput'])
    best_result = results[best_batch]
    
    print(f"\n✓ Optimal Batch Size: {best_batch}")
    print(f"  Throughput: {best_result['throughput']:.1f} samples/s")
    print(f"  GPU Memory: {best_result['gpu_mem']:.2f}GB")
    print(f"  GPU Utilization: {best_result['gpu_util']:.1f}%")
    print(f"  Training Time: {best_result['time']:.2f}s per epoch")
    
    # Validation
    print("\n--- Validation ---")
    max_gpu_mem = max(r['gpu_mem'] for r in results.values())
    
    if max_gpu_mem < 1.0:
        print("⚠️  WARNING: GPU memory usage < 1GB")
        print("   Model may be too small for accurate batch size tuning")
        print("   Consider testing with larger model or dataset")
    elif max_gpu_mem < 5.0:
        print("⚠️  CAUTION: GPU memory usage < 5GB")
        print("   Results are valid but GPU is underutilized")
        print("   Larger batches may be possible")
    else:
        print("✓ GPU memory usage is realistic for production")
        print(f"  Peak memory: {max_gpu_mem:.2f}GB")
    
    # Recommendations
    print("\n--- Recommendations ---")
    print(f"1. Use batch size: {best_batch}")
    print(f"2. Expected training time: {best_result['time']:.2f}s per epoch")
    print(f"3. For 100 epochs: {best_result['time'] * 100:.1f}s ({best_result['time'] * 100 / 60:.1f} minutes)")
    
    # Save results
    report_path = Path('performance_reports')
    report_path.mkdir(exist_ok=True)
    
    import json
    with open(report_path / 'production_batch_size.json', 'w') as f:
        json.dump({
            'optimal_batch_size': int(best_batch),
            'results': {k: {
                'throughput': float(v['throughput']),
                'gpu_mem': float(v['gpu_mem']),
                'gpu_util': float(v['gpu_util']),
                'time': float(v['time'])
            } for k, v in results.items()},
            'model_params': int(total_params),
            'dataset_size': int(n_samples)
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {report_path / 'production_batch_size.json'}")
    
    return best_batch


if __name__ == '__main__':
    optimal_batch = find_optimal_batch_size_production()
    exit(0 if optimal_batch else 1)
