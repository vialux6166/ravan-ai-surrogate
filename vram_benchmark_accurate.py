#!/usr/bin/env python3
"""
Accurate VRAM Benchmark for Ravan Quantum-ML System
Properly measures gradient checkpointing and mixed precision benefits
"""

import torch
import torch.nn as nn
import time
from typing import Dict


class DeepModel(nn.Module):
    """Very deep model to stress test VRAM"""
    
    def __init__(self, input_dim: int, output_dim: int, n_layers: int = 30, 
                 hidden_dim: int = 4096, use_checkpointing: bool = False):
        super().__init__()
        
        self.use_checkpointing = use_checkpointing
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ))
        
        # Output layer
        self.output = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.layers[0](x)
        
        if self.use_checkpointing and self.training:
            # Use checkpointing for hidden layers
            for layer in self.layers[1:]:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
        else:
            for layer in self.layers[1:]:
                x = layer(x)
        
        return self.output(x)


def benchmark_configuration(config_name: str, model_params: Dict, 
                           training_params: Dict) -> Dict:
    """Benchmark a specific configuration"""
    
    device = torch.device('cuda')
    
    # Clear VRAM
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    print(f"\n{'='*70}")
    print(f"BENCHMARKING: {config_name}")
    print(f"{'='*70}")
    
    # Create model
    model = DeepModel(**model_params).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    model_size_gb = total_params * 4 / 1e9  # FP32
    print(f"Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"Model size: {model_size_gb:.2f} GB (FP32)")
    
    # Create data
    batch_size = training_params['batch_size']
    X = torch.randn(batch_size, model_params['input_dim'], device=device)
    y = torch.randn(batch_size, model_params['output_dim'], device=device)
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Mixed precision
    use_amp = training_params.get('use_amp', False)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    if use_amp:
        print(f"Mixed Precision: ENABLED")
    if model_params.get('use_checkpointing', False):
        print(f"Gradient Checkpointing: ENABLED")
    print(f"Batch size: {batch_size}")
    
    # Measure memory after model loading
    torch.cuda.synchronize()
    mem_after_model = torch.cuda.memory_allocated() / 1e9
    print(f"\nVRAM after model load: {mem_after_model:.2f} GB")
    
    # Training loop
    model.train()
    start_time = time.time()
    
    n_iters = 5
    print(f"\nRunning {n_iters} training iterations...")
    
    for i in range(n_iters):
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass
        if use_amp:
            with torch.amp.autocast('cuda'):
                output = model(X)
                loss = criterion(output, y)
        else:
            output = model(X)
            loss = criterion(output, y)
        
        torch.cuda.synchronize()
        mem_after_forward = torch.cuda.memory_allocated() / 1e9
        
        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        torch.cuda.synchronize()
        mem_after_backward = torch.cuda.memory_allocated() / 1e9
        
        if i == 0:  # Print first iteration details
            print(f"\nIteration 1 memory breakdown:")
            print(f"  After forward:  {mem_after_forward:.2f} GB")
            print(f"  After backward: {mem_after_backward:.2f} GB")
            print(f"  Gradient memory: {mem_after_backward - mem_after_forward:.2f} GB")
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    # Get peak memory
    peak_memory = torch.cuda.max_memory_allocated() / 1e9
    training_time = end_time - start_time
    
    print(f"\n{'='*70}")
    print(f"RESULTS:")
    print(f"  Peak VRAM: {peak_memory:.2f} GB")
    print(f"  Training time: {training_time:.4f}s ({training_time/n_iters:.4f}s per iter)")
    print(f"  Throughput: {n_iters * batch_size / training_time:.1f} samples/sec")
    
    return {
        'peak_memory_gb': peak_memory,
        'training_time': training_time,
        'model_size_gb': model_size_gb,
        'total_params': total_params
    }


def main():
    """Run comprehensive VRAM benchmarks"""
    
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    print("\n" + "="*70)
    print("ACCURATE VRAM OPTIMIZATION BENCHMARK")
    print("="*70)
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Model configuration
    model_config = {
        'input_dim': 2048,
        'output_dim': 1024,
        'n_layers': 30,
        'hidden_dim': 4096
    }
    
    results = {}
    
    # 1. Baseline (no optimizations)
    results['Baseline'] = benchmark_configuration(
        "Baseline (FP32, No Checkpointing)",
        {**model_config, 'use_checkpointing': False},
        {'batch_size': 64, 'use_amp': False}
    )
    
    # 2. Gradient checkpointing only
    results['Checkpointing'] = benchmark_configuration(
        "Gradient Checkpointing Only",
        {**model_config, 'use_checkpointing': True},
        {'batch_size': 64, 'use_amp': False}
    )
    
    # 3. Mixed precision only
    results['Mixed Precision'] = benchmark_configuration(
        "Mixed Precision Only (FP16)",
        {**model_config, 'use_checkpointing': False},
        {'batch_size': 64, 'use_amp': True}
    )
    
    # 4. Both optimizations
    results['Full Optimization'] = benchmark_configuration(
        "Full Optimization (FP16 + Checkpointing)",
        {**model_config, 'use_checkpointing': True},
        {'batch_size': 128, 'use_amp': True}  # Can use larger batch!
    )
    
    # Print comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    
    baseline_mem = results['Baseline']['peak_memory_gb']
    baseline_time = results['Baseline']['training_time']
    
    print(f"\n{'Configuration':<30} {'Peak VRAM':<12} {'Saved':<12} {'Time':<10} {'Speedup':<10}")
    print("-" * 74)
    
    for name, result in results.items():
        mem_saved = baseline_mem - result['peak_memory_gb']
        mem_saved_pct = (mem_saved / baseline_mem * 100) if baseline_mem > 0 else 0
        speedup = baseline_time / result['training_time']
        
        print(f"{name:<30} {result['peak_memory_gb']:<12.2f} "
              f"{mem_saved:>5.2f} GB ({mem_saved_pct:>4.1f}%) "
              f"{result['training_time']:<10.4f} {speedup:<10.2f}x")
    
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    
    checkpointing_savings = baseline_mem - results['Checkpointing']['peak_memory_gb']
    mixed_precision_savings = baseline_mem - results['Mixed Precision']['peak_memory_gb']
    full_savings = baseline_mem - results['Full Optimization']['peak_memory_gb']
    
    print(f"\n1. Gradient Checkpointing:")
    print(f"   - Saves {checkpointing_savings:.2f} GB ({checkpointing_savings/baseline_mem*100:.1f}%)")
    print(f"   - Trades compute for memory (recomputes activations)")
    
    print(f"\n2. Mixed Precision (FP16):")
    print(f"   - Saves {mixed_precision_savings:.2f} GB ({mixed_precision_savings/baseline_mem*100:.1f}%)")
    print(f"   - Often FASTER due to Tensor Core acceleration")
    
    print(f"\n3. Combined Optimizations:")
    print(f"   - Saves {full_savings:.2f} GB ({full_savings/baseline_mem*100:.1f}%)")
    print(f"   - Allows {results['Full Optimization']['batch_size'] / results['Baseline']['batch_size']:.1f}x larger batch size")
    print(f"   - Best of both worlds!")
    
    print("\n" + "="*70)
    print("✅ BENCHMARK COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
