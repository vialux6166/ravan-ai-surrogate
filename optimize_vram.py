#!/usr/bin/env python3
"""
VRAM Optimization for Ravan Quantum-ML System
Task 20.3: Optimize VRAM usage

Implements VRAM optimizations:
- Gradient checkpointing
- In-place operations
- Aggressive cache clearing
- Memory profiling and monitoring
"""

import torch
import torch.nn as nn
import numpy as np
import gc
import time
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional
from contextlib import contextmanager

# Add current directory to path
sys.path.insert(0, '.')


class VRAMMonitor:
    """
    VRAM usage monitoring and profiling
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.snapshots = []
        
        if self.device.type != 'cuda':
            print("Warning: CUDA not available, VRAM monitoring disabled")
    
    def get_vram_usage(self) -> Dict[str, float]:
        """Get current VRAM usage in GB"""
        if self.device.type != 'cuda':
            return {'allocated': 0, 'cached': 0, 'total': 0}
        
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        return {
            'allocated': allocated,
            'cached': cached,
            'total': total,
            'free': total - cached
        }
    
    def snapshot(self, label: str = ""):
        """Take a VRAM usage snapshot"""
        if self.device.type != 'cuda':
            return
        
        usage = self.get_vram_usage()
        usage['label'] = label
        usage['timestamp'] = time.time()
        self.snapshots.append(usage)
    
    def print_usage(self, label: str = ""):
        """Print current VRAM usage"""
        if self.device.type != 'cuda':
            print(f"{label}: CUDA not available")
            return
        
        usage = self.get_vram_usage()
        print(f"{label}: Allocated={usage['allocated']:.2f}GB, "
              f"Cached={usage['cached']:.2f}GB, "
              f"Free={usage['free']:.2f}GB")
    
    def clear_cache(self):
        """Clear VRAM cache"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
    
    def reset_peak_stats(self):
        """Reset peak memory statistics"""
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
    
    def get_peak_usage(self) -> float:
        """Get peak VRAM usage in GB"""
        if self.device.type != 'cuda':
            return 0
        return torch.cuda.max_memory_allocated() / 1e9
    
    def generate_report(self) -> str:
        """Generate VRAM usage report"""
        if not self.snapshots:
            return "No VRAM snapshots recorded"
        
        report = ["VRAM Usage Report", "=" * 50, ""]
        
        for i, snapshot in enumerate(self.snapshots):
            report.append(f"{i+1}. {snapshot['label']}:")
            report.append(f"   Allocated: {snapshot['allocated']:.2f} GB")
            report.append(f"   Cached: {snapshot['cached']:.2f} GB")
            report.append(f"   Free: {snapshot['free']:.2f} GB")
            report.append("")
        
        # Peak usage
        peak = self.get_peak_usage()
        report.append(f"Peak Usage: {peak:.2f} GB")
        
        return "\n".join(report)


class MemoryEfficientModel(nn.Module):
    """
    Memory-efficient model with gradient checkpointing
    """
    
    def __init__(self, input_dim: int, output_dim: int, 
                 hidden_dims: List[int] = None,
                 use_checkpointing: bool = True):
        super().__init__()
        
        # Default to very large model for stress testing
        if hidden_dims is None:
            hidden_dims = [4096] * 20  # 20 layers of 4096 neurons each
        
        self.use_checkpointing = use_checkpointing
        
        # Create layers - VERY large for meaningful VRAM testing
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layer = nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),  # In-place to save memory
                nn.Dropout(0.1, inplace=False)  # Dropout can't be in-place
            )
            self.layers.append(layer)
            prev_dim = hidden_dim
        
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        # Initialize weights efficiently
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Memory-efficient weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use in-place initialization to save memory
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpointing and self.training:
            # Use gradient checkpointing to trade compute for memory
            for layer in self.layers:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
        else:
            # Standard forward pass
            for layer in self.layers:
                x = layer(x)
        
        return self.output_layer(x)


class VRAMOptimizer:
    """
    VRAM optimization utilities
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.monitor = VRAMMonitor()
    
    @contextmanager
    def vram_context(self, label: str = ""):
        """Context manager for VRAM monitoring"""
        self.monitor.snapshot(f"{label} - Start")
        self.monitor.clear_cache()
        
        try:
            yield
        finally:
            self.monitor.snapshot(f"{label} - End")
            self.monitor.clear_cache()
    
    def optimize_model_memory(self, model: nn.Module) -> nn.Module:
        """Apply memory optimizations to a model"""
        print("\nApplying memory optimizations...")
        
        # Convert BatchNorm to more memory-efficient alternatives
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                # Replace with GroupNorm (more memory efficient)
                num_channels = module.num_features
                model._modules[name] = nn.GroupNorm(32, num_channels)
                print(f"  Replaced BatchNorm2d with GroupNorm in {name}")
        
        # Enable gradient checkpointing if available
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            print("  ✓ Enabled gradient checkpointing")
        
        return model
    
    def benchmark_memory_strategies(self, input_dim: int = 2048, output_dim: int = 1024):
        """Benchmark different memory optimization strategies"""
        print("\n" + "="*70)
        print("BENCHMARKING MEMORY OPTIMIZATION STRATEGIES")
        print("="*70)
        print("\nUsing VERY LARGE model for meaningful VRAM measurements:")
        print(f"  Input dim: {input_dim}")
        print(f"  Output dim: {output_dim}")
        print(f"  Hidden layers: 20 x 4096 neurons")
        print(f"  Total parameters: ~350M")
        print(f"  Model size: ~1.4 GB (FP32)")
        
        strategies = [
            {
                'name': 'Baseline',
                'checkpointing': False,
                'mixed_precision': False,
                'batch_size': 128,  # Large batch to consume significant VRAM
                'hidden_dims': [4096] * 20
            },
            {
                'name': 'Gradient Checkpointing',
                'checkpointing': True,
                'mixed_precision': False,
                'batch_size': 128,
                'hidden_dims': [4096] * 20
            },
            {
                'name': 'Mixed Precision',
                'checkpointing': False,
                'mixed_precision': True,
                'batch_size': 128,
                'hidden_dims': [4096] * 20
            },
            {
                'name': 'Full Optimization',
                'checkpointing': True,
                'mixed_precision': True,
                'batch_size': 256,  # Can use larger batch with optimizations
                'hidden_dims': [4096] * 20
            }
        ]
        
        results = {}
        baseline_vram = None
        
        for strategy in strategies:
            print(f"\nTesting: {strategy['name']}")
            print("-" * 50)
            
            try:
                result = self._benchmark_strategy(input_dim, output_dim, strategy)
                
                # Store baseline for comparison
                if strategy['name'] == 'Baseline':
                    baseline_vram = result['peak_vram']
                
                # Calculate memory saved compared to baseline
                if baseline_vram is not None:
                    result['memory_saved'] = max(0, baseline_vram - result['peak_vram'])
                    result['memory_saved_pct'] = (result['memory_saved'] / baseline_vram * 100) if baseline_vram > 0 else 0
                else:
                    result['memory_saved'] = 0
                    result['memory_saved_pct'] = 0
                
                results[strategy['name']] = result
                
                print(f"  ✓ Peak VRAM: {result['peak_vram']:.2f} GB")
                print(f"  ✓ Training time: {result['training_time']:.4f}s")
                if baseline_vram is not None and strategy['name'] != 'Baseline':
                    print(f"  ✓ Memory saved: {result['memory_saved']:.2f} GB ({result['memory_saved_pct']:.1f}%)")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                import traceback
                traceback.print_exc()
                results[strategy['name']] = {'error': str(e)}
        
        self._print_memory_comparison(results, baseline_vram)
        return results
    
    def _benchmark_strategy(self, input_dim: int, output_dim: int, 
                           strategy: Dict) -> Dict:
        """Benchmark a single memory optimization strategy"""
        
        if self.device.type != 'cuda':
            raise RuntimeError("CUDA not available for benchmarking")
        
        # Clear VRAM
        self.monitor.clear_cache()
        self.monitor.reset_peak_stats()
        
        print(f"  Creating model with {len(strategy['hidden_dims'])} layers...")
        
        # Create model
        model = MemoryEfficientModel(
            input_dim, output_dim,
            hidden_dims=strategy['hidden_dims'],
            use_checkpointing=strategy['checkpointing']
        )
        model = model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
        
        # Create dummy data
        batch_size = strategy['batch_size']
        print(f"  Batch size: {batch_size}")
        X = torch.randn(batch_size, input_dim, device=self.device)
        y = torch.randn(batch_size, output_dim, device=self.device)
        
        # Setup training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Mixed precision setup (updated syntax)
        scaler = None
        if strategy['mixed_precision']:
            scaler = torch.amp.GradScaler('cuda')
            print(f"  Mixed precision: ENABLED")
        
        # Training loop
        model.train()
        start_time = time.time()
        
        print(f"  Running 10 training iterations...")
        for i in range(10):  # 10 iterations for benchmarking
            optimizer.zero_grad(set_to_none=True)  # More memory efficient
            
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    output = model(X)
                    loss = criterion(output, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(X)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
            
            # Print progress every 3 iterations
            if (i + 1) % 3 == 0:
                current_vram = torch.cuda.memory_allocated() / 1e9
                print(f"    Iteration {i+1}/10 - VRAM: {current_vram:.2f} GB")
        
        end_time = time.time()
        
        # Get memory stats
        peak_vram = self.monitor.get_peak_usage()
        training_time = end_time - start_time
        
        return {
            'peak_vram': peak_vram,
            'training_time': training_time,
            'total_params': total_params
        }
    
    def _print_memory_comparison(self, results: Dict, baseline_vram: Optional[float] = None):
        """Print memory optimization comparison"""
        print("\n" + "="*70)
        print("MEMORY OPTIMIZATION COMPARISON")
        print("="*70)
        
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            print("No valid results to compare")
            return
        
        print(f"{'Strategy':<25} {'Peak VRAM':<12} {'Time (s)':<10} {'Saved':<12} {'Saved %':<10}")
        print("-" * 72)
        
        for name, result in valid_results.items():
            saved_str = f"{result.get('memory_saved', 0):.2f} GB"
            saved_pct_str = f"{result.get('memory_saved_pct', 0):.1f}%"
            
            print(f"{name:<25} {result['peak_vram']:<12.2f} "
                  f"{result['training_time']:<10.4f} {saved_str:<12} {saved_pct_str:<10}")
        
        # Print insights
        if baseline_vram and baseline_vram > 1.0:
            print("\n" + "="*70)
            print("KEY INSIGHTS")
            print("="*70)
            print(f"\nBaseline VRAM usage: {baseline_vram:.2f} GB")
            
            for name, result in valid_results.items():
                if name != 'Baseline' and 'memory_saved' in result:
                    print(f"\n{name}:")
                    print(f"  - Reduces VRAM by {result['memory_saved']:.2f} GB ({result['memory_saved_pct']:.1f}%)")
                    speedup = valid_results['Baseline']['training_time'] / result['training_time']
                    if speedup > 1.1:
                        print(f"  - {speedup:.2f}x FASTER than baseline")
                    elif speedup < 0.9:
                        slowdown = result['training_time'] / valid_results['Baseline']['training_time']
                        print(f"  - {slowdown:.2f}x slower than baseline (trade-off for memory)")
        
        # Save results
        self._save_optimization_results(results)
    
    def _save_optimization_results(self, results: Dict):
        """Save optimization results to file"""
        output_dir = Path("performance_reports")
        output_dir.mkdir(exist_ok=True)
        
        results_file = output_dir / "vram_optimization_results.txt"
        
        with open(results_file, 'w') as f:
            f.write("VRAM Optimization Results\n")
            f.write("=" * 50 + "\n\n")
            
            for name, result in results.items():
                f.write(f"{name}:\n")
                if 'error' in result:
                    f.write(f"  Error: {result['error']}\n")
                else:
                    f.write(f"  Peak VRAM: {result['peak_vram']:.2f} GB\n")
                    f.write(f"  Training time: {result['training_time']:.4f}s\n")
                    f.write(f"  Memory saved: {result['memory_saved']:.2f} GB\n")
                f.write("\n")
            
            f.write("\nOptimization Recommendations:\n")
            f.write("1. Always use gradient checkpointing for deep models\n")
            f.write("2. Enable mixed precision training when possible\n")
            f.write("3. Use set_to_none=True when zeroing gradients\n")
            f.write("4. Clear cache between major operations\n")
            f.write("5. Use in-place operations where safe\n")
        
        print(f"\n✓ Optimization results saved to {results_file}")


def demonstrate_vram_optimizations():
    """Demonstrate VRAM optimization techniques"""
    print("\n" + "="*70)
    print("VRAM OPTIMIZATION DEMONSTRATION")
    print("="*70)
    
    optimizer = VRAMOptimizer()
    
    if optimizer.device.type != 'cuda':
        print("\n⚠ CUDA not available. Skipping VRAM optimization demo.")
        print("These optimizations are most beneficial on GPU.")
        return
    
    # Demonstrate context manager
    print("\n1. VRAM Context Manager:")
    with optimizer.vram_context("Model Training"):
        model = MemoryEfficientModel(100, 10).to(optimizer.device)
        X = torch.randn(32, 100, device=optimizer.device)
        y = model(X)
    
    print(optimizer.monitor.generate_report())
    
    # Benchmark strategies
    print("\n2. Memory Optimization Strategies:")
    results = optimizer.benchmark_memory_strategies()
    
    # Provide recommendations
    print("\n" + "="*70)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("="*70)
    print("""
1. GRADIENT CHECKPOINTING:
   - Reduces memory by ~40-50% for deep networks
   - Increases training time by ~20-30%
   - Essential for training very deep models

2. MIXED PRECISION (FP16):
   - Reduces memory by ~50%
   - Can increase training speed by 2-3x
   - Requires careful loss scaling

3. IN-PLACE OPERATIONS:
   - Use inplace=True for activations (ReLU, etc.)
   - Use set_to_none=True when zeroing gradients
   - Saves memory without performance cost

4. AGGRESSIVE CACHE CLEARING:
   - Call torch.cuda.empty_cache() between operations
   - Use gc.collect() for Python garbage collection
   - Essential for switching between workloads

5. BATCH SIZE TUNING:
   - Start with small batch size and increase
   - Monitor VRAM usage to find optimal size
   - Larger batches with optimizations = better throughput
    """)


def main():
    """Main VRAM optimization workflow"""
    print("\n" + "="*70)
    print("RAVAN VRAM OPTIMIZATION SUITE")
    print("="*70)
    
    demonstrate_vram_optimizations()
    
    print("\n" + "="*70)
    print("VRAM OPTIMIZATION COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. Gradient checkpointing trades compute for memory")
    print("2. Mixed precision can halve memory usage")
    print("3. In-place operations are free memory savings")
    print("4. Aggressive cache clearing prevents fragmentation")
    print("5. Combine all techniques for maximum efficiency")


if __name__ == '__main__':
    main()
