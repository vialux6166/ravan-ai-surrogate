#!/usr/bin/env python3
"""Benchmark GPU vs CPU for Schrödinger solver"""

import sys
sys.path.insert(0, '.')

from schrodinger_solver import SchrodingerSolver
import time
import numpy as np

params = {
    'V0': 5.0,
    'barrier_width': 1.5,
    'k0': 4.5,
    'sigma': 1.0,
    'x0': -5.0
}

print("="*70)
print("SCHRÖDINGER SOLVER: GPU vs CPU BENCHMARK")
print("="*70)

# Test different grid sizes
grid_sizes = [512, 1024, 2048, 4096]

for nx in grid_sizes:
    params['Nx'] = nx
    
    print(f"\n--- Grid Size: {nx} ---")
    
    # CPU test
    solver_cpu = SchrodingerSolver(use_gpu=False)
    result = solver_cpu.run(params)  # Warm-up
    
    start = time.time()
    n_runs = 50
    for _ in range(n_runs):
        result = solver_cpu.run(params)
    cpu_time = (time.time() - start) / n_runs
    
    print(f"CPU: {cpu_time*1000:.2f}ms per run ({n_runs/((time.time()-start)):.2f} sims/s)")
    
    # GPU test
    try:
        solver_gpu = SchrodingerSolver(use_gpu=True)
        if solver_gpu.use_gpu:
            result = solver_gpu.run(params)  # Warm-up
            
            # Synchronize GPU
            import cupy as cp
            cp.cuda.Stream.null.synchronize()
            
            start = time.time()
            for _ in range(n_runs):
                result = solver_gpu.run(params)
            cp.cuda.Stream.null.synchronize()
            gpu_time = (time.time() - start) / n_runs
            
            print(f"GPU: {gpu_time*1000:.2f}ms per run ({n_runs/((time.time()-start)):.2f} sims/s)")
            print(f"Speedup: {cpu_time/gpu_time:.2f}x")
        else:
            print("GPU: Not available")
    except Exception as e:
        print(f"GPU: Error - {e}")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)
print("\nFor dataset generation (10k samples):")
print("- If GPU speedup > 2x: Use GPU")
print("- If GPU speedup < 2x: Use CPU (less overhead)")
print("\nCurrent default: GPU enabled")
