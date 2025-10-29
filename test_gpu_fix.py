#!/usr/bin/env python3
"""Quick test to verify GPU acceleration is working"""

import sys
sys.path.insert(0, '.')

from schrodinger_solver import SchrodingerSolver
import time

# Test with GPU
solver = SchrodingerSolver(use_gpu=True)
print(f"GPU enabled: {solver.use_gpu}")
print(f"Using: {solver.xp.__name__}")

params = {
    'V0': 5.0,
    'barrier_width': 1.5,
    'k0': 4.5,
    'sigma': 1.0,
    'x0': -5.0
}

# Run once to warm up
result = solver.run(params)
print(f"Backend: {result.metadata['backend']}")
print(f"T={result.observables['transmission']:.4f}, R={result.observables['reflection']:.4f}")

# Time 100 runs
print("\nTiming 100 runs...")
start = time.time()
for _ in range(100):
    result = solver.run(params)
elapsed = time.time() - start

print(f"Total time: {elapsed:.2f}s")
print(f"Average time: {elapsed/100*1000:.2f}ms")
print(f"Throughput: {100/elapsed:.2f} sims/s")
