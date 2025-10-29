"""
Quick test of adaptive sampling with small dataset
"""

import sys
sys.path.insert(0, '/home/windows/ravan-quantum-ml/src')

import numpy as np
from adaptive_sampling import AdaptiveSampler
from simulators.quantum_circuit_enhanced import EnhancedQuantumCircuitSimulator

print("=" * 80)
print("Quick Adaptive Sampling Test")
print("=" * 80)

# Initialize simulator
print("\n1. Initializing quantum circuit simulator...")
simulator = EnhancedQuantumCircuitSimulator()
parameter_space = simulator.get_parameter_space()

print(f"Parameter space: {list(parameter_space.keys())}")
print(f"Ranges: {parameter_space}")

# Create adaptive sampler with small parameters for quick test
print("\n2. Creating adaptive sampler...")
sampler = AdaptiveSampler(
    simulator=simulator,
    initial_samples=100,  # Small initial dataset
    samples_per_iteration=50,  # Add 50 per iteration
    n_iterations=2,  # Just 2 iterations for quick test
    candidate_pool_size=500,  # Small candidate pool
    mc_samples=20  # Fewer MC samples for speed
)

print(f"Configuration:")
print(f"  Initial samples: 100")
print(f"  Samples per iteration: 50")
print(f"  Iterations: 2")
print(f"  Candidate pool: 500")
print(f"  MC samples: 20")

# Run adaptive sampling
print("\n3. Running adaptive sampling...")
try:
    dataset, metrics = sampler.run(
        parameter_space=parameter_space,
        output_dir='./data/adaptive_quick_test'
    )
    
    print("\n" + "=" * 80)
    print("Adaptive Sampling Test Complete!")
    print("=" * 80)
    
    print(f"\nFinal Results:")
    print(f"  Dataset size: {dataset.n_samples}")
    print(f"  Features: {dataset.n_features}")
    print(f"  Targets: {dataset.n_targets}")
    
    print(f"\nMetrics History:")
    for i, iteration in enumerate(metrics['iteration']):
        print(f"  Iteration {iteration}:")
        print(f"    Samples: {metrics['n_samples'][i]}")
        print(f"    R²: {metrics['r2_score'][i]:.4f}")
        print(f"    MSE: {metrics['mse'][i]:.6f}")
        print(f"    Mean Uncertainty: {metrics['mean_uncertainty'][i]:.6f}")
    
    print(f"\nImprovement:")
    initial_r2 = metrics['r2_score'][0]
    final_r2 = metrics['r2_score'][-1]
    improvement = final_r2 - initial_r2
    print(f"  Initial R²: {initial_r2:.4f}")
    print(f"  Final R²: {final_r2:.4f}")
    print(f"  Improvement: {improvement:+.4f}")
    
    if improvement > 0:
        print(f"  ✓ Adaptive sampling improved model performance!")
    else:
        print(f"  ⚠ No improvement (may need more iterations or samples)")
    
except Exception as e:
    print(f"\n✗ Error during adaptive sampling: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
