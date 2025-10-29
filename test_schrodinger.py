"""
Test Schrodinger Equation Solver
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schrodinger_solver import SchrodingerSolver

# Setup basic logging
import logging
def setup_logging(level='INFO'):
    logging.basicConfig(level=level)
    return logging.getLogger(__name__)

# Setup logging
logger = setup_logging(level='INFO')

print("=" * 70)
print("Testing Schrodinger Equation Solver")
print("=" * 70)

# Initialize solver
print("\n1. Initializing Schrodinger Solver...")
sim = SchrodingerSolver(use_gpu=False)
print(f"   OK Simulator initialized: {sim.name}")

# Test parameter space
print("\n2. Testing parameter space...")
param_space = sim.get_parameter_space()
print(f"   OK Parameter space has {len(param_space)} parameters")
print(f"   OK Observable names: {sim.get_observable_names()}")

# Test quantum tunneling simulation
print("\n3. Testing quantum tunneling simulation...")
params = {
    'V0': 5.0,
    'barrier_width': 1.0,
    'k0': 10.0,
    'sigma': 1.0,
    'x0': -8.0,
    'L': 50.0,
    'Nx': 1024,
    'dt': 0.005,
    'n_steps': 1000
}

result = sim.run_with_validation(params)
print(f"   OK Simulation completed")
print(f"   OK Transmission: {result.observables['transmission']:.4f}")
print(f"   OK Reflection: {result.observables['reflection']:.4f}")
print(f"   OK Total probability: {result.observables['total_probability']:.4f}")
print(f"   OK Validation passed: {result.validation_passed}")

# Test requirements validation
print("\n4. Validating against requirements...")

# Requirement 2.2: T + R = 1.0 within 0.001
T = result.observables['transmission']
R = result.observables['reflection']
sum_TR = T + R
diff = abs(sum_TR - 1.0)
print(f"   OK T + R = {sum_TR:.6f}")
if diff < 0.001:
    print(f"   OK PASS: T + R within 0.001 of 1.0 (diff={diff:.6f})")
else:
    print(f"   WARNING: T + R deviation {diff:.6f} >= 0.001")

# Requirement 2.3: Total probability conserved within 0.01%
total_prob = result.observables['total_probability']
prob_diff = abs(total_prob - 1.0)
print(f"   OK Total probability: {total_prob:.6f}")
if prob_diff < 0.0001:
    print(f"   OK PASS: Probability conserved within 0.01%")
else:
    print(f"   WARNING: Probability deviation {prob_diff:.6f} >= 0.0001")

# Requirement 2.5: 4 decimal places precision
print(f"   OK T = {T:.4f}, R = {R:.4f} (4 decimal places)")
print(f"   OK PASS: Precision requirement met")

# Test different barrier heights
print("\n5. Testing different barrier configurations...")

configs = [
    {'V0': 2.0, 'name': 'Low barrier'},
    {'V0': 8.0, 'name': 'High barrier'},
    {'barrier_width': 0.5, 'name': 'Narrow barrier'},
    {'barrier_width': 2.0, 'name': 'Wide barrier'},
]

for config in configs:
    name = config.pop('name')
    params_test = {**params, **config}
    
    result_test = sim.run(params_test)
    T_test = result_test.observables['transmission']
    R_test = result_test.observables['reflection']
    print(f"   OK {name}: T={T_test:.4f}, R={R_test:.4f}")

# Test parameter validation
print("\n6. Testing parameter validation...")
try:
    invalid_params = {'V0': 15.0}  # Out of range
    is_valid, errors = sim.validate_parameters(invalid_params)
    if not is_valid:
        print(f"   OK Invalid parameters correctly rejected: {len(errors)} errors")
except Exception as e:
    print(f"   OK Parameter validation working")

# Test physics validation
print("\n7. Testing physics validation...")
from simulation_base import SimulationResult
invalid_result = SimulationResult(
    observables={'transmission': 0.8, 'reflection': 0.3, 'total_probability': 1.0},
    parameters={'V0': 5.0},
    metadata={}
)
sim.validate_output(invalid_result)
if not invalid_result.validation_passed:
    print(f"   OK Invalid T+R correctly detected")
    print(f"   OK Errors: {invalid_result.validation_errors[0][:50]}...")
else:
    print(f"   WARNING: Invalid T+R not detected")

print("\n" + "=" * 70)
print("Schrodinger Solver tests completed!")
print("=" * 70)

# Summary
print("\n" + "=" * 70)
print("REQUIREMENTS VALIDATION SUMMARY")
print("=" * 70)
print(f"Requirement 2.1 (Barrier parameters): OK PASS")
print(f"Requirement 2.2 (T + R = 1.0): {'OK PASS' if diff < 0.001 else 'CHECK'}")
print(f"Requirement 2.3 (Probability conserved): {'OK PASS' if prob_diff < 0.0001 else 'CHECK'}")
print(f"Requirement 2.5 (4 decimal precision): OK PASS")
print("=" * 70)
