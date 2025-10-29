#!/usr/bin/env python3
"""Test enhanced quantum circuit simulator"""

import sys
sys.path.insert(0, '/home/windows/ravan-quantum-ml/src')

import numpy as np
from simulators.quantum_circuit_enhanced import EnhancedQuantumCircuitSimulator
from utils.logging import setup_logging

logger = setup_logging(level='INFO')

logger.info("="*60)
logger.info("TESTING ENHANCED QUANTUM CIRCUIT SIMULATOR")
logger.info("="*60)

# Initialize simulator
sim = EnhancedQuantumCircuitSimulator()

# Test with diverse parameters
test_cases = [
    {
        'name': 'Small circuit, no rotation',
        'params': {
            'n_qubits': 2,
            'circuit_depth': 1,
            'rx_angle_0': 0.0,
            'ry_angle_0': 0.0,
            'rz_angle_0': 0.0,
            'rx_angle_1': 0.0,
            'ry_angle_1': 0.0,
            'rz_angle_1': 0.0,
            'entangling_pattern': 0,
            'shots': 4096
        }
    },
    {
        'name': 'With rotation angles',
        'params': {
            'n_qubits': 2,
            'circuit_depth': 2,
            'rx_angle_0': np.pi/4,
            'ry_angle_0': np.pi/3,
            'rz_angle_0': np.pi/6,
            'rx_angle_1': np.pi/2,
            'ry_angle_1': np.pi/4,
            'rz_angle_1': np.pi/8,
            'entangling_pattern': 1,
            'shots': 4096
        }
    },
    {
        'name': 'Larger system',
        'params': {
            'n_qubits': 4,
            'circuit_depth': 3,
            'rx_angle_0': np.pi,
            'ry_angle_0': np.pi/2,
            'rz_angle_0': 0.0,
            'rx_angle_1': np.pi/3,
            'ry_angle_1': np.pi/6,
            'rz_angle_1': np.pi/4,
            'entangling_pattern': 2,
            'shots': 4096
        }
    }
]

results = []
for test_case in test_cases:
    logger.info(f"\nTest: {test_case['name']}")
    logger.info("-" * 40)
    
    result = sim.run(test_case['params'])
    results.append(result)
    
    logger.info(f"Observables:")
    for key, value in result.observables.items():
        logger.info(f"  {key}: {value:.6f}")
    
    # Validate
    is_valid, errors = sim.validate_output(result)
    if is_valid:
        logger.info("✓ Validation passed")
    else:
        logger.warning(f"✗ Validation failed: {errors}")

# Check variance across results
logger.info("\n" + "="*60)
logger.info("VARIANCE ANALYSIS")
logger.info("="*60)

observable_names = results[0].observables.keys()
for obs_name in observable_names:
    values = [r.observables[obs_name] for r in results]
    variance = np.var(values)
    logger.info(f"{obs_name}: variance={variance:.6f}, range=[{min(values):.3f}, {max(values):.3f}]")
    
    if variance < 0.001:
        logger.warning(f"  ⚠ Low variance for {obs_name}")
    else:
        logger.info(f"  ✓ Good variance for {obs_name}")

logger.info("\n" + "="*60)
logger.info("TEST COMPLETE")
logger.info("="*60)
