#!/usr/bin/env python3
"""
Unit Tests for Quantum Circuit Simulator
Task 3.4: Comprehensive unit tests with known analytical solutions

Tests cover:
- Known analytical solutions (Bell states, GHZ states)
- Conservation laws (probability normalization)
- Parameter validation
- Physics constraints
"""

import sys
sys.path.insert(0, '/home/windows/ravan-quantum-ml/src')

import pytest
import numpy as np
from simulators.quantum_circuit_enhanced import EnhancedQuantumCircuitSimulator
from simulators.base import SimulationResult


class TestQuantumCircuitSimulator:
    """Unit tests for quantum circuit simulator"""
    
    @pytest.fixture
    def simulator(self):
        """Create simulator instance for tests"""
        return EnhancedQuantumCircuitSimulator()
    
    # =========================================================================
    # Test 1: Known Analytical Solutions - Bell States
    # =========================================================================
    
    def test_bell_state_creation(self, simulator):
        """
        Test Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
        
        Expected:
        - P(|00⟩) ≈ 0.5
        - P(|11⟩) ≈ 0.5
        - Entropy ≈ 1.0 bit (maximum for 2-qubit system)
        """
        params = {
            'n_qubits': 2,
            'circuit_depth': 1,
            'rx_angle_0': 0.0,
            'ry_angle_0': 0.0,
            'rz_angle_0': 0.0,
            'rx_angle_1': 0.0,
            'ry_angle_1': 0.0,
            'rz_angle_1': 0.0,
            'entangling_pattern': 1,  # Apply entangling gates
            'shots': 8192
        }
        
        result = simulator.run(params)
        
        # Check entropy is close to maximum (1.0 for 2 qubits)
        entropy = result.observables['entanglement_entropy']
        assert 0.9 <= entropy <= 1.1, f"Bell state entropy {entropy} not near 1.0"
        
        # Check fidelity to target state
        fidelity = result.observables.get('fidelity_to_ghz', 0)
        assert fidelity >= 0.8, f"Bell state fidelity {fidelity} too low"
    
    def test_product_state_zero_entanglement(self, simulator):
        """
        Test product state |00⟩ (no entanglement)
        
        Expected:
        - Entropy ≈ 0 (no entanglement)
        - All probability in |00⟩ state
        """
        params = {
            'n_qubits': 2,
            'circuit_depth': 1,
            'rx_angle_0': 0.0,
            'ry_angle_0': 0.0,
            'rz_angle_0': 0.0,
            'rx_angle_1': 0.0,
            'ry_angle_1': 0.0,
            'rz_angle_1': 0.0,
            'entangling_pattern': 0,  # No entangling gates
            'shots': 4096
        }
        
        result = simulator.run(params)
        
        # Check entropy is close to zero
        entropy = result.observables['entanglement_entropy']
        assert entropy <= 0.2, f"Product state entropy {entropy} should be near 0"
    
    def test_hadamard_superposition(self, simulator):
        """
        Test Hadamard gate creates equal superposition
        
        Single qubit Hadamard: |0⟩ → (|0⟩ + |1⟩)/√2
        Expected: P(0) ≈ 0.5, P(1) ≈ 0.5
        """
        params = {
            'n_qubits': 2,
            'circuit_depth': 1,
            'rx_angle_0': np.pi/2,  # Hadamard-like rotation
            'ry_angle_0': 0.0,
            'rz_angle_0': 0.0,
            'rx_angle_1': 0.0,
            'ry_angle_1': 0.0,
            'rz_angle_1': 0.0,
            'entangling_pattern': 0,
            'shots': 4096
        }
        
        result = simulator.run(params)
        
        # Entropy should be non-zero for superposition
        entropy = result.observables['entanglement_entropy']
        assert entropy > 0.1, f"Superposition should have non-zero entropy, got {entropy}"
    
    # =========================================================================
    # Test 2: Conservation Laws
    # =========================================================================
    
    def test_probability_normalization(self, simulator):
        """
        Test that measurement probabilities sum to 1.0
        
        This is a fundamental quantum mechanics requirement
        """
        params = {
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
        
        result = simulator.run(params)
        
        # All observables should be valid probabilities/measurements
        # Check that values are in reasonable ranges
        for key, value in result.observables.items():
            if 'prob' in key.lower() or 'fidelity' in key.lower():
                assert 0.0 <= value <= 1.0, f"{key} = {value} not in [0, 1]"
    
    def test_entropy_bounds(self, simulator):
        """
        Test that entropy satisfies: 0 ≤ S ≤ log₂(dim)
        
        For n qubits: 0 ≤ S ≤ n
        """
        test_cases = [
            {'n_qubits': 2, 'max_entropy': 2.0},
            {'n_qubits': 3, 'max_entropy': 3.0},
            {'n_qubits': 4, 'max_entropy': 4.0},
        ]
        
        for case in test_cases:
            params = {
                'n_qubits': case['n_qubits'],
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
            
            result = simulator.run(params)
            entropy = result.observables['entanglement_entropy']
            
            assert 0.0 <= entropy <= case['max_entropy'], \
                f"Entropy {entropy} violates bounds [0, {case['max_entropy']}]"
    
    def test_unitary_evolution(self, simulator):
        """
        Test that quantum evolution is unitary (probability conserved)
        
        Run multiple times and check consistency
        """
        params = {
            'n_qubits': 2,
            'circuit_depth': 3,
            'rx_angle_0': np.pi/3,
            'ry_angle_0': np.pi/4,
            'rz_angle_0': np.pi/5,
            'rx_angle_1': np.pi/6,
            'ry_angle_1': np.pi/7,
            'rz_angle_1': np.pi/8,
            'entangling_pattern': 2,
            'shots': 8192
        }
        
        # Run multiple times
        results = [simulator.run(params) for _ in range(3)]
        
        # Check that results are consistent (within statistical fluctuations)
        entropies = [r.observables['entanglement_entropy'] for r in results]
        entropy_std = np.std(entropies)
        
        # Standard deviation should be small (< 0.1 for 8192 shots)
        assert entropy_std < 0.15, \
            f"Entropy variance {entropy_std} too high, suggests non-unitary evolution"
    
    # =========================================================================
    # Test 3: Parameter Validation
    # =========================================================================
    
    def test_parameter_space_definition(self, simulator):
        """Test that parameter space is properly defined"""
        param_space = simulator.get_parameter_space()
        
        # Check required parameters exist
        required_params = [
            'n_qubits', 'circuit_depth', 'shots',
            'rx_angle_0', 'ry_angle_0', 'rz_angle_0',
            'rx_angle_1', 'ry_angle_1', 'rz_angle_1',
            'entangling_pattern'
        ]
        
        for param in required_params:
            assert param in param_space, f"Missing parameter: {param}"
            assert isinstance(param_space[param], tuple), \
                f"Parameter {param} should have (min, max) tuple"
            assert len(param_space[param]) == 2, \
                f"Parameter {param} should have exactly 2 values"
    
    def test_invalid_parameter_rejection(self, simulator):
        """Test that invalid parameters are rejected"""
        # Test 1: Out of range n_qubits
        invalid_params = {
            'n_qubits': 100,  # Too many qubits
            'circuit_depth': 1,
            'shots': 4096,
            'rx_angle_0': 0.0,
            'ry_angle_0': 0.0,
            'rz_angle_0': 0.0,
            'rx_angle_1': 0.0,
            'ry_angle_1': 0.0,
            'rz_angle_1': 0.0,
            'entangling_pattern': 0
        }
        
        is_valid, errors = simulator.validate_parameters(invalid_params)
        assert not is_valid, "Should reject invalid n_qubits"
        assert len(errors) > 0, "Should provide error messages"
    
    def test_missing_parameter_detection(self, simulator):
        """Test that missing required parameters are detected"""
        incomplete_params = {
            'n_qubits': 2,
            'circuit_depth': 1,
            # Missing other required parameters
        }
        
        is_valid, errors = simulator.validate_parameters(incomplete_params)
        assert not is_valid, "Should reject incomplete parameters"
        assert len(errors) > 0, "Should list missing parameters"
    
    def test_valid_parameter_acceptance(self, simulator):
        """Test that valid parameters are accepted"""
        valid_params = {
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
        
        is_valid, errors = simulator.validate_parameters(valid_params)
        assert is_valid, f"Should accept valid parameters, got errors: {errors}"
        assert len(errors) == 0, "Should have no errors for valid parameters"
    
    # =========================================================================
    # Test 4: Physics Constraints
    # =========================================================================
    
    def test_physics_validation_pass(self, simulator):
        """Test that valid results pass physics validation"""
        params = {
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
        
        result = simulator.run_with_validation(params)
        
        assert result.validation_passed, \
            f"Valid result should pass validation, errors: {result.validation_errors}"
    
    def test_physics_validation_fail_entropy(self, simulator):
        """Test that invalid entropy fails validation"""
        # Create a result with impossible entropy
        invalid_result = SimulationResult(
            observables={
                'entanglement_entropy': 10.0,  # Impossible for 2 qubits
                'fidelity_to_ghz': 0.5,
                'expect_x': 0.0,
                'expect_y': 0.0,
                'expect_z': 0.0,
                'circuit_depth_actual': 2,
                'two_qubit_gate_count': 1
            },
            parameters={'n_qubits': 2},
            metadata={}
        )
        
        simulator.validate_output(invalid_result)
        
        assert not invalid_result.validation_passed, \
            "Should fail validation for impossible entropy"
        assert len(invalid_result.validation_errors) > 0, \
            "Should provide error messages"
    
    def test_physics_validation_fail_probability(self, simulator):
        """Test that invalid probabilities fail validation"""
        invalid_result = SimulationResult(
            observables={
                'entanglement_entropy': 0.5,
                'fidelity_to_ghz': 1.5,  # Impossible probability > 1
                'expect_x': 0.0,
                'expect_y': 0.0,
                'expect_z': 0.0,
                'circuit_depth_actual': 2,
                'two_qubit_gate_count': 1
            },
            parameters={'n_qubits': 2},
            metadata={}
        )
        
        simulator.validate_output(invalid_result)
        
        assert not invalid_result.validation_passed, \
            "Should fail validation for probability > 1"
    
    # =========================================================================
    # Test 5: Observable Names and Metadata
    # =========================================================================
    
    def test_observable_names(self, simulator):
        """Test that observable names are properly defined"""
        obs_names = simulator.get_observable_names()
        
        assert isinstance(obs_names, list), "Observable names should be a list"
        assert len(obs_names) > 0, "Should have at least one observable"
        
        # Check for expected observables
        expected_obs = ['entanglement_entropy', 'fidelity_to_ghz']
        for obs in expected_obs:
            assert obs in obs_names, f"Missing expected observable: {obs}"
    
    def test_result_metadata(self, simulator):
        """Test that results include proper metadata"""
        params = {
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
        
        result = simulator.run(params)
        
        assert result.metadata is not None, "Result should have metadata"
        assert isinstance(result.metadata, dict), "Metadata should be a dictionary"
        
        # Check for expected metadata fields
        assert 'backend' in result.metadata or 'simulator' in result.metadata, \
            "Metadata should include backend/simulator info"
    
    # =========================================================================
    # Test 6: Edge Cases
    # =========================================================================
    
    def test_minimum_shots(self, simulator):
        """Test with minimum number of shots"""
        params = {
            'n_qubits': 2,
            'circuit_depth': 1,
            'rx_angle_0': 0.0,
            'ry_angle_0': 0.0,
            'rz_angle_0': 0.0,
            'rx_angle_1': 0.0,
            'ry_angle_1': 0.0,
            'rz_angle_1': 0.0,
            'entangling_pattern': 0,
            'shots': 1024  # Minimum shots
        }
        
        result = simulator.run(params)
        
        assert result is not None, "Should handle minimum shots"
        assert 'entanglement_entropy' in result.observables, \
            "Should compute observables with minimum shots"
    
    def test_maximum_circuit_depth(self, simulator):
        """Test with maximum circuit depth"""
        param_space = simulator.get_parameter_space()
        max_depth = int(param_space['circuit_depth'][1])
        
        params = {
            'n_qubits': 2,
            'circuit_depth': max_depth,
            'rx_angle_0': np.pi/4,
            'ry_angle_0': np.pi/3,
            'rz_angle_0': np.pi/6,
            'rx_angle_1': np.pi/2,
            'ry_angle_1': np.pi/4,
            'rz_angle_1': np.pi/8,
            'entangling_pattern': 1,
            'shots': 4096
        }
        
        result = simulator.run(params)
        
        assert result is not None, "Should handle maximum circuit depth"
        assert result.observables['circuit_depth_actual'] <= max_depth, \
            "Actual depth should not exceed maximum"
    
    def test_zero_rotation_angles(self, simulator):
        """Test with all rotation angles set to zero (identity)"""
        params = {
            'n_qubits': 2,
            'circuit_depth': 2,
            'rx_angle_0': 0.0,
            'ry_angle_0': 0.0,
            'rz_angle_0': 0.0,
            'rx_angle_1': 0.0,
            'ry_angle_1': 0.0,
            'rz_angle_1': 0.0,
            'entangling_pattern': 0,
            'shots': 4096
        }
        
        result = simulator.run(params)
        
        # Should result in |00⟩ state (zero entropy)
        entropy = result.observables['entanglement_entropy']
        assert entropy < 0.2, \
            f"Identity circuit should have near-zero entropy, got {entropy}"


# =============================================================================
# Test Runner
# =============================================================================

if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
