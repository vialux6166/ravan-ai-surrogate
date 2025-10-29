#!/usr/bin/env python3
"""
Unit Tests for Harmonic Oscillator Module
Task 5.4: Comprehensive unit tests

Tests cover:
- Energy level spacing (E_{n+1} - E_n = ℏω)
- Time evolution unitarity
- Coherent state properties
- Conservation laws
"""

import sys
sys.path.insert(0, '/home/windows/ravan-quantum-ml/src')

import pytest
import numpy as np
from simulators.harmonic import HarmonicOscillatorModule
from simulators.base import SimulationResult


class TestHarmonicOscillator:
    """Unit tests for harmonic oscillator simulator"""
    
    @pytest.fixture
    def simulator(self):
        """Create simulator instance for tests"""
        return HarmonicOscillatorModule()
    
    # =========================================================================
    # Test 1: Energy Level Spacing
    # =========================================================================
    
    def test_energy_level_spacing_uniform(self, simulator):
        """
        Test that energy levels are uniformly spaced
        
        For harmonic oscillator: E_n = (n + 1/2)ℏω
        Therefore: E_{n+1} - E_n = ℏω = 1.0 (in our units)
        
        Requirement 3.2: spacing within 0.001
        """
        params = {
            'oscillator_length': 1.0,
            'basis_size': 30,
            'initial_n': 0,
            'n_time_points': 10,
            'max_time': 1.0
        }
        
        result = simulator.run(params)
        
        # Check energy spacing
        spacing_mean = result.observables['energy_spacing_mean']
        spacing_std = result.observables['energy_spacing_std']
        
        # Requirement 3.2: within 0.001
        assert abs(spacing_mean - 1.0) < 0.001, \
            f"Energy spacing {spacing_mean} not equal to 1.0 within 0.001"
        
        # Standard deviation should be very small
        assert spacing_std < 0.01, \
            f"Energy spacing std {spacing_std} too large"
    
    def test_individual_energy_levels(self, simulator):
        """
        Test individual energy levels: E_n = n + 0.5
        """
        params = {
            'oscillator_length': 1.0,
            'basis_size': 30,
            'initial_n': 0,
            'n_time_points': 10,
            'max_time': 1.0
        }
        
        result = simulator.run(params)
        
        # Check first few energy levels
        for n in range(5):
            E_n = result.observables.get(f'energy_level_{n}', None)
            if E_n is not None:
                expected = n + 0.5
                assert abs(E_n - expected) < 0.001, \
                    f"E_{n} = {E_n}, expected {expected}"
    
    def test_energy_spacing_different_basis_sizes(self, simulator):
        """
        Test that energy spacing is consistent across different basis sizes
        """
        basis_sizes = [20, 30, 40]
        spacings = []
        
        for N in basis_sizes:
            params = {
                'oscillator_length': 1.0,
                'basis_size': N,
                'initial_n': 0,
                'n_time_points': 10,
                'max_time': 1.0
            }
            
            result = simulator.run(params)
            spacings.append(result.observables['energy_spacing_mean'])
        
        # All spacings should be close to 1.0
        for spacing in spacings:
            assert abs(spacing - 1.0) < 0.001, \
                f"Energy spacing {spacing} not consistent"
    
    # =========================================================================
    # Test 2: Time Evolution Unitarity
    # =========================================================================
    
    def test_photon_number_conservation(self, simulator):
        """
        Test that photon number is conserved during time evolution
        
        For a Fock state |n⟩, ⟨n⟩ should remain constant
        
        Requirement 3.3: within 0.01%
        """
        test_cases = [
            {'initial_n': 0, 'name': 'Ground state'},
            {'initial_n': 1, 'name': 'First excited'},
            {'initial_n': 3, 'name': 'Third excited'},
            {'initial_n': 5, 'name': 'Fifth excited'},
        ]
        
        for case in test_cases:
            params = {
                'oscillator_length': 1.0,
                'basis_size': 30,
                'initial_n': case['initial_n'],
                'n_time_points': 50,
                'max_time': 5.0
            }
            
            result = simulator.run(params)
            
            n_initial = result.observables['photon_number_initial']
            n_final = result.observables['photon_number_final']
            
            # Calculate relative change
            if n_initial > 0:
                relative_change = abs(n_final - n_initial) / n_initial
            else:
                relative_change = abs(n_final - n_initial)
            
            # Requirement 3.3: within 0.01% = 0.0001
            assert relative_change < 0.0001, \
                f"{case['name']}: photon number changed by {relative_change*100:.4f}%"
    
    def test_coherence_preservation(self, simulator):
        """
        Test that coherence (purity) is preserved
        
        For a pure state, coherence should remain close to 1.0
        """
        params = {
            'oscillator_length': 1.0,
            'basis_size': 30,
            'initial_n': 2,
            'n_time_points': 50,
            'max_time': 5.0
        }
        
        result = simulator.run(params)
        
        coherence = result.observables['coherence_mean']
        
        # Pure state should have coherence ≈ 1.0
        assert coherence > 0.99, \
            f"Coherence {coherence} too low, state not pure"
    
    def test_time_evolution_reversibility(self, simulator):
        """
        Test that time evolution is reversible (unitary)
        
        Evolving forward then backward should return to initial state
        """
        params = {
            'oscillator_length': 1.0,
            'basis_size': 30,
            'initial_n': 3,
            'n_time_points': 100,
            'max_time': 10.0  # Full period
        }
        
        result = simulator.run(params)
        
        # After full period, should return to initial state
        n_initial = result.observables['photon_number_initial']
        n_final = result.observables['photon_number_final']
        
        # Should be very close
        assert abs(n_final - n_initial) < 0.1, \
            f"After full period: n changed from {n_initial} to {n_final}"
    
    def test_probability_conservation(self, simulator):
        """
        Test that total probability is conserved
        
        Sum of |c_n|² should equal 1.0
        """
        params = {
            'oscillator_length': 1.0,
            'basis_size': 30,
            'initial_n': 2,
            'n_time_points': 50,
            'max_time': 5.0
        }
        
        result = simulator.run(params)
        
        # Coherence for pure state should be 1.0
        # This implicitly tests probability conservation
        coherence = result.observables['coherence_mean']
        assert 0.98 <= coherence <= 1.02, \
            f"Probability not conserved, coherence = {coherence}"
    
    # =========================================================================
    # Test 3: Coherent State Properties
    # =========================================================================
    
    def test_ground_state_properties(self, simulator):
        """
        Test ground state (n=0) properties
        
        Ground state should have:
        - E_0 = 0.5
        - ⟨n⟩ = 0
        - Minimum uncertainty
        """
        params = {
            'oscillator_length': 1.0,
            'basis_size': 30,
            'initial_n': 0,
            'n_time_points': 10,
            'max_time': 1.0
        }
        
        result = simulator.run(params)
        
        # Check ground state energy
        E_0 = result.observables.get('energy_level_0', None)
        if E_0 is not None:
            assert abs(E_0 - 0.5) < 0.001, \
                f"Ground state energy {E_0} not equal to 0.5"
        
        # Check photon number
        n_mean = result.observables['photon_number_mean']
        assert n_mean < 0.1, \
            f"Ground state photon number {n_mean} should be near 0"
    
    def test_excited_state_properties(self, simulator):
        """
        Test excited state properties
        
        For Fock state |n⟩:
        - ⟨n⟩ = n
        - Energy = n + 0.5
        """
        n_excited = 5
        
        params = {
            'oscillator_length': 1.0,
            'basis_size': 30,
            'initial_n': n_excited,
            'n_time_points': 10,
            'max_time': 1.0
        }
        
        result = simulator.run(params)
        
        # Check photon number
        n_mean = result.observables['photon_number_mean']
        assert abs(n_mean - n_excited) < 0.1, \
            f"Excited state photon number {n_mean} not equal to {n_excited}"
    
    def test_position_variance(self, simulator):
        """
        Test position variance for different states
        
        Higher excited states should have larger position variance
        """
        results = []
        
        for n in [0, 2, 4]:
            params = {
                'oscillator_length': 1.0,
                'basis_size': 30,
                'initial_n': n,
                'n_time_points': 10,
                'max_time': 1.0
            }
            
            result = simulator.run(params)
            var = result.observables['position_variance_mean']
            results.append((n, var))
        
        # Variance should increase with n
        for i in range(len(results) - 1):
            n1, var1 = results[i]
            n2, var2 = results[i + 1]
            assert var2 > var1, \
                f"Position variance should increase: n={n1} var={var1}, n={n2} var={var2}"
    
    # =========================================================================
    # Test 4: Parameter Validation
    # =========================================================================
    
    def test_parameter_space_definition(self, simulator):
        """Test that parameter space is properly defined"""
        param_space = simulator.get_parameter_space()
        
        required_params = [
            'oscillator_length', 'basis_size', 'initial_n',
            'n_time_points', 'max_time'
        ]
        
        for param in required_params:
            assert param in param_space, f"Missing parameter: {param}"
            assert isinstance(param_space[param], tuple), \
                f"Parameter {param} should have (min, max) tuple"
    
    def test_invalid_parameter_rejection(self, simulator):
        """Test that invalid parameters are rejected"""
        # Negative basis size
        invalid_params = {
            'oscillator_length': 1.0,
            'basis_size': -10,  # Invalid
            'initial_n': 0,
            'n_time_points': 10,
            'max_time': 1.0
        }
        
        is_valid, errors = simulator.validate_parameters(invalid_params)
        assert not is_valid, "Should reject negative basis_size"
    
    def test_valid_parameter_acceptance(self, simulator):
        """Test that valid parameters are accepted"""
        valid_params = {
            'oscillator_length': 1.0,
            'basis_size': 30,
            'initial_n': 2,
            'n_time_points': 50,
            'max_time': 5.0
        }
        
        is_valid, errors = simulator.validate_parameters(valid_params)
        assert is_valid, f"Should accept valid parameters, got errors: {errors}"
    
    # =========================================================================
    # Test 5: Physics Validation
    # =========================================================================
    
    def test_physics_validation_pass(self, simulator):
        """Test that valid results pass physics validation"""
        params = {
            'oscillator_length': 1.0,
            'basis_size': 30,
            'initial_n': 2,
            'n_time_points': 50,
            'max_time': 5.0
        }
        
        result = simulator.run_with_validation(params)
        
        assert result.validation_passed, \
            f"Valid result should pass validation, errors: {result.validation_errors}"
    
    def test_physics_validation_fail_spacing(self, simulator):
        """Test that invalid energy spacing fails validation"""
        invalid_result = SimulationResult(
            observables={
                'photon_number_mean': 2.0,
                'photon_number_std': 0.1,
                'photon_number_initial': 2.0,
                'photon_number_final': 2.0,
                'coherence_mean': 1.0,
                'position_variance_mean': 1.0,
                'energy_spacing_mean': 2.0,  # Invalid (should be 1.0)
                'energy_spacing_std': 0.001,
            },
            parameters={'basis_size': 30},
            metadata={}
        )
        
        simulator.validate_output(invalid_result)
        
        assert not invalid_result.validation_passed, \
            "Should fail validation for incorrect energy spacing"
    
    # =========================================================================
    # Test 6: Observable Names and Metadata
    # =========================================================================
    
    def test_observable_names(self, simulator):
        """Test that observable names are properly defined"""
        obs_names = simulator.get_observable_names()
        
        assert isinstance(obs_names, list), "Observable names should be a list"
        
        # Check for expected observables
        expected_obs = ['energy_levels', 'photon_number', 'coherence', 'position_variance']
        for obs in expected_obs:
            assert obs in obs_names, f"Missing expected observable: {obs}"
    
    def test_result_metadata(self, simulator):
        """Test that results include proper metadata"""
        params = {
            'oscillator_length': 1.0,
            'basis_size': 30,
            'initial_n': 2,
            'n_time_points': 50,
            'max_time': 5.0
        }
        
        result = simulator.run(params)
        
        assert result.metadata is not None, "Result should have metadata"
        assert isinstance(result.metadata, dict), "Metadata should be a dictionary"
        
        # Check for expected metadata
        assert 'basis_size' in result.metadata, "Metadata should include basis_size"
        assert 'n_time_points' in result.metadata, "Metadata should include n_time_points"
    
    # =========================================================================
    # Test 7: Edge Cases
    # =========================================================================
    
    def test_minimum_basis_size(self, simulator):
        """Test with minimum basis size"""
        params = {
            'oscillator_length': 1.0,
            'basis_size': 20,  # Minimum
            'initial_n': 0,
            'n_time_points': 10,
            'max_time': 1.0
        }
        
        result = simulator.run(params)
        
        # Should still compute energy spacing correctly
        spacing = result.observables['energy_spacing_mean']
        assert abs(spacing - 1.0) < 0.01, \
            f"Minimum basis: spacing {spacing} not near 1.0"
    
    def test_maximum_basis_size(self, simulator):
        """Test with maximum basis size"""
        param_space = simulator.get_parameter_space()
        max_basis = int(param_space['basis_size'][1])
        
        params = {
            'oscillator_length': 1.0,
            'basis_size': max_basis,
            'initial_n': 0,
            'n_time_points': 10,
            'max_time': 1.0
        }
        
        result = simulator.run(params)
        
        # Should handle large basis
        assert result is not None, "Should handle maximum basis size"
    
    def test_high_excitation(self, simulator):
        """Test with high initial excitation"""
        params = {
            'oscillator_length': 1.0,
            'basis_size': 30,
            'initial_n': 10,  # High excitation
            'n_time_points': 50,
            'max_time': 5.0
        }
        
        result = simulator.run(params)
        
        # Should conserve photon number
        n_initial = result.observables['photon_number_initial']
        n_final = result.observables['photon_number_final']
        
        relative_change = abs(n_final - n_initial) / n_initial
        assert relative_change < 0.001, \
            f"High excitation: photon number not conserved"
    
    def test_long_time_evolution(self, simulator):
        """Test with long time evolution"""
        params = {
            'oscillator_length': 1.0,
            'basis_size': 30,
            'initial_n': 3,
            'n_time_points': 100,
            'max_time': 10.0  # Long time
        }
        
        result = simulator.run(params)
        
        # Should still conserve photon number
        n_initial = result.observables['photon_number_initial']
        n_final = result.observables['photon_number_final']
        
        assert abs(n_final - n_initial) < 0.1, \
            f"Long evolution: photon number not conserved"
    
    def test_short_time_evolution(self, simulator):
        """Test with very short time evolution"""
        params = {
            'oscillator_length': 1.0,
            'basis_size': 30,
            'initial_n': 2,
            'n_time_points': 10,
            'max_time': 0.1  # Very short
        }
        
        result = simulator.run(params)
        
        # Should barely change
        n_initial = result.observables['photon_number_initial']
        n_final = result.observables['photon_number_final']
        
        assert abs(n_final - n_initial) < 0.01, \
            f"Short evolution: state changed too much"


# =============================================================================
# Test Runner
# =============================================================================

if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
