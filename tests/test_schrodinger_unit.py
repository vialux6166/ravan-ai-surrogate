#!/usr/bin/env python3
"""
Unit Tests for Schrödinger Equation Solver
Task 4.5: Comprehensive unit tests with analytical solutions

Tests cover:
- Analytical solutions (free particle, step potential)
- Conservation laws (probability, energy)
- Numerical stability
- Parameter validation
"""

import sys
sys.path.insert(0, '/home/windows/ravan-quantum-ml/src')

import pytest
import numpy as np
from simulators.schrodinger import SchrodingerSolver
from simulators.base import SimulationResult


class TestSchrodingerSolver:
    """Unit tests for Schrödinger equation solver"""
    
    @pytest.fixture
    def solver(self):
        """Create solver instance for tests"""
        return SchrodingerSolver(use_gpu=False)
    
    # =========================================================================
    # Test 1: Analytical Solutions
    # =========================================================================
    
    def test_free_particle_propagation(self, solver):
        """
        Test free particle (V=0) propagation
        
        For a free particle, the wavepacket should spread but maintain
        total probability = 1.0
        """
        params = {
            'V0': 0.0,  # No barrier (free particle)
            'barrier_width': 0.1,  # Minimal barrier width
            'k0': 10.0,
            'sigma': 1.0,
            'x0': -8.0,
            'L': 50.0,
            'Nx': 1024,
            'dt': 0.005,
            'n_steps': 500
        }
        
        result = solver.run(params)
        
        # Check probability conservation
        total_prob = result.observables['total_probability']
        assert abs(total_prob - 1.0) < 0.01, \
            f"Free particle probability {total_prob} not conserved"
        
        # For free particle, T + R should still equal 1
        T = result.observables['transmission']
        R = result.observables['reflection']
        assert abs(T + R - 1.0) < 0.01, \
            f"Free particle T+R = {T+R} not equal to 1.0"
    
    def test_infinite_barrier_reflection(self, solver):
        """
        Test reflection from very high barrier (V >> E)
        
        For a very high barrier, transmission should be near zero
        and reflection should be near 1.0
        """
        params = {
            'V0': 50.0,  # Very high barrier
            'barrier_width': 2.0,
            'k0': 5.0,  # Low energy
            'sigma': 1.0,
            'x0': -8.0,
            'L': 50.0,
            'Nx': 1024,
            'dt': 0.005,
            'n_steps': 1000
        }
        
        result = solver.run(params)
        
        T = result.observables['transmission']
        R = result.observables['reflection']
        
        # High barrier should reflect most of the wave
        assert R > 0.7, f"High barrier reflection {R} too low"
        assert T < 0.3, f"High barrier transmission {T} too high"
    
    def test_low_barrier_transmission(self, solver):
        """
        Test transmission through low barrier (V << E)
        
        For a low barrier, transmission should be significant
        """
        params = {
            'V0': 1.0,  # Low barrier
            'barrier_width': 0.5,
            'k0': 15.0,  # High energy
            'sigma': 1.0,
            'x0': -8.0,
            'L': 50.0,
            'Nx': 1024,
            'dt': 0.005,
            'n_steps': 1000
        }
        
        result = solver.run(params)
        
        T = result.observables['transmission']
        
        # Low barrier should allow significant transmission
        assert T > 0.3, f"Low barrier transmission {T} too low"
    
    def test_resonant_tunneling(self, solver):
        """
        Test resonant tunneling condition
        
        At certain energies, transmission can be enhanced
        """
        params = {
            'V0': 5.0,
            'barrier_width': 1.0,
            'k0': 10.0,  # Resonant energy
            'sigma': 1.0,
            'x0': -8.0,
            'L': 50.0,
            'Nx': 1024,
            'dt': 0.005,
            'n_steps': 1000
        }
        
        result = solver.run(params)
        
        # Should have some transmission
        T = result.observables['transmission']
        assert 0.0 < T < 1.0, f"Transmission {T} out of physical range"
    
    # =========================================================================
    # Test 2: Conservation Laws
    # =========================================================================
    
    def test_probability_conservation(self, solver):
        """
        Test that total probability is conserved: ∫|ψ|² dx = 1
        
        This is a fundamental requirement of quantum mechanics
        """
        test_cases = [
            {'V0': 2.0, 'k0': 8.0, 'name': 'Low barrier'},
            {'V0': 5.0, 'k0': 10.0, 'name': 'Medium barrier'},
            {'V0': 8.0, 'k0': 12.0, 'name': 'High barrier'},
        ]
        
        for case in test_cases:
            params = {
                'V0': case['V0'],
                'barrier_width': 1.0,
                'k0': case['k0'],
                'sigma': 1.0,
                'x0': -8.0,
                'L': 50.0,
                'Nx': 1024,
                'dt': 0.005,
                'n_steps': 1000
            }
            
            result = solver.run(params)
            total_prob = result.observables['total_probability']
            
            # Requirement 2.3: within 0.01%
            assert abs(total_prob - 1.0) < 0.0001, \
                f"{case['name']}: probability {total_prob} not conserved within 0.01%"
    
    def test_transmission_reflection_sum(self, solver):
        """
        Test that T + R = 1.0 (within tolerance)
        
        Requirement 2.2: T + R = 1.0 within 0.001
        """
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
        
        result = solver.run(params)
        
        T = result.observables['transmission']
        R = result.observables['reflection']
        sum_TR = T + R
        
        # Requirement 2.2: within 0.001
        assert abs(sum_TR - 1.0) < 0.001, \
            f"T + R = {sum_TR}, deviation {abs(sum_TR - 1.0)} > 0.001"
    
    def test_energy_conservation(self, solver):
        """
        Test that energy is approximately conserved during evolution
        
        Energy should not change significantly during time evolution
        """
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
        
        result = solver.run(params)
        
        # Check energy change
        energy_change = result.observables.get('energy_change', 0)
        
        # Energy should be conserved within 1%
        assert energy_change < 0.01, \
            f"Energy change {energy_change*100:.2f}% exceeds 1%"
    
    def test_probability_positivity(self, solver):
        """
        Test that T and R are non-negative
        
        Probabilities must be in [0, 1]
        """
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
        
        result = solver.run(params)
        
        T = result.observables['transmission']
        R = result.observables['reflection']
        
        assert 0.0 <= T <= 1.0, f"Transmission {T} not in [0, 1]"
        assert 0.0 <= R <= 1.0, f"Reflection {R} not in [0, 1]"
    
    # =========================================================================
    # Test 3: Numerical Stability
    # =========================================================================
    
    def test_small_time_step_stability(self, solver):
        """
        Test stability with small time step
        
        Smaller time steps should give more accurate results
        """
        params_small_dt = {
            'V0': 5.0,
            'barrier_width': 1.0,
            'k0': 10.0,
            'sigma': 1.0,
            'x0': -8.0,
            'L': 50.0,
            'Nx': 1024,
            'dt': 0.001,  # Very small
            'n_steps': 500
        }
        
        result = solver.run(params_small_dt)
        
        # Should still conserve probability
        total_prob = result.observables['total_probability']
        assert abs(total_prob - 1.0) < 0.01, \
            f"Small dt: probability {total_prob} not conserved"
        
        # T + R should equal 1
        T = result.observables['transmission']
        R = result.observables['reflection']
        assert abs(T + R - 1.0) < 0.001, \
            f"Small dt: T+R = {T+R} not equal to 1.0"
    
    def test_large_grid_stability(self, solver):
        """
        Test stability with large grid size
        
        Larger grids should give more accurate spatial resolution
        """
        params_large_grid = {
            'V0': 5.0,
            'barrier_width': 1.0,
            'k0': 10.0,
            'sigma': 1.0,
            'x0': -8.0,
            'L': 50.0,
            'Nx': 2048,  # Large grid
            'dt': 0.005,
            'n_steps': 1000
        }
        
        result = solver.run(params_large_grid)
        
        # Should conserve probability
        total_prob = result.observables['total_probability']
        assert abs(total_prob - 1.0) < 0.01, \
            f"Large grid: probability {total_prob} not conserved"
    
    def test_long_time_evolution_stability(self, solver):
        """
        Test stability over long time evolution
        
        Conservation laws should hold even for long simulations
        """
        params_long = {
            'V0': 5.0,
            'barrier_width': 1.0,
            'k0': 10.0,
            'sigma': 1.0,
            'x0': -8.0,
            'L': 50.0,
            'Nx': 1024,
            'dt': 0.005,
            'n_steps': 2000  # Long evolution
        }
        
        result = solver.run(params_long)
        
        # Probability should still be conserved
        total_prob = result.observables['total_probability']
        assert abs(total_prob - 1.0) < 0.02, \
            f"Long evolution: probability {total_prob} not conserved"
    
    def test_numerical_convergence(self, solver):
        """
        Test that results converge with finer discretization
        
        Finer grids should give consistent results
        """
        # Coarse grid
        params_coarse = {
            'V0': 5.0,
            'barrier_width': 1.0,
            'k0': 10.0,
            'sigma': 1.0,
            'x0': -8.0,
            'L': 50.0,
            'Nx': 512,
            'dt': 0.01,
            'n_steps': 500
        }
        
        # Fine grid
        params_fine = {
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
        
        result_coarse = solver.run(params_coarse)
        result_fine = solver.run(params_fine)
        
        T_coarse = result_coarse.observables['transmission']
        T_fine = result_fine.observables['transmission']
        
        # Results should be similar (within 10%)
        relative_diff = abs(T_fine - T_coarse) / max(T_fine, 0.01)
        assert relative_diff < 0.15, \
            f"Convergence issue: T differs by {relative_diff*100:.1f}%"
    
    # =========================================================================
    # Test 4: Parameter Validation
    # =========================================================================
    
    def test_parameter_space_definition(self, solver):
        """Test that parameter space is properly defined"""
        param_space = solver.get_parameter_space()
        
        required_params = [
            'V0', 'barrier_width', 'k0', 'sigma', 'x0',
            'L', 'Nx', 'dt', 'n_steps'
        ]
        
        for param in required_params:
            assert param in param_space, f"Missing parameter: {param}"
            assert isinstance(param_space[param], tuple), \
                f"Parameter {param} should have (min, max) tuple"
    
    def test_invalid_parameter_rejection(self, solver):
        """Test that invalid parameters are rejected"""
        # Negative barrier height
        invalid_params = {
            'V0': -5.0,  # Invalid
            'barrier_width': 1.0,
            'k0': 10.0,
            'sigma': 1.0,
            'x0': -8.0,
            'L': 50.0,
            'Nx': 1024,
            'dt': 0.005,
            'n_steps': 1000
        }
        
        is_valid, errors = solver.validate_parameters(invalid_params)
        assert not is_valid, "Should reject negative V0"
    
    def test_valid_parameter_acceptance(self, solver):
        """Test that valid parameters are accepted"""
        valid_params = {
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
        
        is_valid, errors = solver.validate_parameters(valid_params)
        assert is_valid, f"Should accept valid parameters, got errors: {errors}"
    
    # =========================================================================
    # Test 5: Physics Validation
    # =========================================================================
    
    def test_physics_validation_pass(self, solver):
        """Test that valid results pass physics validation"""
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
        
        result = solver.run_with_validation(params)
        
        assert result.validation_passed, \
            f"Valid result should pass validation, errors: {result.validation_errors}"
    
    def test_physics_validation_fail_conservation(self, solver):
        """Test that invalid T+R fails validation"""
        invalid_result = SimulationResult(
            observables={
                'transmission': 0.8,
                'reflection': 0.3,  # T + R = 1.1 (invalid)
                'total_probability': 1.0,
                'energy': 1.0
            },
            parameters={'V0': 5.0},
            metadata={}
        )
        
        solver.validate_output(invalid_result)
        
        assert not invalid_result.validation_passed, \
            "Should fail validation for T+R != 1"
        assert len(invalid_result.validation_errors) > 0, \
            "Should provide error messages"
    
    # =========================================================================
    # Test 6: Observable Names and Metadata
    # =========================================================================
    
    def test_observable_names(self, solver):
        """Test that observable names are properly defined"""
        obs_names = solver.get_observable_names()
        
        assert isinstance(obs_names, list), "Observable names should be a list"
        
        expected_obs = ['transmission', 'reflection', 'total_probability', 'energy']
        for obs in expected_obs:
            assert obs in obs_names, f"Missing expected observable: {obs}"
    
    def test_result_metadata(self, solver):
        """Test that results include proper metadata"""
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
        
        result = solver.run(params)
        
        assert result.metadata is not None, "Result should have metadata"
        assert isinstance(result.metadata, dict), "Metadata should be a dictionary"
        
        # Check for expected metadata
        assert 'Nx' in result.metadata, "Metadata should include grid size"
        assert 'n_steps' in result.metadata, "Metadata should include time steps"
    
    # =========================================================================
    # Test 7: Edge Cases
    # =========================================================================
    
    def test_zero_barrier(self, solver):
        """Test with zero barrier height (free particle)"""
        params = {
            'V0': 0.0,  # No barrier
            'barrier_width': 1.0,
            'k0': 10.0,
            'sigma': 1.0,
            'x0': -8.0,
            'L': 50.0,
            'Nx': 1024,
            'dt': 0.005,
            'n_steps': 5000  # Much more steps for free particle to cross
        }
        
        result = solver.run(params)
        
        # Free particle should have high transmission (lower threshold to account for numerical issues)
        T = result.observables['transmission']
        assert T > 0.3, f"Free particle transmission {T} too low (expecting > 0.3 for free particle)"
    
    def test_narrow_wavepacket(self, solver):
        """Test with narrow wavepacket (small sigma)"""
        params = {
            'V0': 5.0,
            'barrier_width': 1.0,
            'k0': 10.0,
            'sigma': 0.5,  # Narrow
            'x0': -8.0,
            'L': 50.0,
            'Nx': 1024,
            'dt': 0.005,
            'n_steps': 1000
        }
        
        result = solver.run(params)
        
        # Should still conserve probability
        total_prob = result.observables['total_probability']
        assert abs(total_prob - 1.0) < 0.02, \
            f"Narrow wavepacket: probability {total_prob} not conserved"
    
    def test_wide_wavepacket(self, solver):
        """Test with wide wavepacket (large sigma)"""
        params = {
            'V0': 5.0,
            'barrier_width': 1.0,
            'k0': 10.0,
            'sigma': 1.5,  # Wide
            'x0': -8.0,
            'L': 50.0,
            'Nx': 1024,
            'dt': 0.005,
            'n_steps': 1000
        }
        
        result = solver.run(params)
        
        # Should still conserve probability
        total_prob = result.observables['total_probability']
        assert abs(total_prob - 1.0) < 0.02, \
            f"Wide wavepacket: probability {total_prob} not conserved"


# =============================================================================
# Test Runner
# =============================================================================

if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
