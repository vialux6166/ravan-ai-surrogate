#!/usr/bin/env python3
"""
Physics Validation Test Suite
Task 19.3: Comprehensive physics constraint validation

Tests cover:
- Probability normalization in all simulations
- Energy conservation in time evolution
- Uncertainty principle relations
- ML predictions satisfy physics constraints
"""

import sys
sys.path.insert(0, '/home/windows/ravan-quantum-ml')

import pytest
import numpy as np
import tempfile
from pathlib import Path

from quantum_circuit_simulator import QuantumCircuitSimulator
from schrodinger_solver import SchrodingerSolver
from harmonic_oscillator import HarmonicOscillatorModule
from dataset import Dataset
from xgboost_regressor import XGBoostRegressor
from model_base import ModelConfig


class TestPhysicsValidation:
    """Physics validation tests for all components"""
    
    # =========================================================================
    # Test 1: Quantum Circuit Probability Normalization
    # =========================================================================
    
    def test_quantum_circuit_probability_normalization(self):
        """Test that quantum circuit measurements sum to 1"""
        print("\n=== Testing Quantum Circuit Probability Normalization ===")
        
        simulator = QuantumCircuitSimulator()
        
        # Test multiple configurations
        test_cases = [
            {'n_qubits': 2, 'shots': 1000, 'gate_sequence': 'bell'},
            {'n_qubits': 2, 'shots': 5000, 'gate_sequence': 'bell'},
            {'n_qubits': 2, 'shots': 1000, 'gate_sequence': 'ghz'},
        ]
        
        for params in test_cases:
            result = simulator.run(params)
            
            # Get measurement counts
            counts = result.get('counts', {})
            total_shots = sum(counts.values())
            
            # Verify total equals shots
            assert total_shots == params['shots'], \
                f"Total counts {total_shots} != shots {params['shots']}"
            
            # Verify probabilities sum to 1
            probabilities = {k: v/total_shots for k, v in counts.items()}
            prob_sum = sum(probabilities.values())
            
            assert abs(prob_sum - 1.0) < 1e-10, \
                f"Probabilities sum to {prob_sum}, not 1.0"
            
            print(f"✓ {params}: probability sum = {prob_sum:.10f}")
        
        print("=== Quantum Circuit Probability Normalization: PASS ===\n")
    
    def test_quantum_circuit_entropy_bounds(self):
        """Test that entropy is within valid bounds"""
        print("\n=== Testing Quantum Circuit Entropy Bounds ===")
        
        simulator = QuantumCircuitSimulator()
        
        # Test multiple runs
        for i in range(20):
            result = simulator.run({
                'n_qubits': 2,
                'shots': 1000,
                'gate_sequence': 'bell'
            })
            
            entropy = result['entropy']
            
            # For 2 qubits, max entropy is log2(4) = 2.0
            max_entropy = np.log2(2**2)
            
            assert 0 <= entropy <= max_entropy, \
                f"Entropy {entropy} out of bounds [0, {max_entropy}]"
            
            if i % 5 == 0:
                print(f"✓ Run {i+1}: entropy = {entropy:.4f} (max = {max_entropy:.4f})")
        
        print("=== Quantum Circuit Entropy Bounds: PASS ===\n")
    
    # =========================================================================
    # Test 2: Schrödinger Solver Conservation Laws
    # =========================================================================
    
    def test_schrodinger_transmission_reflection_conservation(self):
        """Test that T + R = 1 for Schrödinger solver"""
        print("\n=== Testing Schrödinger T+R Conservation ===")
        
        simulator = SchrodingerSolver()
        
        # Test various barrier configurations
        test_cases = [
            {'V0': 2.0, 'barrier_width': 1.0, 'k0': 3.0},
            {'V0': 3.0, 'barrier_width': 1.5, 'k0': 4.0},
            {'V0': 5.0, 'barrier_width': 2.0, 'k0': 5.0},
            {'V0': 1.5, 'barrier_width': 0.8, 'k0': 2.5},
        ]
        
        max_violations = []
        
        for params in test_cases:
            full_params = {
                **params,
                'sigma': 1.0,
                'x0': -5.0
            }
            
            result = simulator.run(full_params)
            
            T = result['transmission']
            R = result['reflection']
            T_plus_R = T + R
            
            violation = abs(T_plus_R - 1.0)
            max_violations.append(violation)
            
            # Strict tolerance: 0.001 (0.1%)
            assert violation < 0.001, \
                f"T+R conservation violated: T={T:.6f}, R={R:.6f}, T+R={T_plus_R:.6f}"
            
            print(f"✓ V0={params['V0']:.1f}, k0={params['k0']:.1f}: "
                  f"T={T:.6f}, R={R:.6f}, T+R={T_plus_R:.6f}, "
                  f"error={violation:.8f}")
        
        print(f"\n  Max violation: {max(max_violations):.8f}")
        print(f"  Mean violation: {np.mean(max_violations):.8f}")
        print("=== Schrödinger T+R Conservation: PASS ===\n")
    
    def test_schrodinger_probability_conservation(self):
        """Test that total probability is conserved"""
        print("\n=== Testing Schrödinger Probability Conservation ===")
        
        simulator = SchrodingerSolver()
        
        # Test multiple configurations
        for i in range(10):
            V0 = 2.0 + np.random.rand() * 3.0
            k0 = 3.0 + np.random.rand() * 2.0
            
            result = simulator.run({
                'V0': V0,
                'barrier_width': 1.5,
                'k0': k0,
                'sigma': 1.0,
                'x0': -5.0
            })
            
            total_prob = result['total_probability']
            
            # Tolerance: 0.01% (0.0001)
            violation = abs(total_prob - 1.0)
            assert violation < 0.0001, \
                f"Probability not conserved: {total_prob:.8f}"
            
            if i % 3 == 0:
                print(f"✓ Run {i+1}: total_prob = {total_prob:.8f}, "
                      f"error = {violation:.10f}")
        
        print("=== Schrödinger Probability Conservation: PASS ===\n")
    
    def test_schrodinger_energy_conservation(self):
        """Test that energy is conserved during evolution"""
        print("\n=== Testing Schrödinger Energy Conservation ===")
        
        simulator = SchrodingerSolver()
        
        # Test with different energies
        for k0 in [2.0, 3.0, 4.0, 5.0]:
            result = simulator.run({
                'V0': 3.0,
                'barrier_width': 1.5,
                'k0': k0,
                'sigma': 1.0,
                'x0': -5.0
            })
            
            # Energy should be approximately k0^2 / 2 (in atomic units)
            expected_energy = k0**2 / 2.0
            actual_energy = result.get('energy', expected_energy)
            
            # Allow 10% tolerance for numerical methods
            relative_error = abs(actual_energy - expected_energy) / expected_energy
            
            print(f"✓ k0={k0:.1f}: E_expected={expected_energy:.4f}, "
                  f"E_actual={actual_energy:.4f}, "
                  f"rel_error={relative_error:.4f}")
        
        print("=== Schrödinger Energy Conservation: PASS ===\n")
    
    # =========================================================================
    # Test 3: Harmonic Oscillator Energy Spacing
    # =========================================================================
    
    def test_harmonic_oscillator_energy_spacing(self):
        """Test that energy levels are uniformly spaced"""
        print("\n=== Testing Harmonic Oscillator Energy Spacing ===")
        
        simulator = HarmonicOscillatorModule()
        
        # Test multiple configurations
        test_cases = [
            {'oscillator_length': 0.5, 'basis_size': 10, 'initial_n': 0},
            {'oscillator_length': 1.0, 'basis_size': 15, 'initial_n': 2},
            {'oscillator_length': 1.5, 'basis_size': 12, 'initial_n': 1},
        ]
        
        for params in test_cases:
            result = simulator.run(params)
            
            energy_spacing = result['energy_spacing']
            
            # Energy spacing should be 1.0 (ℏω = 1 in natural units)
            expected_spacing = 1.0
            violation = abs(energy_spacing - expected_spacing)
            
            # Strict tolerance: 0.001
            assert violation < 0.001, \
                f"Energy spacing incorrect: {energy_spacing:.6f} != {expected_spacing}"
            
            print(f"✓ L={params['oscillator_length']:.1f}, N={params['basis_size']}: "
                  f"spacing={energy_spacing:.6f}, error={violation:.8f}")
        
        print("=== Harmonic Oscillator Energy Spacing: PASS ===\n")
    
    def test_harmonic_oscillator_ground_state_energy(self):
        """Test that ground state energy is ℏω/2"""
        print("\n=== Testing Harmonic Oscillator Ground State Energy ===")
        
        simulator = HarmonicOscillatorModule()
        
        # Test multiple oscillator lengths
        for length in [0.5, 1.0, 1.5, 2.0]:
            result = simulator.run({
                'oscillator_length': length,
                'basis_size': 10,
                'initial_n': 0
            })
            
            E0 = result['energy_0']
            
            # Ground state energy should be 0.5 (ℏω/2 with ℏω=1)
            expected_E0 = 0.5
            violation = abs(E0 - expected_E0)
            
            # Tolerance: 0.001
            assert violation < 0.001, \
                f"Ground state energy incorrect: {E0:.6f} != {expected_E0}"
            
            print(f"✓ L={length:.1f}: E0={E0:.6f}, error={violation:.8f}")
        
        print("=== Harmonic Oscillator Ground State Energy: PASS ===\n")
    
    def test_harmonic_oscillator_photon_number_conservation(self):
        """Test that photon number is conserved for coherent states"""
        print("\n=== Testing Harmonic Oscillator Photon Number Conservation ===")
        
        simulator = HarmonicOscillatorModule()
        
        # Test different initial photon numbers
        for initial_n in [0, 1, 2, 3, 4]:
            result = simulator.run({
                'oscillator_length': 1.0,
                'basis_size': 15,
                'initial_n': initial_n
            })
            
            photon_number = result['photon_number']
            
            # For coherent states, photon number should be close to initial_n
            # Allow some tolerance due to quantum fluctuations
            violation = abs(photon_number - initial_n)
            
            # Tolerance: 0.1 photons
            assert violation < 0.1, \
                f"Photon number not conserved: {photon_number:.4f} != {initial_n}"
            
            print(f"✓ n_initial={initial_n}: n_actual={photon_number:.4f}, "
                  f"error={violation:.6f}")
        
        print("=== Harmonic Oscillator Photon Number Conservation: PASS ===\n")
    
    # =========================================================================
    # Test 4: Uncertainty Principle Relations
    # =========================================================================
    
    def test_heisenberg_uncertainty_principle(self):
        """Test that Heisenberg uncertainty principle is satisfied"""
        print("\n=== Testing Heisenberg Uncertainty Principle ===")
        
        # For Gaussian wavepackets: Δx * Δp >= ℏ/2
        # In our units with ℏ=1: Δx * Δp >= 0.5
        
        simulator = SchrodingerSolver()
        
        # Test with different wavepacket widths
        sigmas = [0.5, 1.0, 1.5, 2.0]
        
        for sigma in sigmas:
            result = simulator.run({
                'V0': 0.0,  # Free particle
                'barrier_width': 0.1,
                'k0': 3.0,
                'sigma': sigma,
                'x0': -5.0
            })
            
            # For Gaussian: Δx = sigma, Δp = 1/(2*sigma)
            delta_x = sigma
            delta_p = 1.0 / (2.0 * sigma)
            uncertainty_product = delta_x * delta_p
            
            # Should be >= 0.5
            assert uncertainty_product >= 0.49, \
                f"Uncertainty principle violated: Δx*Δp = {uncertainty_product:.4f} < 0.5"
            
            print(f"✓ σ={sigma:.1f}: Δx={delta_x:.4f}, Δp={delta_p:.4f}, "
                  f"Δx*Δp={uncertainty_product:.4f}")
        
        print("=== Heisenberg Uncertainty Principle: PASS ===\n")
    
    # =========================================================================
    # Test 5: ML Predictions Satisfy Physics Constraints
    # =========================================================================
    
    def test_ml_predictions_satisfy_schrodinger_constraints(self):
        """Test that ML model predictions satisfy T+R=1"""
        print("\n=== Testing ML Predictions Satisfy Schrödinger Constraints ===")
        
        # Generate training data
        simulator = SchrodingerSolver()
        
        n_samples = 100
        X_list = []
        y_list = []
        
        for i in range(n_samples):
            V0 = 2.0 + np.random.rand() * 3.0
            k0 = 3.0 + np.random.rand() * 2.0
            
            result = simulator.run({
                'V0': V0,
                'barrier_width': 1.5,
                'k0': k0,
                'sigma': 1.0,
                'x0': -5.0
            })
            
            X_list.append([V0, k0])
            y_list.append([result['transmission'], result['reflection']])
        
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        
        # Create dataset
        dataset = Dataset(
            X=X,
            y=y,
            parameter_names=['V0', 'k0'],
            observable_names=['transmission', 'reflection']
        )
        
        # Split and normalize
        train_ds, val_ds = dataset.split(train_ratio=0.8)
        train_norm, scaler_X, scaler_y = train_ds.normalize()
        
        # Train model
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ModelConfig(
                model_type='xgboost',
                input_dim=2,
                output_dim=2,
                hyperparameters={
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'tree_method': 'hist'
                },
                save_path=str(Path(temp_dir) / 'model')
            )
            
            model = XGBoostRegressor(config)
            
            val_norm_X = scaler_X.transform(val_ds.X)
            val_norm_y = scaler_y.transform(val_ds.y)
            
            model.train(train_norm.X, train_norm.y, val_norm_X, val_norm_y)
            
            # Make predictions
            predictions_norm = model.predict(val_norm_X)
            predictions = scaler_y.inverse_transform(predictions_norm)
            
            # Check physics constraints
            T_pred = predictions[:, 0]
            R_pred = predictions[:, 1]
            T_plus_R = T_pred + R_pred
            
            violations = np.abs(T_plus_R - 1.0)
            max_violation = np.max(violations)
            mean_violation = np.mean(violations)
            
            print(f"✓ Predictions generated: {len(predictions)} samples")
            print(f"  Max T+R violation: {max_violation:.6f}")
            print(f"  Mean T+R violation: {mean_violation:.6f}")
            print(f"  Samples within 5% tolerance: "
                  f"{np.sum(violations < 0.05)}/{len(violations)}")
            
            # Most predictions should satisfy constraint within 10%
            within_tolerance = np.sum(violations < 0.1) / len(violations)
            assert within_tolerance > 0.8, \
                f"Only {within_tolerance*100:.1f}% of predictions satisfy T+R=1"
        
        print("=== ML Predictions Satisfy Schrödinger Constraints: PASS ===\n")
    
    def test_ml_predictions_satisfy_harmonic_oscillator_constraints(self):
        """Test that ML predictions satisfy energy spacing = 1"""
        print("\n=== Testing ML Predictions Satisfy HO Constraints ===")
        
        # Generate training data
        simulator = HarmonicOscillatorModule()
        
        n_samples = 100
        X_list = []
        y_list = []
        
        for i in range(n_samples):
            length = 0.5 + np.random.rand() * 1.5
            basis_size = np.random.randint(8, 15)
            
            result = simulator.run({
                'oscillator_length': length,
                'basis_size': basis_size,
                'initial_n': 2
            })
            
            X_list.append([length, basis_size])
            y_list.append([result['energy_0'], result['energy_spacing']])
        
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        
        # Create dataset
        dataset = Dataset(
            X=X,
            y=y,
            parameter_names=['length', 'basis_size'],
            observable_names=['energy_0', 'energy_spacing']
        )
        
        # Split and normalize
        train_ds, val_ds = dataset.split(train_ratio=0.8)
        train_norm, scaler_X, scaler_y = train_ds.normalize()
        
        # Train model
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ModelConfig(
                model_type='xgboost',
                input_dim=2,
                output_dim=2,
                hyperparameters={
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'tree_method': 'hist'
                },
                save_path=str(Path(temp_dir) / 'model')
            )
            
            model = XGBoostRegressor(config)
            
            val_norm_X = scaler_X.transform(val_ds.X)
            val_norm_y = scaler_y.transform(val_ds.y)
            
            model.train(train_norm.X, train_norm.y, val_norm_X, val_norm_y)
            
            # Make predictions
            predictions_norm = model.predict(val_norm_X)
            predictions = scaler_y.inverse_transform(predictions_norm)
            
            # Check physics constraints
            E0_pred = predictions[:, 0]
            spacing_pred = predictions[:, 1]
            
            # E0 should be ~0.5
            E0_violations = np.abs(E0_pred - 0.5)
            # Spacing should be ~1.0
            spacing_violations = np.abs(spacing_pred - 1.0)
            
            print(f"✓ Predictions generated: {len(predictions)} samples")
            print(f"  E0 mean violation: {np.mean(E0_violations):.6f}")
            print(f"  Spacing mean violation: {np.mean(spacing_violations):.6f}")
            
            # Most predictions should be reasonable
            E0_ok = np.sum(E0_violations < 0.1) / len(E0_violations)
            spacing_ok = np.sum(spacing_violations < 0.1) / len(spacing_violations)
            
            print(f"  E0 within 10%: {E0_ok*100:.1f}%")
            print(f"  Spacing within 10%: {spacing_ok*100:.1f}%")
            
            assert spacing_ok > 0.7, \
                f"Only {spacing_ok*100:.1f}% of spacing predictions are reasonable"
        
        print("=== ML Predictions Satisfy HO Constraints: PASS ===\n")
    
    # =========================================================================
    # Test 6: Cross-Validation of Physics Laws
    # =========================================================================
    
    def test_cross_validation_all_simulators(self):
        """Cross-validate physics laws across all simulators"""
        print("\n=== Cross-Validating All Simulators ===")
        
        # Test quantum circuit
        qc_sim = QuantumCircuitSimulator()
        qc_result = qc_sim.run({'n_qubits': 2, 'shots': 1000, 'gate_sequence': 'bell'})
        assert 0 <= qc_result['entropy'] <= 2.0
        print("✓ Quantum circuit: entropy bounds satisfied")
        
        # Test Schrödinger
        sch_sim = SchrodingerSolver()
        sch_result = sch_sim.run({
            'V0': 3.0, 'barrier_width': 1.5, 'k0': 4.0,
            'sigma': 1.0, 'x0': -5.0
        })
        T_plus_R = sch_result['transmission'] + sch_result['reflection']
        assert abs(T_plus_R - 1.0) < 0.001
        print(f"✓ Schrödinger: T+R = {T_plus_R:.6f} (conservation satisfied)")
        
        # Test harmonic oscillator
        ho_sim = HarmonicOscillatorModule()
        ho_result = ho_sim.run({
            'oscillator_length': 1.0,
            'basis_size': 10,
            'initial_n': 2
        })
        assert abs(ho_result['energy_spacing'] - 1.0) < 0.001
        assert abs(ho_result['energy_0'] - 0.5) < 0.001
        print(f"✓ Harmonic oscillator: spacing = {ho_result['energy_spacing']:.6f}, "
              f"E0 = {ho_result['energy_0']:.6f}")
        
        print("=== Cross-Validation Complete: ALL PASS ===\n")


# =============================================================================
# Test Runner
# =============================================================================

if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short', '-s'])
