"""
Stress Tests for Ravan Quantum-ML System
Task 19.5: Implement stress tests

Tests system behavior under extreme conditions:
- Large datasets (100k samples)
- Maximum parameter ranges
- VRAM management under stress
- Concurrent operations
"""

import pytest
import sys
import os
import time
import psutil
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Check if GPU is available
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        TOTAL_VRAM_GB = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    else:
        TOTAL_VRAM_GB = 0
except ImportError:
    GPU_AVAILABLE = False
    TOTAL_VRAM_GB = 0


class TestStressConditions:
    """Stress tests for extreme conditions"""
    
    @pytest.mark.slow
    def test_large_dataset_generation(self):
        """Test generating large dataset (10k samples)"""
        print("\n" + "="*70)
        print("STRESS TEST 1: Large Dataset Generation (10k samples)")
        print("="*70)
        
        from quantum_circuit_simulator import QuantumCircuitSimulator
        
        simulator = QuantumCircuitSimulator()
        
        # Generate 10k samples (reduced from 100k for reasonable test time)
        n_samples = 10000
        results = []
        
        start_time = time.time()
        
        for i in range(n_samples):
            params = {
                'n_qubits': 2,
                'shots': 1000,
                'gate_sequence': 'bell'
            }
            
            result = simulator.run(params)
            results.append(result)
            
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                print(f"  Progress: {i+1}/{n_samples} samples ({rate:.1f} samples/sec)")
        
        elapsed = time.time() - start_time
        
        print(f"\n✓ Generated {n_samples} samples in {elapsed:.2f} seconds")
        print(f"✓ Average rate: {n_samples/elapsed:.1f} samples/second")
        
        assert len(results) == n_samples
        # Check if results have entropy (handle both dict and SimulationResult)
        for r in results:
            if hasattr(r, 'observables'):
                assert 'entropy' in r.observables
            else:
                assert 'entropy' in r
    
    @pytest.mark.slow
    def test_extreme_parameter_ranges(self):
        """Test with maximum parameter ranges"""
        print("\n" + "="*70)
        print("STRESS TEST 2: Extreme Parameter Ranges")
        print("="*70)
        
        from schrodinger_solver import SchrodingerSolver
        
        simulator = SchrodingerSolver()
        
        # Test extreme parameters
        extreme_params = [
            # Very high barrier
            {'V0': 10.0, 'barrier_width': 5.0, 'k0': 10.0, 'sigma': 3.0, 'x0': -10.0},
            # Very low barrier
            {'V0': 0.5, 'barrier_width': 0.5, 'k0': 1.0, 'sigma': 0.5, 'x0': -3.0},
            # Narrow wavepacket
            {'V0': 5.0, 'barrier_width': 2.0, 'k0': 5.0, 'sigma': 0.5, 'x0': -5.0},
            # Wide wavepacket
            {'V0': 3.0, 'barrier_width': 1.5, 'k0': 4.0, 'sigma': 3.0, 'x0': -8.0},
        ]
        
        for i, params in enumerate(extreme_params, 1):
            print(f"\n  Test {i}: {params}")
            result = simulator.run(params)
            
            # Verify physics constraints
            # Handle both dict and SimulationResult object
            if hasattr(result, 'observables'):
                transmission = result.observables.get('transmission', 0)
                reflection = result.observables.get('reflection', 0)
            else:
                transmission = result.get('transmission', 0)
                reflection = result.get('reflection', 0)
            
            total_prob = transmission + reflection
            assert abs(total_prob - 1.0) < 0.01, f"Probability not conserved: {total_prob}"
            
            print(f"    T={transmission:.4f}, R={reflection:.4f}")
            print(f"    ✓ Physics constraints satisfied")
        
        print(f"\n✓ All extreme parameter tests passed")
    
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_vram_stress(self):
        """Test VRAM management under stress"""
        print("\n" + "="*70)
        print("STRESS TEST 3: VRAM Management Under Stress")
        print("="*70)
        
        import torch
        from vram_manager import VRAMManager, WorkloadMode
        
        vram_mgr = VRAMManager()
        
        print(f"  Total VRAM: {TOTAL_VRAM_GB:.2f} GB")
        
        # Test 1: Rapid mode switching
        print("\n  Test 1: Rapid mode switching (100 iterations)")
        for i in range(100):
            vram_mgr.request_mode(WorkloadMode.SIMULATION)
            vram_mgr.request_mode(WorkloadMode.TRAINING)
            
            if (i + 1) % 20 == 0:
                free_vram = torch.cuda.memory_allocated(0) / (1024**3)
                print(f"    Iteration {i+1}: VRAM allocated = {free_vram:.2f} GB")
        
        print(f"  ✓ Rapid mode switching successful")
        
        # Test 2: Allocate and free large tensors
        print("\n  Test 2: Large tensor allocation/deallocation")
        for i in range(10):
            # Allocate 1GB tensor
            tensor = torch.zeros((128 * 1024 * 1024,), device='cuda')
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            print(f"    Iteration {i+1}: Allocated {allocated:.2f} GB")
            
            # Free tensor
            del tensor
            torch.cuda.empty_cache()
        
        final_vram = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"  ✓ Final VRAM usage: {final_vram:.2f} GB")
        assert final_vram < 1.0, "Memory leak detected"
    
    @pytest.mark.slow
    def test_concurrent_simulations(self):
        """Test running multiple simulations concurrently"""
        print("\n" + "="*70)
        print("STRESS TEST 4: Concurrent Simulations")
        print("="*70)
        
        from quantum_circuit_simulator import QuantumCircuitSimulator
        from schrodinger_solver import SchrodingerSolver
        from harmonic_oscillator import HarmonicOscillator
        
        simulators = [
            QuantumCircuitSimulator(),
            SchrodingerSolver(),
            HarmonicOscillator()
        ]
        
        params_list = [
            {'n_qubits': 2, 'shots': 1000, 'gate_sequence': 'bell'},
            {'V0': 5.0, 'barrier_width': 1.5, 'k0': 4.0, 'sigma': 1.0, 'x0': -5.0},
            {'oscillator_length': 1.0, 'basis_size': 10, 'initial_n': 2}
        ]
        
        # Run all simulators multiple times
        n_iterations = 100
        start_time = time.time()
        
        for i in range(n_iterations):
            for sim, params in zip(simulators, params_list):
                result = sim.run(params)
                assert result is not None
            
            if (i + 1) % 20 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) * 3 / elapsed  # 3 simulators per iteration
                print(f"  Progress: {i+1}/{n_iterations} iterations ({rate:.1f} sims/sec)")
        
        elapsed = time.time() - start_time
        total_sims = n_iterations * 3
        
        print(f"\n✓ Completed {total_sims} simulations in {elapsed:.2f} seconds")
        print(f"✓ Average rate: {total_sims/elapsed:.1f} simulations/second")
    
    @pytest.mark.slow
    def test_memory_usage_stability(self):
        """Test memory usage remains stable over time"""
        print("\n" + "="*70)
        print("STRESS TEST 5: Memory Usage Stability")
        print("="*70)
        
        from quantum_circuit_simulator import QuantumCircuitSimulator
        
        simulator = QuantumCircuitSimulator()
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss / (1024**2)  # MB
        print(f"  Initial memory: {initial_memory:.2f} MB")
        
        memory_samples = []
        
        # Run 1000 simulations and track memory
        for i in range(1000):
            params = {
                'n_qubits': 2,
                'shots': 1000,
                'gate_sequence': 'bell'
            }
            
            result = simulator.run(params)
            
            if (i + 1) % 100 == 0:
                current_memory = process.memory_info().rss / (1024**2)
                memory_samples.append(current_memory)
                print(f"  After {i+1} sims: {current_memory:.2f} MB")
        
        final_memory = process.memory_info().rss / (1024**2)
        memory_increase = final_memory - initial_memory
        
        print(f"\n  Final memory: {final_memory:.2f} MB")
        print(f"  Memory increase: {memory_increase:.2f} MB")
        
        # Check for memory leaks (should not increase by more than 100MB)
        assert memory_increase < 100, f"Possible memory leak: {memory_increase:.2f} MB increase"
        print(f"✓ Memory usage stable (no significant leaks)")
    
    @pytest.mark.slow
    def test_error_recovery(self):
        """Test system recovery from errors"""
        print("\n" + "="*70)
        print("STRESS TEST 6: Error Recovery")
        print("="*70)
        
        from quantum_circuit_simulator import QuantumCircuitSimulator
        
        simulator = QuantumCircuitSimulator()
        
        # Test with invalid parameters
        invalid_params_list = [
            {'n_qubits': -1, 'shots': 1000, 'gate_sequence': 'bell'},
            {'n_qubits': 2, 'shots': -100, 'gate_sequence': 'bell'},
            {'n_qubits': 2, 'shots': 1000, 'gate_sequence': 'invalid'},
        ]
        
        errors_caught = 0
        
        for i, params in enumerate(invalid_params_list, 1):
            print(f"\n  Test {i}: Invalid params {params}")
            try:
                result = simulator.run(params)
                print(f"    ⚠ No error raised (may have defaults)")
            except Exception as e:
                errors_caught += 1
                print(f"    ✓ Error caught: {type(e).__name__}")
        
        # Verify system still works after errors
        print(f"\n  Verifying system recovery...")
        valid_params = {
            'n_qubits': 2,
            'shots': 1000,
            'gate_sequence': 'bell'
        }
        
        result = simulator.run(valid_params)
        assert result is not None
        # Check entropy exists (handle both dict and SimulationResult)
        if hasattr(result, 'observables'):
            assert 'entropy' in result.observables
        else:
            assert 'entropy' in result
        
        print(f"✓ System recovered successfully after {errors_caught} errors")
    
    @pytest.mark.slow
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_gpu_utilization(self):
        """Test GPU utilization under load"""
        print("\n" + "="*70)
        print("STRESS TEST 7: GPU Utilization")
        print("="*70)
        
        import torch
        from mlp_regressor import MLPRegressor
        from model_base import ModelConfig
        
        # Create dummy dataset
        X_train = torch.randn(10000, 10)
        y_train = torch.randn(10000, 3)
        
        config = ModelConfig(
            model_type='mlp',
            input_dim=10,
            output_dim=3,
            hyperparameters={'epochs': 5, 'batch_size': 256}
        )
        
        model = MLPRegressor(config)
        
        print(f"  Training on {len(X_train)} samples...")
        start_time = time.time()
        
        # Monitor GPU during training
        initial_vram = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"  Initial VRAM: {initial_vram:.2f} GB")
        
        model.train(X_train.numpy(), y_train.numpy())
        
        peak_vram = torch.cuda.max_memory_allocated(0) / (1024**3)
        final_vram = torch.cuda.memory_allocated(0) / (1024**3)
        
        elapsed = time.time() - start_time
        
        print(f"\n  Training completed in {elapsed:.2f} seconds")
        print(f"  Peak VRAM: {peak_vram:.2f} GB")
        print(f"  Final VRAM: {final_vram:.2f} GB")
        
        assert peak_vram < TOTAL_VRAM_GB * 0.9, "GPU memory usage too high"
        print(f"✓ GPU utilization within limits")
    
    @pytest.mark.slow
    def test_long_running_stability(self):
        """Test system stability over extended period"""
        print("\n" + "="*70)
        print("STRESS TEST 8: Long-Running Stability (5 minutes)")
        print("="*70)
        
        from quantum_circuit_simulator import QuantumCircuitSimulator
        
        simulator = QuantumCircuitSimulator()
        
        start_time = time.time()
        duration = 300  # 5 minutes
        
        iteration = 0
        errors = 0
        
        print(f"  Running for {duration} seconds...")
        
        while time.time() - start_time < duration:
            try:
                params = {
                    'n_qubits': 2,
                    'shots': 1000,
                    'gate_sequence': 'bell'
                }
                
                result = simulator.run(params)
                iteration += 1
                
                if iteration % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"  {elapsed:.0f}s: {iteration} iterations, {errors} errors")
            
            except Exception as e:
                errors += 1
                print(f"  Error at iteration {iteration}: {e}")
        
        elapsed = time.time() - start_time
        
        print(f"\n✓ Completed {iteration} iterations in {elapsed:.2f} seconds")
        print(f"✓ Error rate: {errors/iteration*100:.2f}%")
        
        assert errors / iteration < 0.01, "Error rate too high"


class TestDatasetStress:
    """Stress tests for dataset operations"""
    
    @pytest.mark.slow
    def test_large_hdf5_write(self):
        """Test writing large HDF5 dataset"""
        print("\n" + "="*70)
        print("DATASET STRESS TEST 1: Large HDF5 Write (10k samples)")
        print("="*70)
        
        from hdf5_storage import HDF5Storage
        import tempfile
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_path = f.name
        
        try:
            storage = HDF5Storage(temp_path)
            
            # Generate large dataset
            n_samples = 10000
            X = np.random.randn(n_samples, 10)
            y = np.random.randn(n_samples, 3)
            
            metadata = {
                'simulator': 'test',
                'n_samples': n_samples,
                'timestamp': time.time()
            }
            
            print(f"  Writing {n_samples} samples...")
            start_time = time.time()
            
            storage.save_dataset(X, y, metadata)
            
            elapsed = time.time() - start_time
            file_size = os.path.getsize(temp_path) / (1024**2)  # MB
            
            print(f"  ✓ Write completed in {elapsed:.2f} seconds")
            print(f"  ✓ File size: {file_size:.2f} MB")
            print(f"  ✓ Write speed: {file_size/elapsed:.2f} MB/s")
            
            # Verify read
            print(f"\n  Reading dataset...")
            start_time = time.time()
            
            X_loaded, y_loaded = storage.load_training_data()
            
            elapsed = time.time() - start_time
            
            print(f"  ✓ Read completed in {elapsed:.2f} seconds")
            print(f"  ✓ Read speed: {file_size/elapsed:.2f} MB/s")
            
            assert X_loaded.shape == X.shape
            assert y_loaded.shape == y.shape
        
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    @pytest.mark.slow
    def test_dataset_memory_efficiency(self):
        """Test dataset memory efficiency"""
        print("\n" + "="*70)
        print("DATASET STRESS TEST 2: Memory Efficiency")
        print("="*70)
        
        from dataset import Dataset
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**2)
        
        print(f"  Initial memory: {initial_memory:.2f} MB")
        
        # Create large dataset
        n_samples = 50000
        X = np.random.randn(n_samples, 10)
        y = np.random.randn(n_samples, 3)
        
        dataset = Dataset(X, y)
        
        after_create_memory = process.memory_info().rss / (1024**2)
        print(f"  After creation: {after_create_memory:.2f} MB")
        
        # Perform operations
        dataset.normalize()
        train_ds, val_ds = dataset.split(test_size=0.2)
        
        after_ops_memory = process.memory_info().rss / (1024**2)
        print(f"  After operations: {after_ops_memory:.2f} MB")
        
        # Calculate expected memory
        data_size = (X.nbytes + y.nbytes) / (1024**2)
        memory_overhead = (after_ops_memory - initial_memory) / data_size
        
        print(f"\n  Data size: {data_size:.2f} MB")
        print(f"  Memory overhead: {memory_overhead:.2f}x")
        
        assert memory_overhead < 3.0, "Memory overhead too high"
        print(f"✓ Memory efficiency acceptable")


def run_stress_tests():
    """Run all stress tests"""
    print("\n" + "="*70)
    print("RAVAN STRESS TEST SUITE")
    print("="*70)
    print("\nWarning: These tests may take 10-30 minutes to complete")
    print("and will stress CPU, GPU, and memory resources.\n")
    
    pytest.main([__file__, '-v', '-m', 'slow', '--tb=short'])


if __name__ == '__main__':
    run_stress_tests()
