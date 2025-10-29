"""
QuantumML-1K Benchmark Dataset Generator
Creates a standardized benchmark for quantum ML model evaluation
"""

import sys
sys.path.insert(0, '/home/windows/ravan-quantum-ml/src')

import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import json
import logging
from datetime import datetime

from pipeline.parameter_sweep import ParameterSweepGenerator
from pipeline.hdf5_storage import HDF5Storage
from pipeline.dataset import Dataset
from simulators.quantum_circuit_enhanced import EnhancedQuantumCircuitSimulator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BenchmarkDatasetDesigner:
    """
    Design and generate QuantumML-1K benchmark dataset
    
    Composition:
    - 500 uniform coverage samples (LHS)
    - 200 edge case samples
    - 200 challenging regime samples
    - 100 validation samples with analytical solutions
    Total: 1000 samples
    """
    
    def __init__(self, simulator):
        """
        Initialize benchmark designer
        
        Args:
            simulator: Quantum simulator instance
        """
        self.simulator = simulator
        self.parameter_space = simulator.get_parameter_space()
        self.sweep_generator = ParameterSweepGenerator()
        
        logger.info("BenchmarkDatasetDesigner initialized")
        logger.info(f"Parameter space: {list(self.parameter_space.keys())}")
    
    def design_uniform_coverage_samples(self, n_samples: int = 500) -> List[Dict]:
        """
        Design uniform coverage samples using LHS
        
        Args:
            n_samples: Number of samples (default: 500)
            
        Returns:
            List of parameter dictionaries
        """
        logger.info(f"Designing {n_samples} uniform coverage samples (LHS)...")
        
        samples = self.sweep_generator.latin_hypercube_sampling(
            self.parameter_space,
            n_samples=n_samples
        )
        
        logger.info(f"Generated {len(samples)} uniform coverage samples")
        return samples
    
    def design_edge_case_samples(self, n_samples: int = 200) -> List[Dict]:
        """
        Design edge case samples
        
        Edge cases include:
        - Maximum/minimum parameter values
        - Boundary conditions
        - Critical points (resonances, phase transitions)
        
        Args:
            n_samples: Number of samples (default: 200)
            
        Returns:
            List of parameter dictionaries
        """
        logger.info(f"Designing {n_samples} edge case samples...")
        
        edge_samples = []
        
        # Strategy 1: Corners of parameter space (2^n combinations)
        param_names = list(self.parameter_space.keys())
        n_params = len(param_names)
        
        # Generate corner points (min/max combinations)
        n_corners = min(2**n_params, n_samples // 2)
        for i in range(n_corners):
            sample = {}
            for j, param_name in enumerate(param_names):
                min_val, max_val = self.parameter_space[param_name]
                # Use binary representation to select min or max
                use_max = (i >> j) & 1
                sample[param_name] = max_val if use_max else min_val
            edge_samples.append(sample)
        
        # Strategy 2: One parameter at extreme, others at center
        remaining = n_samples - len(edge_samples)
        samples_per_param = remaining // (2 * n_params)
        
        for param_name in param_names:
            min_val, max_val = self.parameter_space[param_name]
            center_val = (min_val + max_val) / 2
            
            # Min extreme
            for _ in range(samples_per_param):
                sample = {name: (self.parameter_space[name][0] + self.parameter_space[name][1]) / 2 
                         for name in param_names}
                sample[param_name] = min_val
                edge_samples.append(sample)
            
            # Max extreme
            for _ in range(samples_per_param):
                sample = {name: (self.parameter_space[name][0] + self.parameter_space[name][1]) / 2 
                         for name in param_names}
                sample[param_name] = max_val
                edge_samples.append(sample)
        
        # Fill remaining with random edge samples
        while len(edge_samples) < n_samples:
            sample = {}
            for param_name in param_names:
                min_val, max_val = self.parameter_space[param_name]
                # Randomly choose min, max, or near-boundary
                choice = np.random.choice(['min', 'max', 'near_min', 'near_max'])
                if choice == 'min':
                    sample[param_name] = min_val
                elif choice == 'max':
                    sample[param_name] = max_val
                elif choice == 'near_min':
                    sample[param_name] = min_val + (max_val - min_val) * 0.1
                else:  # near_max
                    sample[param_name] = max_val - (max_val - min_val) * 0.1
            edge_samples.append(sample)
        
        logger.info(f"Generated {len(edge_samples)} edge case samples")
        return edge_samples[:n_samples]
    
    def design_challenging_regime_samples(self, n_samples: int = 200) -> List[Dict]:
        """
        Design challenging regime samples
        
        Challenging regimes for quantum circuits:
        - High qubit counts (maximum entanglement)
        - Deep circuits (complex dynamics)
        - Specific entangling patterns
        - Low shot counts (high noise)
        
        Args:
            n_samples: Number of samples (default: 200)
            
        Returns:
            List of parameter dictionaries
        """
        logger.info(f"Designing {n_samples} challenging regime samples...")
        
        challenging_samples = []
        samples_per_regime = n_samples // 4
        
        # Regime 1: High qubit count (maximum entanglement)
        for _ in range(samples_per_regime):
            sample = {}
            for param_name in self.parameter_space.keys():
                min_val, max_val = self.parameter_space[param_name]
                if 'qubit' in param_name.lower():
                    # Use high qubit count
                    sample[param_name] = max_val - (max_val - min_val) * np.random.uniform(0, 0.2)
                else:
                    # Random for other parameters
                    sample[param_name] = np.random.uniform(min_val, max_val)
            challenging_samples.append(sample)
        
        # Regime 2: Deep circuits
        for _ in range(samples_per_regime):
            sample = {}
            for param_name in self.parameter_space.keys():
                min_val, max_val = self.parameter_space[param_name]
                if 'depth' in param_name.lower():
                    # Use deep circuits
                    sample[param_name] = max_val - (max_val - min_val) * np.random.uniform(0, 0.2)
                else:
                    sample[param_name] = np.random.uniform(min_val, max_val)
            challenging_samples.append(sample)
        
        # Regime 3: Low shot count (high noise)
        for _ in range(samples_per_regime):
            sample = {}
            for param_name in self.parameter_space.keys():
                min_val, max_val = self.parameter_space[param_name]
                if 'shot' in param_name.lower():
                    # Use low shot count
                    sample[param_name] = min_val + (max_val - min_val) * np.random.uniform(0, 0.2)
                else:
                    sample[param_name] = np.random.uniform(min_val, max_val)
            challenging_samples.append(sample)
        
        # Regime 4: Combined challenges
        remaining = n_samples - len(challenging_samples)
        for _ in range(remaining):
            sample = {}
            for param_name in self.parameter_space.keys():
                min_val, max_val = self.parameter_space[param_name]
                # Bias towards extremes
                if np.random.random() < 0.5:
                    sample[param_name] = min_val + (max_val - min_val) * np.random.uniform(0, 0.3)
                else:
                    sample[param_name] = max_val - (max_val - min_val) * np.random.uniform(0, 0.3)
            challenging_samples.append(sample)
        
        logger.info(f"Generated {len(challenging_samples)} challenging regime samples")
        return challenging_samples
    
    def design_validation_samples(self, n_samples: int = 100) -> List[Dict]:
        """
        Design validation samples with known analytical solutions
        
        For quantum circuits:
        - Simple 2-qubit Bell states
        - Single qubit rotations
        - Known entanglement patterns
        
        Args:
            n_samples: Number of samples (default: 100)
            
        Returns:
            List of parameter dictionaries
        """
        logger.info(f"Designing {n_samples} validation samples...")
        
        validation_samples = []
        samples_per_type = n_samples // 3
        
        # Type 1: Simple 2-qubit systems
        for _ in range(samples_per_type):
            sample = {}
            for param_name in self.parameter_space.keys():
                min_val, max_val = self.parameter_space[param_name]
                if 'qubit' in param_name.lower():
                    sample[param_name] = 2  # Minimum qubits
                elif 'depth' in param_name.lower():
                    sample[param_name] = 1  # Shallow circuit
                else:
                    sample[param_name] = np.random.uniform(min_val, max_val)
            validation_samples.append(sample)
        
        # Type 2: Known rotation angles (multiples of π/2)
        for _ in range(samples_per_type):
            sample = {}
            for param_name in self.parameter_space.keys():
                min_val, max_val = self.parameter_space[param_name]
                if 'angle' in param_name.lower() or 'theta' in param_name.lower():
                    # Use multiples of π/2
                    sample[param_name] = np.random.choice([0, np.pi/2, np.pi, 3*np.pi/2])
                else:
                    sample[param_name] = np.random.uniform(min_val, max_val)
            validation_samples.append(sample)
        
        # Type 3: High shot count (low noise, easier to validate)
        remaining = n_samples - len(validation_samples)
        for _ in range(remaining):
            sample = {}
            for param_name in self.parameter_space.keys():
                min_val, max_val = self.parameter_space[param_name]
                if 'shot' in param_name.lower():
                    sample[param_name] = max_val  # Maximum shots
                else:
                    sample[param_name] = np.random.uniform(min_val, max_val)
            validation_samples.append(sample)
        
        logger.info(f"Generated {len(validation_samples)} validation samples")
        return validation_samples
    
    def create_benchmark_design(self) -> Dict:
        """
        Create complete benchmark dataset design
        
        Returns:
            Dictionary with all sample categories
        """
        logger.info("=" * 80)
        logger.info("Creating QuantumML-1K Benchmark Design")
        logger.info("=" * 80)
        
        design = {
            'uniform_coverage': self.design_uniform_coverage_samples(500),
            'edge_cases': self.design_edge_case_samples(200),
            'challenging_regimes': self.design_challenging_regime_samples(200),
            'validation': self.design_validation_samples(100)
        }
        
        total_samples = sum(len(samples) for samples in design.values())
        
        logger.info(f"\nBenchmark Design Summary:")
        logger.info(f"  Uniform Coverage: {len(design['uniform_coverage'])} samples")
        logger.info(f"  Edge Cases: {len(design['edge_cases'])} samples")
        logger.info(f"  Challenging Regimes: {len(design['challenging_regimes'])} samples")
        logger.info(f"  Validation: {len(design['validation'])} samples")
        logger.info(f"  Total: {total_samples} samples")
        
        return design
    
    def save_design(self, design: Dict, output_path: str):
        """
        Save benchmark design to JSON
        
        Args:
            design: Benchmark design dictionary
            output_path: Path to save JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        serializable_design = {}
        for category, samples in design.items():
            serializable_design[category] = [
                {k: float(v) for k, v in sample.items()}
                for sample in samples
            ]
        
        with open(output_path, 'w') as f:
            json.dump(serializable_design, f, indent=2)
        
        logger.info(f"Benchmark design saved to: {output_path}")


def create_quantumml_1k_design():
    """
    Create QuantumML-1K benchmark design
    """
    # Initialize simulator
    simulator = EnhancedQuantumCircuitSimulator()
    
    # Create designer
    designer = BenchmarkDatasetDesigner(simulator)
    
    # Create design
    design = designer.create_benchmark_design()
    
    # Save design
    designer.save_design(design, './data/benchmark/quantumml_1k_design.json')
    
    return design


if __name__ == '__main__':
    design = create_quantumml_1k_design()
    
    print("\n" + "=" * 80)
    print("QuantumML-1K Benchmark Design Complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review design in: ./data/benchmark/quantumml_1k_design.json")
    print("2. Run simulations to generate dataset")
    print("3. Package and document benchmark")
