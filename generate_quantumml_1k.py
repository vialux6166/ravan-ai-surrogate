"""
Generate QuantumML-1K Benchmark Dataset
Runs simulations for all designed samples and creates the benchmark
"""

import sys
sys.path.insert(0, '/home/windows/ravan-quantum-ml/src')

import numpy as np
import json
from pathlib import Path
from typing import Dict, List
import logging
from datetime import datetime
import time

from pipeline.hdf5_storage import HDF5Storage
from pipeline.dataset import Dataset
from simulators.quantum_circuit_enhanced import EnhancedQuantumCircuitSimulator
from uncertainty_quantifier import UncertaintyQuantifier
from models.mlp import MLPRegressor
from models.base import ModelConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantumML1KGenerator:
    """
    Generate QuantumML-1K benchmark dataset
    """
    
    def __init__(self, design_path: str):
        """
        Initialize generator
        
        Args:
            design_path: Path to benchmark design JSON
        """
        self.design_path = Path(design_path)
        self.simulator = EnhancedQuantumCircuitSimulator()
        self.design = self._load_design()
        
        logger.info("QuantumML1KGenerator initialized")
        logger.info(f"Loaded design with {self._count_total_samples()} samples")
    
    def _load_design(self) -> Dict:
        """Load benchmark design from JSON"""
        with open(self.design_path, 'r') as f:
            design = json.load(f)
        return design
    
    def _count_total_samples(self) -> int:
        """Count total samples in design"""
        return sum(len(samples) for samples in self.design.values())
    
    def simulate_samples(
        self,
        samples: List[Dict],
        category: str
    ) -> tuple:
        """
        Simulate a list of samples
        
        Args:
            samples: List of parameter dictionaries
            category: Category name for logging
            
        Returns:
            Tuple of (X, y, validation_flags)
        """
        logger.info(f"Simulating {len(samples)} {category} samples...")
        
        X_list = []
        y_list = []
        validation_flags = []
        
        param_names = list(self.simulator.get_parameter_space().keys())
        
        for i, sample in enumerate(samples):
            try:
                # Run simulation
                result = self.simulator.run(sample)
                
                # Extract parameters and observables
                X = np.array([sample[name] for name in param_names])
                
                if hasattr(result, 'observables'):
                    y = np.array(list(result.observables.values()))
                    validation_passed = result.validation_passed
                else:
                    y = np.array(list(result.values()))
                    validation_passed = True
                
                X_list.append(X)
                y_list.append(y)
                validation_flags.append(validation_passed)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"  Completed {i + 1}/{len(samples)} samples")
                
            except Exception as e:
                logger.warning(f"  Failed to simulate sample {i}: {e}")
                continue
        
        X = np.array(X_list)
        y = np.array(y_list)
        validation_flags = np.array(validation_flags)
        
        # Check validation
        n_passed = validation_flags.sum()
        logger.info(f"  Validation: {n_passed}/{len(validation_flags)} samples passed")
        
        return X, y, validation_flags
    
    def compute_difficulty_scores(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_path: str = None
    ) -> np.ndarray:
        """
        Compute difficulty scores based on model uncertainty
        
        Args:
            X: Input features
            y: Target observables
            model_path: Path to trained model (if None, trains a quick model)
            
        Returns:
            Difficulty scores (higher = more difficult)
        """
        logger.info("Computing difficulty scores...")
        
        if model_path is None:
            # Train a quick model for difficulty estimation
            logger.info("  Training quick model for difficulty estimation...")
            
            # Split data
            n_train = int(0.8 * len(X))
            X_train, X_val = X[:n_train], X[n_train:]
            y_train, y_val = y[:n_train], y[n_train:]
            
            # Remove purity if present
            if y.shape[1] > 7:
                purity_idx = 1
                y_train = np.delete(y_train, purity_idx, axis=1)
                y_val = np.delete(y_val, purity_idx, axis=1)
                y = np.delete(y, purity_idx, axis=1)
            
            # Create and train model
            config = ModelConfig(
                model_type='mlp',
                input_dim=X.shape[1],
                output_dim=y.shape[1],
                hyperparameters={
                    'hidden_layers': [128, 64],
                    'dropout': 0.2,
                    'learning_rate': 0.001,
                    'batch_size': 64,
                    'epochs': 30,
                    'patience': 5
                },
                save_path='./temp_difficulty_model'
            )
            
            model = MLPRegressor(config)
            model.train(X_train, y_train, X_val, y_val)
        else:
            # Load existing model
            logger.info(f"  Loading model from {model_path}...")
            # TODO: Implement model loading
            raise NotImplementedError("Model loading not yet implemented")
        
        # Compute uncertainty as difficulty score
        logger.info("  Computing uncertainty scores...")
        uq = UncertaintyQuantifier(model, method='mc_dropout', n_samples=20)
        _, uncertainties = uq.predict_with_uncertainty(X)
        
        # Average uncertainty across all observables
        difficulty_scores = uncertainties.mean(axis=1)
        
        logger.info(f"  Difficulty scores: min={difficulty_scores.min():.4f}, "
                   f"max={difficulty_scores.max():.4f}, mean={difficulty_scores.mean():.4f}")
        
        return difficulty_scores
    
    def generate_dataset(
        self,
        output_path: str = './data/benchmark/quantumml_1k.h5',
        compute_difficulty: bool = True
    ) -> Dataset:
        """
        Generate complete QuantumML-1K dataset
        
        Args:
            output_path: Path to save HDF5 file
            compute_difficulty: Whether to compute difficulty scores
            
        Returns:
            Complete dataset
        """
        logger.info("=" * 80)
        logger.info("Generating QuantumML-1K Benchmark Dataset")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Simulate all categories
        all_X = []
        all_y = []
        all_validation = []
        all_categories = []
        
        for category, samples in self.design.items():
            logger.info(f"\nCategory: {category}")
            X, y, validation = self.simulate_samples(samples, category)
            
            all_X.append(X)
            all_y.append(y)
            all_validation.append(validation)
            all_categories.extend([category] * len(X))
        
        # Concatenate all data
        X_full = np.vstack(all_X)
        y_full = np.vstack(all_y)
        validation_full = np.concatenate(all_validation)
        
        logger.info(f"\nTotal samples: {len(X_full)}")
        logger.info(f"Features: {X_full.shape[1]}")
        logger.info(f"Observables: {y_full.shape[1]}")
        logger.info(f"Validation passed: {validation_full.sum()}/{len(validation_full)}")
        
        # Remove purity column if present
        param_names = list(self.simulator.get_parameter_space().keys())
        
        # Get observable names
        test_params = {name: self.simulator.get_parameter_space()[name][0] 
                      for name in param_names}
        test_result = self.simulator.run(test_params)
        
        if hasattr(test_result, 'observables'):
            obs_names = list(test_result.observables.keys())
        else:
            obs_names = list(test_result.keys())
        
        # Remove purity if present
        if 'purity' in obs_names:
            purity_idx = obs_names.index('purity')
            obs_names.remove('purity')
            y_full = np.delete(y_full, purity_idx, axis=1)
            logger.info(f"Removed constant 'purity' column")
        
        # Compute difficulty scores
        if compute_difficulty:
            difficulty_scores = self.compute_difficulty_scores(X_full, y_full)
        else:
            difficulty_scores = np.zeros(len(X_full))
        
        # Create dataset
        metadata = {
            'name': 'QuantumML-1K',
            'version': '1.0',
            'description': 'Benchmark dataset for quantum machine learning',
            'creation_date': datetime.now().isoformat(),
            'simulator': 'EnhancedQuantumCircuitSimulator',
            'n_samples': len(X_full),
            'categories': {
                'uniform_coverage': 500,
                'edge_cases': 200,
                'challenging_regimes': 200,
                'validation': 100
            },
            'validation_passed': int(validation_full.sum()),
            'generation_time_seconds': time.time() - start_time
        }
        
        dataset = Dataset(
            X=X_full,
            y=y_full,
            parameter_names=param_names,
            observable_names=obs_names,
            metadata=metadata
        )
        
        # Save dataset
        logger.info(f"\nSaving dataset to {output_path}...")
        storage = HDF5Storage()
        storage.save_dataset(dataset, output_path)
        
        # Save additional metadata
        metadata_path = Path(output_path).parent / 'quantumml_1k_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump({
                **metadata,
                'categories_list': all_categories,
                'difficulty_scores': difficulty_scores.tolist(),
                'validation_flags': validation_full.tolist()
            }, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_path}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"\n" + "=" * 80)
        logger.info(f"QuantumML-1K Generation Complete!")
        logger.info(f"=" * 80)
        logger.info(f"Total time: {elapsed_time:.2f}s ({elapsed_time/60:.2f} minutes)")
        logger.info(f"Dataset saved to: {output_path}")
        
        return dataset


def generate_quantumml_1k_quick_test():
    """
    Quick test with reduced samples (100 total)
    """
    logger.info("Running quick test with 100 samples...")
    
    # Load design
    design_path = './data/benchmark/quantumml_1k_design.json'
    
    if not Path(design_path).exists():
        logger.error(f"Design file not found: {design_path}")
        logger.info("Run benchmark_dataset_generator.py first to create the design")
        return
    
    # Modify design for quick test
    with open(design_path, 'r') as f:
        design = json.load(f)
    
    # Reduce sample counts
    quick_design = {
        'uniform_coverage': design['uniform_coverage'][:50],
        'edge_cases': design['edge_cases'][:20],
        'challenging_regimes': design['challenging_regimes'][:20],
        'validation': design['validation'][:10]
    }
    
    # Save quick design
    quick_design_path = './data/benchmark/quantumml_1k_design_quick.json'
    with open(quick_design_path, 'w') as f:
        json.dump(quick_design, f, indent=2)
    
    # Generate dataset
    generator = QuantumML1KGenerator(quick_design_path)
    dataset = generator.generate_dataset(
        output_path='./data/benchmark/quantumml_100_test.h5',
        compute_difficulty=True
    )
    
    return dataset


if __name__ == '__main__':
    # Run quick test
    dataset = generate_quantumml_1k_quick_test()
    
    print("\n" + "=" * 80)
    print("Quick Test Complete!")
    print("=" * 80)
    print("\nTo generate full QuantumML-1K dataset (1000 samples):")
    print("  generator = QuantumML1KGenerator('./data/benchmark/quantumml_1k_design.json')")
    print("  dataset = generator.generate_dataset()")
