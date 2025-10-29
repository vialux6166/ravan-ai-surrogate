"""
Adaptive Sampling for Ravan Quantum-ML System
Intelligently selects high-uncertainty regions for simulation
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from typing import Tuple, Dict, List
import logging
from pathlib import Path
import time

from dataset import Dataset
from training_pipeline import TrainingPipeline
from uncertainty_quantifier import UncertaintyQuantifier
from quantum_circuit_simulator import QuantumCircuitSimulator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdaptiveSampler:
    """
    Adaptive sampling using uncertainty-based selection
    
    Iteratively:
    1. Train model on current dataset
    2. Generate candidate points
    3. Select high-uncertainty points
    4. Simulate selected points
    5. Augment dataset and repeat
    """
    
    def __init__(
        self,
        simulator,
        initial_samples: int = 2000,
        samples_per_iteration: int = 500,
        n_iterations: int = 5,
        candidate_pool_size: int = 10000,
        mc_samples: int = 50
    ):
        """
        Initialize adaptive sampler
        
        Args:
            simulator: Simulation module
            initial_samples: Initial dataset size (LHS)
            samples_per_iteration: Samples to add per iteration
            n_iterations: Number of adaptive iterations
            candidate_pool_size: Size of candidate pool for selection
            mc_samples: MC dropout samples for uncertainty
        """
        self.simulator = simulator
        self.initial_samples = initial_samples
        self.samples_per_iteration = samples_per_iteration
        self.n_iterations = n_iterations
        self.candidate_pool_size = candidate_pool_size
        self.mc_samples = mc_samples
        
        self.pipeline = TrainingPipeline()
        
        logger.info(
            f"AdaptiveSampler initialized: "
            f"initial={initial_samples}, per_iter={samples_per_iteration}, "
            f"iterations={n_iterations}"
        )
    
    def run(
        self,
        parameter_space: Dict[str, Tuple[float, float]],
        output_dir: str = './data/adaptive'
    ) -> Tuple[Dataset, Dict]:
        """
        Run adaptive sampling loop
        
        Args:
            parameter_space: Parameter ranges for sampling
            output_dir: Output directory for results
            
        Returns:
            Tuple of (final_dataset, metrics_history)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 80)
        logger.info("Starting Adaptive Sampling")
        logger.info("=" * 80)
        
        # Step 1: Generate initial dataset with LHS
        logger.info(f"\nStep 1: Generating initial dataset ({self.initial_samples} samples)...")
        X_train, y_train = self._generate_initial_dataset(parameter_space)
        
        # Create dataset
        dataset = self._create_dataset(X_train, y_train, parameter_space)
        
        # Metrics tracking
        metrics_history = {
            'iteration': [],
            'n_samples': [],
            'r2_score': [],
            'mse': [],
            'mae': [],
            'mean_uncertainty': [],
            'max_uncertainty': [],
            'training_time': []
        }
        
        # Adaptive loop
        for iteration in range(self.n_iterations):
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Iteration {iteration + 1}/{self.n_iterations}")
            logger.info(f"{'=' * 80}")
            
            # Step 2: Train model
            logger.info(f"\nTraining model on {dataset.n_samples} samples...")
            start_time = time.time()
            
            train_dataset, val_dataset = self.pipeline.prepare_data(
                dataset,
                train_ratio=0.8,
                normalize=True
            )
            
            model, train_metrics = self.pipeline.train_model(
                'mlp',
                train_dataset,
                val_dataset
            )
            
            training_time = time.time() - start_time
            
            logger.info(
                f"Model trained: R²={train_metrics.r2_score:.4f}, "
                f"MSE={train_metrics.val_mse:.6f}, time={training_time:.2f}s"
            )
            
            # Step 3: Generate candidate points
            logger.info(f"\nGenerating {self.candidate_pool_size} candidate points...")
            X_candidates = self._generate_candidates(parameter_space)
            
            # Step 4: Compute uncertainty
            logger.info("Computing uncertainty for candidates...")
            uq = UncertaintyQuantifier(
                model=model,
                method='mc_dropout',
                n_samples=self.mc_samples
            )
            
            total_uncertainty = uq.get_total_uncertainty(X_candidates)
            
            logger.info(
                f"Uncertainty stats: mean={total_uncertainty.mean():.6f}, "
                f"max={total_uncertainty.max():.6f}, min={total_uncertainty.min():.6f}"
            )
            
            # Step 5: Select high-uncertainty points
            logger.info(f"Selecting top {self.samples_per_iteration} high-uncertainty points...")
            high_unc_indices = np.argsort(total_uncertainty)[-self.samples_per_iteration:]
            X_new = X_candidates[high_unc_indices]
            
            logger.info(
                f"Selected samples uncertainty: "
                f"mean={total_uncertainty[high_unc_indices].mean():.6f}"
            )
            
            # Step 6: Simulate new points
            logger.info(f"Simulating {len(X_new)} new points...")
            y_new = self._simulate_points(X_new, parameter_space)
            
            # Step 7: Augment dataset
            X_train = np.vstack([X_train, X_new])
            y_train = np.vstack([y_train, y_new])
            
            dataset = self._create_dataset(X_train, y_train, parameter_space)
            
            logger.info(f"Dataset augmented: now {dataset.n_samples} samples")
            
            # Record metrics
            metrics_history['iteration'].append(iteration + 1)
            metrics_history['n_samples'].append(dataset.n_samples)
            metrics_history['r2_score'].append(train_metrics.r2_score)
            metrics_history['mse'].append(train_metrics.val_mse)
            metrics_history['mae'].append(train_metrics.val_mae)
            metrics_history['mean_uncertainty'].append(total_uncertainty.mean())
            metrics_history['max_uncertainty'].append(total_uncertainty.max())
            metrics_history['training_time'].append(training_time)
        
        # Final training
        logger.info(f"\n{'=' * 80}")
        logger.info("Final Training on Complete Dataset")
        logger.info(f"{'=' * 80}")
        
        train_dataset, val_dataset = self.pipeline.prepare_data(
            dataset,
            train_ratio=0.8,
            normalize=True
        )
        
        final_model, final_metrics = self.pipeline.train_model(
            'mlp',
            train_dataset,
            val_dataset
        )
        
        logger.info(
            f"\nFinal model: R²={final_metrics.r2_score:.4f}, "
            f"MSE={final_metrics.val_mse:.6f}"
        )
        
        # Save results
        self._save_results(dataset, metrics_history, output_dir)
        
        logger.info(f"\n{'=' * 80}")
        logger.info("Adaptive Sampling Complete!")
        logger.info(f"{'=' * 80}")
        logger.info(f"Final dataset size: {dataset.n_samples}")
        logger.info(f"Final R²: {final_metrics.r2_score:.4f}")
        logger.info(f"Results saved to: {output_dir}")
        
        return dataset, metrics_history
    
    def _generate_initial_dataset(
        self,
        parameter_space: Dict[str, Tuple[float, float]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate initial dataset using LHS"""
        generator = ParameterSweepGenerator()
        samples = generator.latin_hypercube_sampling(parameter_space, self.initial_samples)
        X = self._samples_to_array(samples, parameter_space)
        y = self._simulate_points(X, parameter_space)
        return X, y
    
    def _generate_candidates(
        self,
        parameter_space: Dict[str, Tuple[float, float]]
    ) -> np.ndarray:
        """Generate candidate points using LHS"""
        generator = ParameterSweepGenerator()
        samples = generator.latin_hypercube_sampling(parameter_space, self.candidate_pool_size)
        X_candidates = self._samples_to_array(samples, parameter_space)
        return X_candidates
    
    def _samples_to_array(
        self,
        samples: List[Dict[str, float]],
        parameter_space: Dict[str, Tuple[float, float]]
    ) -> np.ndarray:
        """Convert list of parameter dicts to numpy array"""
        param_names = list(parameter_space.keys())
        X = np.array([[sample[name] for name in param_names] for sample in samples])
        return X
    
    def _simulate_points(
        self,
        X: np.ndarray,
        parameter_space: Dict[str, Tuple[float, float]]
    ) -> np.ndarray:
        """Simulate points using the simulator"""
        param_names = list(parameter_space.keys())
        results = []
        
        for i, x in enumerate(X):
            # Convert array to parameter dict
            params = {name: float(x[j]) for j, name in enumerate(param_names)}
            
            # Run simulation
            result = self.simulator.run_with_validation(params)
            
            # Extract observables from SimulationResult
            if hasattr(result, 'observables'):
                results.append(list(result.observables.values()))
            else:
                # Fallback if result is already a dict
                results.append(list(result.values()))
            
            if (i + 1) % 100 == 0:
                logger.info(f"  Simulated {i + 1}/{len(X)} points")
        
        return np.array(results)
    
    def _create_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        parameter_space: Dict[str, Tuple[float, float]]
    ) -> Dataset:
        """Create dataset from arrays"""
        param_names = list(parameter_space.keys())
        
        # Get observable names from a test run
        test_params = {name: parameter_space[name][0] for name in param_names}
        test_result = self.simulator.run_with_validation(test_params)
        
        if hasattr(test_result, 'observables'):
            obs_names = list(test_result.observables.keys())
        else:
            obs_names = list(test_result.keys())
        
        # Remove 'purity' if present (constant for pure states)
        if 'purity' in obs_names:
            purity_idx = obs_names.index('purity')
            obs_names.remove('purity')
            y = np.delete(y, purity_idx, axis=1)
            logger.info(f"Removed constant 'purity' column from observables")
        
        return Dataset(
            X=X,
            y=y,
            parameter_names=param_names,
            observable_names=obs_names,
            metadata={'sampling_method': 'adaptive'}
        )
    
    def _save_results(
        self,
        dataset: Dataset,
        metrics_history: Dict,
        output_dir: Path
    ):
        """Save results to disk"""
        # Save dataset
        from hdf5_storage import HDF5Storage
        storage = HDF5Storage()
        dataset_path = output_dir / 'adaptive_dataset.h5'
        storage.save_dataset(dataset, str(dataset_path))
        logger.info(f"Dataset saved: {dataset_path}")
        
        # Save metrics (convert numpy types to Python types for JSON)
        import json
        metrics_serializable = {}
        for key, values in metrics_history.items():
            if isinstance(values, list):
                metrics_serializable[key] = [float(v) if hasattr(v, 'item') else v for v in values]
            else:
                metrics_serializable[key] = values
        
        metrics_path = output_dir / 'metrics_history.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        logger.info(f"Metrics saved: {metrics_path}")
        
        # Create plots
        self._plot_metrics(metrics_history, output_dir)
    
    def _plot_metrics(self, metrics_history: Dict, output_dir: Path):
        """Plot adaptive sampling metrics"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # R² score over iterations
        ax = axes[0, 0]
        ax.plot(metrics_history['iteration'], metrics_history['r2_score'], 'o-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('R² Score')
        ax.set_title('Model Performance Over Iterations')
        ax.grid(True, alpha=0.3)
        
        # MSE over iterations
        ax = axes[0, 1]
        ax.plot(metrics_history['iteration'], metrics_history['mse'], 'o-', linewidth=2, color='red')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('MSE')
        ax.set_title('Validation MSE Over Iterations')
        ax.grid(True, alpha=0.3)
        
        # Dataset size over iterations
        ax = axes[1, 0]
        ax.plot(metrics_history['iteration'], metrics_history['n_samples'], 'o-', linewidth=2, color='green')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Dataset Growth')
        ax.grid(True, alpha=0.3)
        
        # Uncertainty over iterations
        ax = axes[1, 1]
        ax.plot(metrics_history['iteration'], metrics_history['mean_uncertainty'], 'o-', linewidth=2, label='Mean', color='purple')
        ax.plot(metrics_history['iteration'], metrics_history['max_uncertainty'], 'o-', linewidth=2, label='Max', color='orange')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Uncertainty')
        ax.set_title('Uncertainty Over Iterations')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_dir / 'adaptive_sampling_metrics.png'
        plt.savefig(plot_path, dpi=150)
        logger.info(f"Metrics plot saved: {plot_path}")
        plt.close()


def run_adaptive_sampling_demo():
    """Demo of adaptive sampling on quantum circuit simulator"""
    
    # Initialize simulator
    simulator = QuantumCircuitSimulator()
    
    # Define parameter space
    parameter_space = simulator.get_parameter_space()
    
    # Create adaptive sampler
    sampler = AdaptiveSampler(
        simulator=simulator,
        initial_samples=500,  # Start small for demo
        samples_per_iteration=200,
        n_iterations=3,
        candidate_pool_size=2000,
        mc_samples=30
    )
    
    # Run adaptive sampling
    dataset, metrics = sampler.run(
        parameter_space=parameter_space,
        output_dir='./data/adaptive_demo'
    )
    
    return dataset, metrics


if __name__ == '__main__':
    dataset, metrics = run_adaptive_sampling_demo()
