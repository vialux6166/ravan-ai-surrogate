#!/usr/bin/env python3
"""
Validate Adaptive Sampling Efficiency
Task 11.4: Compare adaptive sampling vs uniform sampling

This script:
1. Trains a model with uniform sampling (baseline)
2. Trains a model with adaptive sampling
3. Compares final model accuracy
4. Measures reduction in total simulations
5. Generates comparison plots
"""

import sys
sys.path.insert(0, '/home/windows/ravan-quantum-ml/src')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json
from typing import Dict, Tuple

from adaptive_sampling import AdaptiveSampler
from simulators.quantum_circuit_enhanced import EnhancedQuantumCircuitSimulator
from pipeline.training import TrainingPipeline
from pipeline.parameter_sweep import ParameterSweepGenerator
from pipeline.dataset import Dataset
from pipeline.hdf5_storage import HDF5Storage
from utils.logging import setup_logging

# Setup logging
logger = setup_logging(level='INFO')


class AdaptiveSamplingValidator:
    """
    Validates adaptive sampling efficiency against uniform sampling
    """
    
    def __init__(
        self,
        simulator,
        target_accuracy: float = 0.90,  # Target R² score
        max_samples: int = 5000,
        output_dir: str = './results/adaptive_validation'
    ):
        """
        Initialize validator
        
        Args:
            simulator: Simulation module
            target_accuracy: Target R² score to achieve
            max_samples: Maximum samples for comparison
            output_dir: Output directory for results
        """
        self.simulator = simulator
        self.target_accuracy = target_accuracy
        self.max_samples = max_samples
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.pipeline = TrainingPipeline()
        self.parameter_space = simulator.get_parameter_space()
        
        logger.info("AdaptiveSamplingValidator initialized")
        logger.info(f"Target accuracy: R² >= {target_accuracy:.2f}")
        logger.info(f"Max samples: {max_samples}")
    
    def run_uniform_sampling(
        self,
        sample_sizes: list = [500, 1000, 2000, 3000, 4000, 5000]
    ) -> Dict:
        """
        Run uniform sampling baseline with different dataset sizes
        
        Args:
            sample_sizes: List of dataset sizes to test
            
        Returns:
            Dictionary with results for each sample size
        """
        logger.info("\n" + "=" * 80)
        logger.info("UNIFORM SAMPLING BASELINE")
        logger.info("=" * 80)
        
        results = {
            'sample_sizes': [],
            'r2_scores': [],
            'mse_values': [],
            'mae_values': [],
            'training_times': [],
            'simulation_times': []
        }
        
        generator = ParameterSweepGenerator()
        
        for n_samples in sample_sizes:
            if n_samples > self.max_samples:
                continue
            
            logger.info(f"\n--- Uniform Sampling: {n_samples} samples ---")
            
            # Generate uniform samples using LHS
            logger.info("Generating samples with LHS...")
            start_sim = time.time()
            samples = generator.latin_hypercube_sampling(self.parameter_space, n_samples)
            
            # Convert to arrays
            X, y = self._simulate_samples(samples)
            simulation_time = time.time() - start_sim
            
            # Create dataset
            dataset = self._create_dataset(X, y)
            
            # Train model
            logger.info("Training model...")
            start_train = time.time()
            train_dataset, val_dataset = self.pipeline.prepare_data(
                dataset, train_ratio=0.8, normalize=True
            )
            
            model, metrics = self.pipeline.train_model(
                'mlp', train_dataset, val_dataset
            )
            training_time = time.time() - start_train
            
            # Record results
            results['sample_sizes'].append(n_samples)
            results['r2_scores'].append(metrics.r2_score)
            results['mse_values'].append(metrics.val_mse)
            results['mae_values'].append(metrics.val_mae)
            results['training_times'].append(training_time)
            results['simulation_times'].append(simulation_time)
            
            logger.info(
                f"Results: R²={metrics.r2_score:.4f}, "
                f"MSE={metrics.val_mse:.6f}, "
                f"sim_time={simulation_time:.2f}s, "
                f"train_time={training_time:.2f}s"
            )
            
            # Check if target accuracy reached
            if metrics.r2_score >= self.target_accuracy:
                logger.info(
                    f"✓ Target accuracy {self.target_accuracy:.2f} reached "
                    f"with {n_samples} samples"
                )
                break
        
        # Save results
        self._save_results(results, 'uniform_sampling_results.json')
        
        return results
    
    def run_adaptive_sampling(
        self,
        initial_samples: int = 500,
        samples_per_iteration: int = 300,
        max_iterations: int = 10
    ) -> Dict:
        """
        Run adaptive sampling
        
        Args:
            initial_samples: Initial dataset size
            samples_per_iteration: Samples to add per iteration
            max_iterations: Maximum iterations
            
        Returns:
            Dictionary with adaptive sampling results
        """
        logger.info("\n" + "=" * 80)
        logger.info("ADAPTIVE SAMPLING")
        logger.info("=" * 80)
        
        results = {
            'iterations': [],
            'sample_sizes': [],
            'r2_scores': [],
            'mse_values': [],
            'mae_values': [],
            'training_times': [],
            'simulation_times': [],
            'mean_uncertainties': []
        }
        
        # Generate initial dataset
        logger.info(f"\nGenerating initial dataset ({initial_samples} samples)...")
        start_sim = time.time()
        generator = ParameterSweepGenerator()
        samples = generator.latin_hypercube_sampling(self.parameter_space, initial_samples)
        X_train, y_train = self._simulate_samples(samples)
        total_simulation_time = time.time() - start_sim
        
        dataset = self._create_dataset(X_train, y_train)
        
        # Adaptive loop
        for iteration in range(max_iterations):
            logger.info(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
            logger.info(f"Current dataset size: {dataset.n_samples}")
            
            # Train model
            logger.info("Training model...")
            start_train = time.time()
            train_dataset, val_dataset = self.pipeline.prepare_data(
                dataset, train_ratio=0.8, normalize=True
            )
            
            model, metrics = self.pipeline.train_model(
                'mlp', train_dataset, val_dataset
            )
            training_time = time.time() - start_train
            
            # Record results
            results['iterations'].append(iteration + 1)
            results['sample_sizes'].append(dataset.n_samples)
            results['r2_scores'].append(metrics.r2_score)
            results['mse_values'].append(metrics.val_mse)
            results['mae_values'].append(metrics.val_mae)
            results['training_times'].append(training_time)
            results['simulation_times'].append(total_simulation_time)
            
            logger.info(
                f"Results: R²={metrics.r2_score:.4f}, "
                f"MSE={metrics.val_mse:.6f}"
            )
            
            # Check if target accuracy reached
            if metrics.r2_score >= self.target_accuracy:
                logger.info(
                    f"✓ Target accuracy {self.target_accuracy:.2f} reached "
                    f"with {dataset.n_samples} samples at iteration {iteration + 1}"
                )
                break
            
            # Check if max samples reached
            if dataset.n_samples >= self.max_samples:
                logger.info(f"Max samples {self.max_samples} reached")
                break
            
            # Generate candidates and select high-uncertainty points
            logger.info("Selecting high-uncertainty points...")
            from uncertainty_quantifier import UncertaintyQuantifier
            
            uq = UncertaintyQuantifier(model=model, method='mc_dropout', n_samples=30)
            
            # Generate candidates
            n_candidates = min(5000, self.max_samples)
            candidate_samples = generator.latin_hypercube_sampling(
                self.parameter_space, n_candidates
            )
            X_candidates = self._samples_to_array(candidate_samples)
            
            # Compute uncertainty
            total_uncertainty = uq.get_total_uncertainty(X_candidates)
            mean_uncertainty = total_uncertainty.mean()
            results['mean_uncertainties'].append(mean_uncertainty)
            
            logger.info(f"Mean uncertainty: {mean_uncertainty:.6f}")
            
            # Select high-uncertainty points
            n_new = min(samples_per_iteration, self.max_samples - dataset.n_samples)
            high_unc_indices = np.argsort(total_uncertainty)[-n_new:]
            X_new = X_candidates[high_unc_indices]
            
            # Simulate new points
            logger.info(f"Simulating {n_new} new points...")
            start_sim = time.time()
            y_new = self._simulate_array(X_new)
            total_simulation_time += time.time() - start_sim
            
            # Augment dataset
            X_train = np.vstack([X_train, X_new])
            y_train = np.vstack([y_train, y_new])
            dataset = self._create_dataset(X_train, y_train)
        
        # Save results
        self._save_results(results, 'adaptive_sampling_results.json')
        
        return results
    
    def compare_methods(
        self,
        uniform_results: Dict,
        adaptive_results: Dict
    ) -> Dict:
        """
        Compare uniform vs adaptive sampling
        
        Args:
            uniform_results: Results from uniform sampling
            adaptive_results: Results from adaptive sampling
            
        Returns:
            Comparison metrics
        """
        logger.info("\n" + "=" * 80)
        logger.info("COMPARISON: UNIFORM VS ADAPTIVE SAMPLING")
        logger.info("=" * 80)
        
        # Find samples needed to reach target accuracy
        uniform_samples_for_target = None
        adaptive_samples_for_target = None
        
        for i, r2 in enumerate(uniform_results['r2_scores']):
            if r2 >= self.target_accuracy:
                uniform_samples_for_target = uniform_results['sample_sizes'][i]
                break
        
        for i, r2 in enumerate(adaptive_results['r2_scores']):
            if r2 >= self.target_accuracy:
                adaptive_samples_for_target = adaptive_results['sample_sizes'][i]
                break
        
        # Calculate reduction
        if uniform_samples_for_target and adaptive_samples_for_target:
            reduction = (
                (uniform_samples_for_target - adaptive_samples_for_target) /
                uniform_samples_for_target * 100
            )
        else:
            reduction = None
        
        # Final accuracy comparison
        uniform_final_r2 = uniform_results['r2_scores'][-1]
        adaptive_final_r2 = adaptive_results['r2_scores'][-1]
        
        uniform_final_samples = uniform_results['sample_sizes'][-1]
        adaptive_final_samples = adaptive_results['sample_sizes'][-1]
        
        comparison = {
            'uniform_samples_for_target': uniform_samples_for_target,
            'adaptive_samples_for_target': adaptive_samples_for_target,
            'reduction_percentage': reduction,
            'uniform_final_r2': uniform_final_r2,
            'adaptive_final_r2': adaptive_final_r2,
            'uniform_final_samples': uniform_final_samples,
            'adaptive_final_samples': adaptive_final_samples,
            'target_accuracy': self.target_accuracy
        }
        
        # Print comparison
        logger.info(f"\nTarget Accuracy: R² >= {self.target_accuracy:.2f}")
        logger.info(f"\nSamples needed to reach target:")
        logger.info(f"  Uniform:  {uniform_samples_for_target or 'Not reached'}")
        logger.info(f"  Adaptive: {adaptive_samples_for_target or 'Not reached'}")
        
        if reduction is not None:
            logger.info(f"\n✓ Reduction: {reduction:.1f}%")
            
            if reduction >= 50:
                logger.info(f"  SUCCESS: Meets target of 50-70% reduction!")
            elif reduction >= 30:
                logger.info(f"  GOOD: Significant reduction achieved")
            else:
                logger.info(f"  MODEST: Some reduction but below target")
        
        logger.info(f"\nFinal Accuracy:")
        logger.info(f"  Uniform:  R²={uniform_final_r2:.4f} ({uniform_final_samples} samples)")
        logger.info(f"  Adaptive: R²={adaptive_final_r2:.4f} ({adaptive_final_samples} samples)")
        
        # Save comparison
        self._save_results(comparison, 'comparison_results.json')
        
        return comparison
    
    def generate_comparison_plots(
        self,
        uniform_results: Dict,
        adaptive_results: Dict,
        comparison: Dict
    ):
        """
        Generate comparison plots
        
        Args:
            uniform_results: Uniform sampling results
            adaptive_results: Adaptive sampling results
            comparison: Comparison metrics
        """
        logger.info("\nGenerating comparison plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: R² vs Samples
        ax = axes[0, 0]
        ax.plot(
            uniform_results['sample_sizes'],
            uniform_results['r2_scores'],
            'o-', linewidth=2, markersize=8,
            label='Uniform Sampling', color='blue'
        )
        ax.plot(
            adaptive_results['sample_sizes'],
            adaptive_results['r2_scores'],
            's-', linewidth=2, markersize=8,
            label='Adaptive Sampling', color='red'
        )
        ax.axhline(
            y=self.target_accuracy, color='green',
            linestyle='--', linewidth=2, label=f'Target R²={self.target_accuracy:.2f}'
        )
        ax.set_xlabel('Number of Samples', fontsize=12)
        ax.set_ylabel('R² Score', fontsize=12)
        ax.set_title('Model Accuracy vs Dataset Size', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: MSE vs Samples
        ax = axes[0, 1]
        ax.plot(
            uniform_results['sample_sizes'],
            uniform_results['mse_values'],
            'o-', linewidth=2, markersize=8,
            label='Uniform Sampling', color='blue'
        )
        ax.plot(
            adaptive_results['sample_sizes'],
            adaptive_results['mse_values'],
            's-', linewidth=2, markersize=8,
            label='Adaptive Sampling', color='red'
        )
        ax.set_xlabel('Number of Samples', fontsize=12)
        ax.set_ylabel('Validation MSE', fontsize=12)
        ax.set_title('Validation Error vs Dataset Size', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Efficiency comparison
        ax = axes[1, 0]
        
        if comparison['reduction_percentage'] is not None:
            categories = ['Uniform', 'Adaptive']
            samples = [
                comparison['uniform_samples_for_target'],
                comparison['adaptive_samples_for_target']
            ]
            colors = ['blue', 'red']
            
            bars = ax.bar(categories, samples, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
            ax.set_ylabel('Samples to Reach Target', fontsize=12)
            ax.set_title(
                f'Efficiency Comparison\n({comparison["reduction_percentage"]:.1f}% Reduction)',
                fontsize=14, fontweight='bold'
            )
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, sample in zip(bars, samples):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2., height,
                    f'{int(sample)}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold'
                )
        else:
            ax.text(
                0.5, 0.5, 'Target not reached\nby both methods',
                ha='center', va='center', fontsize=12,
                transform=ax.transAxes
            )
            ax.set_title('Efficiency Comparison', fontsize=14, fontweight='bold')
        
        # Plot 4: Training time comparison
        ax = axes[1, 1]
        ax.plot(
            uniform_results['sample_sizes'],
            np.cumsum(uniform_results['training_times']),
            'o-', linewidth=2, markersize=8,
            label='Uniform (cumulative)', color='blue'
        )
        ax.plot(
            adaptive_results['sample_sizes'],
            np.cumsum(adaptive_results['training_times']),
            's-', linewidth=2, markersize=8,
            label='Adaptive (cumulative)', color='red'
        )
        ax.set_xlabel('Number of Samples', fontsize=12)
        ax.set_ylabel('Cumulative Training Time (s)', fontsize=12)
        ax.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'adaptive_vs_uniform_comparison.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"Comparison plot saved: {plot_path}")
        plt.close()
        
        # Additional plot: Learning curves
        self._plot_learning_curves(uniform_results, adaptive_results)
    
    def _plot_learning_curves(
        self,
        uniform_results: Dict,
        adaptive_results: Dict
    ):
        """Plot detailed learning curves"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(
            uniform_results['sample_sizes'],
            uniform_results['r2_scores'],
            'o-', linewidth=3, markersize=10,
            label='Uniform Sampling', color='blue', alpha=0.7
        )
        ax.plot(
            adaptive_results['sample_sizes'],
            adaptive_results['r2_scores'],
            's-', linewidth=3, markersize=10,
            label='Adaptive Sampling', color='red', alpha=0.7
        )
        ax.axhline(
            y=self.target_accuracy, color='green',
            linestyle='--', linewidth=2, label=f'Target (R²={self.target_accuracy:.2f})'
        )
        
        ax.set_xlabel('Number of Training Samples', fontsize=14)
        ax.set_ylabel('R² Score (Validation Set)', fontsize=14)
        ax.set_title(
            'Learning Curves: Adaptive vs Uniform Sampling',
            fontsize=16, fontweight='bold'
        )
        ax.legend(fontsize=12, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.0])
        
        plt.tight_layout()
        plot_path = self.output_dir / 'learning_curves.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"Learning curves saved: {plot_path}")
        plt.close()
    
    def _simulate_samples(self, samples: list) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate a list of parameter samples"""
        X = self._samples_to_array(samples)
        y = self._simulate_array(X)
        return X, y
    
    def _samples_to_array(self, samples: list) -> np.ndarray:
        """Convert list of parameter dicts to array"""
        param_names = list(self.parameter_space.keys())
        X = np.array([[s[name] for name in param_names] for s in samples])
        return X
    
    def _simulate_array(self, X: np.ndarray) -> np.ndarray:
        """Simulate points from array"""
        param_names = list(self.parameter_space.keys())
        results = []
        
        for x in X:
            params = {name: float(x[i]) for i, name in enumerate(param_names)}
            result = self.simulator.run(params)
            
            if hasattr(result, 'observables'):
                results.append(list(result.observables.values()))
            else:
                results.append(list(result.values()))
        
        return np.array(results)
    
    def _create_dataset(self, X: np.ndarray, y: np.ndarray) -> Dataset:
        """Create dataset from arrays"""
        param_names = list(self.parameter_space.keys())
        
        # Get observable names
        test_params = {name: self.parameter_space[name][0] for name in param_names}
        test_result = self.simulator.run(test_params)
        
        if hasattr(test_result, 'observables'):
            obs_names = list(test_result.observables.keys())
        else:
            obs_names = list(test_result.keys())
        
        # Remove purity if present
        if 'purity' in obs_names:
            purity_idx = obs_names.index('purity')
            obs_names.remove('purity')
            y = np.delete(y, purity_idx, axis=1)
        
        return Dataset(
            X=X, y=y,
            parameter_names=param_names,
            observable_names=obs_names,
            metadata={}
        )
    
    def _save_results(self, results: Dict, filename: str):
        """Save results to JSON"""
        # Convert numpy types to Python types
        results_serializable = {}
        for key, value in results.items():
            if isinstance(value, (list, tuple)):
                results_serializable[key] = [
                    float(v) if hasattr(v, 'item') else v for v in value
                ]
            elif hasattr(value, 'item'):
                results_serializable[key] = float(value)
            else:
                results_serializable[key] = value
        
        path = self.output_dir / filename
        with open(path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        logger.info(f"Results saved: {path}")


def main():
    """Main validation script"""
    logger.info("=" * 80)
    logger.info("ADAPTIVE SAMPLING VALIDATION - Task 11.4")
    logger.info("=" * 80)
    
    # Initialize simulator
    logger.info("\nInitializing quantum circuit simulator...")
    simulator = EnhancedQuantumCircuitSimulator()
    
    # Create validator
    validator = AdaptiveSamplingValidator(
        simulator=simulator,
        target_accuracy=0.85,  # Target R² = 0.85
        max_samples=5000,
        output_dir='./results/adaptive_validation'
    )
    
    # Run uniform sampling baseline
    uniform_results = validator.run_uniform_sampling(
        sample_sizes=[500, 1000, 1500, 2000, 2500, 3000, 4000, 5000]
    )
    
    # Run adaptive sampling
    adaptive_results = validator.run_adaptive_sampling(
        initial_samples=500,
        samples_per_iteration=300,
        max_iterations=10
    )
    
    # Compare methods
    comparison = validator.compare_methods(uniform_results, adaptive_results)
    
    # Generate plots
    validator.generate_comparison_plots(
        uniform_results,
        adaptive_results,
        comparison
    )
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION COMPLETE - SUMMARY")
    logger.info("=" * 80)
    
    if comparison['reduction_percentage'] is not None:
        reduction = comparison['reduction_percentage']
        logger.info(f"\n✓ Adaptive sampling achieved {reduction:.1f}% reduction in samples")
        
        if reduction >= 50:
            logger.info("  ✓ SUCCESS: Meets requirement of 50-70% reduction!")
        elif reduction >= 30:
            logger.info("  ✓ GOOD: Significant improvement demonstrated")
        else:
            logger.info("  ⚠ MODEST: Some improvement but below target")
    else:
        logger.info("\n⚠ Target accuracy not reached by both methods")
    
    logger.info(f"\nResults saved to: {validator.output_dir}")
    logger.info("\nTask 11.4 validation complete!")


if __name__ == '__main__':
    main()
