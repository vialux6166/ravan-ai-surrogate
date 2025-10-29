"""
Test script for uncertainty quantification validation
Tests calibration and visualization of uncertainty estimates
"""

import sys
sys.path.insert(0, '/home/windows/ravan-quantum-ml/src')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import required modules
from uncertainty_quantifier import UncertaintyQuantifier
from pipeline.hdf5_storage import HDF5Storage
from pipeline.training import TrainingPipeline
from models.base import ModelConfig
from models.mlp import MLPRegressor


def test_uncertainty_on_trained_model():
    """Test uncertainty quantification on existing trained model"""
    
    print("=" * 80)
    print("Testing Uncertainty Quantification")
    print("=" * 80)
    
    # Load dataset
    print("\n1. Loading dataset...")
    storage = HDF5Storage()
    dataset = storage.load_dataset('./data/raw/quantum_circuit_500_20251019_153622.h5')
    
    # Remove purity column (index 1)
    purity_idx = 1
    new_y = np.delete(dataset.y, purity_idx, axis=1)
    new_obs_names = [
        'entanglement_entropy', 'fidelity_to_ghz', 'expect_x', 
        'expect_y', 'expect_z', 'circuit_depth_actual', 'two_qubit_gate_count'
    ]
    
    # Create new dataset with updated values
    from pipeline.dataset import Dataset
    dataset = Dataset(
        X=dataset.X,
        y=new_y,
        parameter_names=dataset.parameter_names,
        observable_names=new_obs_names,
        metadata=dataset.metadata
    )
    
    print(f"Dataset: {dataset.n_samples} samples, {dataset.n_targets} targets")
    
    # Prepare data
    print("\n2. Preparing data...")
    pipeline = TrainingPipeline()
    train_dataset, val_dataset = pipeline.prepare_data(dataset, train_ratio=0.8, normalize=True)
    
    print(f"Training: {train_dataset.n_samples}, Validation: {val_dataset.n_samples}")
    
    # Train model
    print("\n3. Training MLP model...")
    model, metrics = pipeline.train_model('mlp', train_dataset, val_dataset)
    
    print(f"Model trained: R²={metrics.r2_score:.4f}, MSE={metrics.val_mse:.6f}")
    
    # Create uncertainty quantifier
    print("\n4. Creating uncertainty quantifier...")
    uq = UncertaintyQuantifier(
        model=model,
        method='mc_dropout',
        n_samples=50,
        confidence_level=0.95
    )
    
    # Test on validation set
    print("\n5. Computing uncertainty estimates...")
    mean_pred, std_pred = uq.predict_with_uncertainty(val_dataset.X)
    
    print(f"Predictions shape: {mean_pred.shape}")
    print(f"Uncertainty shape: {std_pred.shape}")
    print(f"Mean uncertainty: {std_pred.mean():.6f}")
    print(f"Max uncertainty: {std_pred.max():.6f}")
    print(f"Min uncertainty: {std_pred.min():.6f}")
    
    # Get confidence intervals
    print("\n6. Computing confidence intervals...")
    mean_pred, lower_bound, upper_bound = uq.get_confidence_intervals(val_dataset.X)
    
    # Calibration metrics
    print("\n7. Computing calibration metrics...")
    calibration = uq.calibration_metrics(val_dataset.X, val_dataset.y)
    
    print(f"\nCalibration Results:")
    print(f"  Coverage: {calibration['coverage']:.3f}")
    print(f"  Expected: {calibration['expected_coverage']:.3f}")
    print(f"  Calibration gap: {calibration['calibration_gap']:.3f}")
    print(f"  Interval width: {calibration['interval_width']:.6f}")
    print(f"  Prediction error: {calibration['prediction_error']:.6f}")
    
    # Identify high uncertainty samples
    print("\n8. Identifying high-uncertainty samples...")
    high_unc_indices = uq.identify_high_uncertainty_samples(val_dataset.X, top_k=10)
    total_uncertainty = uq.get_total_uncertainty(val_dataset.X)
    
    print(f"Top 10 high-uncertainty sample indices: {high_unc_indices}")
    print(f"Their uncertainties: {total_uncertainty[high_unc_indices]}")
    
    # Visualization
    print("\n9. Generating visualizations...")
    create_uncertainty_plots(
        val_dataset.y,
        mean_pred,
        std_pred,
        lower_bound,
        upper_bound,
        dataset.observable_names
    )
    
    print("\n" + "=" * 80)
    print("Uncertainty quantification validation complete!")
    print("=" * 80)
    
    return uq, calibration


def create_uncertainty_plots(
    y_true,
    y_pred,
    std_pred,
    lower_bound,
    upper_bound,
    observable_names
):
    """Create uncertainty visualization plots"""
    
    output_dir = Path('./results/uncertainty_validation')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_targets = y_true.shape[1]
    
    # Plot 1: Predictions with confidence intervals for each target
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for i in range(min(n_targets, 9)):
        ax = axes[i]
        
        # Sort by true values for better visualization
        sort_idx = np.argsort(y_true[:, i])
        
        ax.plot(y_true[sort_idx, i], label='True', color='black', linewidth=2)
        ax.plot(y_pred[sort_idx, i], label='Predicted', color='blue', linewidth=1.5)
        ax.fill_between(
            range(len(sort_idx)),
            lower_bound[sort_idx, i],
            upper_bound[sort_idx, i],
            alpha=0.3,
            color='blue',
            label='95% CI'
        )
        
        ax.set_title(observable_names[i])
        ax.set_xlabel('Sample (sorted)')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_intervals.png', dpi=150)
    print(f"Saved: {output_dir / 'confidence_intervals.png'}")
    plt.close()
    
    # Plot 2: Uncertainty vs prediction error
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for i in range(min(n_targets, 9)):
        ax = axes[i]
        
        prediction_error = np.abs(y_true[:, i] - y_pred[:, i])
        uncertainty = std_pred[:, i]
        
        ax.scatter(uncertainty, prediction_error, alpha=0.5, s=20)
        ax.set_xlabel('Uncertainty (std)')
        ax.set_ylabel('Prediction Error')
        ax.set_title(observable_names[i])
        ax.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr = np.corrcoef(uncertainty, prediction_error)[0, 1]
        ax.text(
            0.05, 0.95,
            f'Corr: {corr:.3f}',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
    
    plt.tight_layout()
    plt.savefig(output_dir / 'uncertainty_vs_error.png', dpi=150)
    print(f"Saved: {output_dir / 'uncertainty_vs_error.png'}")
    plt.close()
    
    # Plot 3: Uncertainty distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    total_uncertainty = std_pred.mean(axis=1)
    ax.hist(total_uncertainty, bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Total Uncertainty (avg std across targets)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Uncertainty Estimates')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    ax.axvline(total_uncertainty.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    ax.axvline(np.median(total_uncertainty), color='green', linestyle='--', linewidth=2, label='Median')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'uncertainty_distribution.png', dpi=150)
    print(f"Saved: {output_dir / 'uncertainty_distribution.png'}")
    plt.close()
    
    # Plot 4: Calibration plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Compute coverage at different confidence levels
    confidence_levels = np.linspace(0.1, 0.99, 20)
    observed_coverage = []
    
    for conf_level in confidence_levels:
        from scipy import stats
        z_score = stats.norm.ppf((1 + conf_level) / 2)
        
        lower = y_pred - z_score * std_pred
        upper = y_pred + z_score * std_pred
        
        within_interval = np.logical_and(y_true >= lower, y_true <= upper)
        coverage = within_interval.mean()
        observed_coverage.append(coverage)
    
    ax.plot(confidence_levels, observed_coverage, 'o-', label='Observed', linewidth=2)
    ax.plot([0, 1], [0, 1], 'r--', label='Perfect calibration', linewidth=2)
    ax.set_xlabel('Expected Coverage')
    ax.set_ylabel('Observed Coverage')
    ax.set_title('Calibration Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'calibration_plot.png', dpi=150)
    print(f"Saved: {output_dir / 'calibration_plot.png'}")
    plt.close()


if __name__ == '__main__':
    uq, calibration = test_uncertainty_on_trained_model()
