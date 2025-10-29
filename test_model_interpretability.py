"""
Test SHAP-based model interpretability
"""

import sys
sys.path.insert(0, '/home/windows/ravan-quantum-ml/src')

import numpy as np
from model_interpretability import analyze_model_interpretability
from pipeline.hdf5_storage import HDF5Storage
from pipeline.training import TrainingPipeline
from pipeline.dataset import Dataset

print("=" * 80)
print("Testing Model Interpretability with SHAP")
print("=" * 80)

# Load dataset
print("\n1. Loading dataset...")
storage = HDF5Storage()
dataset = storage.load_dataset('./data/raw/quantum_circuit_500_20251019_153622.h5')

# Remove purity column
purity_idx = 1
new_y = np.delete(dataset.y, purity_idx, axis=1)
new_obs_names = [
    'entanglement_entropy', 'fidelity_to_ghz', 'expect_x', 
    'expect_y', 'expect_z', 'circuit_depth_actual', 'two_qubit_gate_count'
]

dataset = Dataset(
    X=dataset.X,
    y=new_y,
    parameter_names=dataset.parameter_names,
    observable_names=new_obs_names,
    metadata=dataset.metadata
)

print(f"Dataset: {dataset.n_samples} samples")
print(f"Parameters: {dataset.parameter_names}")
print(f"Observables: {dataset.observable_names}")

# Prepare data
print("\n2. Preparing data and training model...")
pipeline = TrainingPipeline()
train_dataset, val_dataset = pipeline.prepare_data(dataset, train_ratio=0.8, normalize=True)

# Train MLP model
print("\n3. Training MLP model...")
model, metrics = pipeline.train_model('mlp', train_dataset, val_dataset)

print(f"Model trained: R²={metrics.r2_score:.4f}, MSE={metrics.val_mse:.6f}")

# Run interpretability analysis
print("\n4. Running SHAP interpretability analysis...")
print("   (This may take a few minutes...)")

try:
    interpreter = analyze_model_interpretability(
        model=model,
        model_type='mlp',
        X_train=train_dataset.X,
        X_test=val_dataset.X[:50],  # Use first 50 test samples
        parameter_names=dataset.parameter_names,
        observable_names=dataset.observable_names,
        output_dir='./results/interpretability_test'
    )
    
    print("\n✓ SHAP analysis completed successfully!")
    print("\nCheck ./results/interpretability_test/ for visualizations:")
    print("  - shap_summary_*.png: Feature importance for each observable")
    print("  - shap_summary_overall.png: Overall feature importance")
    print("  - force_plot_*.png: Individual prediction explanations")
    
except ImportError as e:
    print(f"\n⚠ SHAP library not installed: {e}")
    print("\nTo install SHAP, run:")
    print("  pip install shap")
    print("\nSkipping SHAP analysis...")

except Exception as e:
    print(f"\n✗ Error during SHAP analysis: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Model Interpretability Test Complete!")
print("=" * 80)
