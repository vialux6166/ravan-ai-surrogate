#!/usr/bin/env python3
"""
Quick training script to demonstrate the Ravan system
Generates synthetic data and trains an MLP model
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tempfile
from pathlib import Path
from quantum_circuit_simulator import QuantumCircuitSimulator
from dataset import Dataset
from mlp_regressor import MLPRegressor
from model_base import ModelConfig

print("="*70)
print("RAVAN QUICK TRAINING DEMONSTRATION")
print("="*70)

# Step 1: Generate training data
print("\n[1/3] Generating 100 quantum circuit samples...")
simulator = QuantumCircuitSimulator()

X_train = []
y_train = []

for i in range(100):
    # Random parameters
    params = {
        'n_qubits': 2,
        'shots': 8192,
        'apply_hadamard': np.random.choice([0, 1]),
        'apply_cnot': np.random.choice([0, 1])
    }
    
    # Run simulation
    result = simulator.run(params)
    
    # Extract features and targets
    features = [
        params['apply_hadamard'],
        params['apply_cnot'],
        params['shots'] / 8192.0  # Normalize shots
    ]
    
    targets = [
        result.observables.get('entropy', 0),
        result.observables.get('fidelity_to_ghz', 0)
    ]
    
    X_train.append(features)
    y_train.append(targets)
    
    if (i + 1) % 20 == 0:
        print(f"  Generated {i+1}/100 samples...")

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"[OK] Generated {len(X_train)} samples")
print(f"  Features: {X_train.shape[1]} parameters")
print(f"  Targets: {y_train.shape[1]} observables")

# Step 2: Create and split dataset
print("\n[2/3] Creating dataset and splitting...")
dataset = Dataset(
    X=X_train,
    y=y_train,
    parameter_names=['apply_hadamard', 'apply_cnot', 'shots_norm'],
    observable_names=['entropy', 'fidelity_to_ghz']
)

train_ds, val_ds = dataset.split(train_ratio=0.8)
print(f"[OK] Split into {train_ds.n_samples} train, {val_ds.n_samples} val samples")

# Step 3: Train model
print("\n[3/3] Training MLP model...")
config = ModelConfig(
    model_type='mlp',
    input_dim=train_ds.n_features,
    output_dim=train_ds.n_targets,
    hyperparameters={'epochs': 50, 'batch_size': 32, 'patience': 5}
)

model = MLPRegressor(config)

# Train with auto-split (no separate val data needed)
metrics = model.train(
    X_train=train_ds.X,
    y_train=train_ds.y,
    X_val=val_ds.X,
    y_val=val_ds.y
)

print(f"\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"Final R² score: {metrics.r2_score:.4f}")
print(f"Training time: {metrics.training_time:.2f}s")
print(f"Final validation MSE: {metrics.val_mse:.6f}")
print("="*70)

# Step 4: Test predictions
print("\n[Bonus] Testing predictions on new data...")
test_params = {
    'n_qubits': 2,
    'shots': 8192,
    'apply_hadamard': 1,
    'apply_cnot': 1
}

# Get ground truth
true_result = simulator.run(test_params)
ground_truth = np.array([
    true_result.observables.get('entropy', 0),
    true_result.observables.get('fidelity_to_ghz', 0)
])

# Predict
X_test = np.array([[test_params['apply_hadamard'], test_params['apply_cnot'], test_params['shots'] / 8192.0]])
prediction = model.predict(X_test)

print("\nGround Truth vs Prediction:")
for i, name in enumerate(dataset.observable_names):
    print(f"  {name}: {ground_truth[i]:.4f} -> {prediction[0][i]:.4f}")

print("\n[OK] Training demonstration complete!")

