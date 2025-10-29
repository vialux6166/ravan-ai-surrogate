#!/usr/bin/env python3
"""
Train a Local AI Model - Quick Demo
Trains a small MLP to learn quantum circuit patterns
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
from quantum_circuit_simulator import QuantumCircuitSimulator
from mlp_regressor import MLPRegressor
from model_base import ModelConfig

print("="*70)
print("TRAINING LOCAL AI MODEL - QUANTUM CIRCUIT PATTERNS")
print("="*70)

# Step 1: Generate comprehensive training data
print("\n[1/3] Generating 500 diverse quantum circuit samples...")
simulator = QuantumCircuitSimulator()

X_train = []
y_train = []

variations = [
    (0, 0),  # Product state
    (1, 0),  # Superposition only
    (0, 1),  # Entanglement only
    (1, 1),  # Bell state
]

shots_options = [1024, 2048, 4096, 8192]

for i in range(500):
    # Random variation
    apply_h, apply_c = variations[np.random.randint(0, len(variations))]
    
    params = {
        'n_qubits': 2,
        'shots': shots_options[np.random.randint(0, len(shots_options))],
        'apply_hadamard': apply_h,
        'apply_cnot': apply_c
    }
    
    # Run simulation
    result = simulator.run(params)
    
    # Extract features and targets
    features = [
        params['apply_hadamard'],
        params['apply_cnot'],
        params['shots'] / 8192.0  # Normalize
    ]
    
    targets = [
        result.observables.get('entropy', 0),
        result.observables.get('fidelity_to_ghz', 0)
    ]
    
    X_train.append(features)
    y_train.append(targets)
    
    if (i + 1) % 50 == 0:
        print(f"  Generated {i+1}/500 samples...")

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"[OK] Generated {len(X_train)} samples")
print(f"  Features: {X_train.shape[1]} (hadamard, cnot, shots_norm)")
print(f"  Targets: {y_train.shape[1]} (entropy, fidelity)")

# Step 2: Train model with more capacity
print("\n[2/3] Training MLP with enhanced architecture...")

config = ModelConfig(
    model_type='mlp',
    input_dim=X_train.shape[1],
    output_dim=y_train.shape[1],
    hyperparameters={
        'hidden_layers': [512, 256, 128],  # Larger network
        'epochs': 100,
        'batch_size': 64,
        'patience': 15,
        'learning_rate': 0.001,
        'dropout': 0.2
    }
)

model = MLPRegressor(config)

# Split data
from sklearn.model_selection import train_test_split
X_t, X_val, y_t, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

print(f"  Training samples: {X_t.shape[0]}")
print(f"  Validation samples: {X_val.shape[0]}")
print(f"  Network: {config.input_dim} -> [512, 256, 128] -> {config.output_dim}")
print(f"  This may take 30-60 seconds...")

# Train
metrics = model.train(
    X_train=X_t,
    y_train=y_t,
    X_val=X_val,
    y_val=y_val
)

print(f"\n[OK] Training complete!")
print(f"  Training time: {metrics.training_time:.2f}s")
print(f"  Final R² score: {metrics.r2_score:.4f}")
print(f"  Final validation MSE: {metrics.val_mse:.6f}")
print(f"  Best epoch: {metrics.best_epoch + 1}/{100}")

# Step 3: Test on edge cases
print("\n[3/3] Testing on quantum states...")

test_cases = [
    ("Product State |00>", {'apply_hadamard': 0, 'apply_cnot': 0, 'shots': 8192}),
    ("Superposition Only", {'apply_hadamard': 1, 'apply_cnot': 0, 'shots': 8192}),
    ("Bell State", {'apply_hadamard': 1, 'apply_cnot': 1, 'shots': 8192}),
]

print("\nTest Results:")
print("-" * 70)
print(f"{'State':<20} {'True':>12} {'Predicted':>12} {'Error':>12}")
print("-" * 70)

for name, params in test_cases:
    # Get ground truth
    result = simulator.run(params)
    true_entropy = result.observables.get('entropy', 0)
    true_fidelity = result.observables.get('fidelity_to_ghz', 0)
    
    # Predict
    X_test = np.array([[
        params['apply_hadamard'],
        params['apply_cnot'],
        params['shots'] / 8192.0
    ]])
    pred = model.predict(X_test)[0]
    
    pred_entropy, pred_fidelity = pred[0], pred[1]
    
    entropy_error = abs(true_entropy - pred_entropy)
    fidelity_error = abs(true_fidelity - pred_fidelity)
    
    print(f"{name}:")
    print(f"  Entropy:    {true_entropy:8.4f} -> {pred_entropy:8.4f} (err: {entropy_error:.4f})")
    print(f"  Fidelity:   {true_fidelity:8.4f} -> {pred_fidelity:8.4f} (err: {fidelity_error:.4f})")
    print()

# Summary
print("="*70)
print("LOCAL AI MODEL TRAINING COMPLETE!")
print("="*70)
print(f"Model successfully learned quantum circuit patterns.")
print(f"Average prediction error: {entropy_error + fidelity_error:.4f}")
print(f"R² score: {metrics.r2_score:.4f} ({metrics.r2_score*100:.1f}% variance explained)")
print("="*70)

