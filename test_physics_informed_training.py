"""
Test physics-informed loss integration with MLP training
"""

import sys
sys.path.insert(0, '/home/windows/ravan-quantum-ml/src')

import numpy as np
import torch
from physics_informed_loss import PhysicsInformedLoss
from models.base import ModelConfig
from models.mlp import MLPRegressor

print("=" * 80)
print("Testing Physics-Informed MLP Training")
print("=" * 80)

# Generate synthetic data simulating Schrödinger equation
print("\n1. Generating synthetic quantum tunneling data...")
n_samples = 500
n_features = 5  # barrier_height, barrier_width, energy, momentum, position
n_observables = 5  # transmission, reflection, energy_out, momentum_out, position_out

# Generate random features
X = np.random.rand(n_samples, n_features).astype(np.float32)

# Generate targets with physics constraint: T + R ≈ 1
# Transmission and reflection should sum to 1
T = np.random.rand(n_samples, 1).astype(np.float32)
R = (1.0 - T + np.random.randn(n_samples, 1) * 0.01).astype(np.float32)  # Small noise
other_obs = np.random.rand(n_samples, 3).astype(np.float32)

y = np.hstack([T, R, other_obs])

observable_names = ['transmission', 'reflection', 'energy_out', 'momentum_out', 'position_out']

print(f"Data shape: X={X.shape}, y={y.shape}")
print(f"Observables: {observable_names}")

# Check initial constraint satisfaction
initial_constraint = np.mean((T + R - 1.0) ** 2)
print(f"Initial constraint violation (T+R-1)²: {initial_constraint:.6f}")

# Split data
split_idx = int(0.8 * n_samples)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

print(f"Train: {X_train.shape}, Val: {X_val.shape}")

# Test 1: Train without physics constraints (baseline)
print("\n" + "=" * 80)
print("Test 1: Training WITHOUT Physics Constraints (Baseline)")
print("=" * 80)

config_baseline = ModelConfig(
    model_type='mlp',
    input_dim=n_features,
    output_dim=n_observables,
    hyperparameters={
        'hidden_layers': [64, 32],
        'dropout': 0.2,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 50,
        'patience': 10
    },
    save_path='./test_baseline_model'
)

model_baseline = MLPRegressor(config_baseline)
metrics_baseline = model_baseline.train(X_train, y_train, X_val, y_val)

print(f"\nBaseline Results:")
print(f"  R²: {metrics_baseline.r2_score:.4f}")
print(f"  MSE: {metrics_baseline.val_mse:.6f}")
print(f"  MAE: {metrics_baseline.val_mae:.6f}")

# Check constraint violation on predictions
y_pred_baseline = model_baseline.predict(X_val)
T_pred = y_pred_baseline[:, 0]
R_pred = y_pred_baseline[:, 1]
constraint_violation_baseline = np.mean((T_pred + R_pred - 1.0) ** 2)
print(f"  Constraint violation (T+R-1)²: {constraint_violation_baseline:.6f}")

# Test 2: Train with physics constraints
print("\n" + "=" * 80)
print("Test 2: Training WITH Physics Constraints")
print("=" * 80)

config_physics = ModelConfig(
    model_type='mlp',
    input_dim=n_features,
    output_dim=n_observables,
    hyperparameters={
        'hidden_layers': [64, 32],
        'dropout': 0.2,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 50,
        'patience': 10
    },
    save_path='./test_physics_model'
)

model_physics = MLPRegressor(config_physics)

# Create physics-informed loss
physics_loss = PhysicsInformedLoss(
    constraint_weight=0.5,  # Higher weight for stronger constraint
    constraints=['schrodinger']
)

print(f"Physics loss: {physics_loss.get_constraint_info()}")

# Train with physics constraints
metrics_physics = model_physics.train(
    X_train, y_train, X_val, y_val,
    custom_loss_fn=physics_loss,
    observable_names=observable_names
)

print(f"\nPhysics-Informed Results:")
print(f"  R²: {metrics_physics.r2_score:.4f}")
print(f"  MSE: {metrics_physics.val_mse:.6f}")
print(f"  MAE: {metrics_physics.val_mae:.6f}")

# Check constraint violation on predictions
y_pred_physics = model_physics.predict(X_val)
T_pred_physics = y_pred_physics[:, 0]
R_pred_physics = y_pred_physics[:, 1]
constraint_violation_physics = np.mean((T_pred_physics + R_pred_physics - 1.0) ** 2)
print(f"  Constraint violation (T+R-1)²: {constraint_violation_physics:.6f}")

# Comparison
print("\n" + "=" * 80)
print("Comparison: Baseline vs Physics-Informed")
print("=" * 80)

print(f"\nR² Score:")
print(f"  Baseline: {metrics_baseline.r2_score:.4f}")
print(f"  Physics:  {metrics_physics.r2_score:.4f}")
print(f"  Difference: {metrics_physics.r2_score - metrics_baseline.r2_score:+.4f}")

print(f"\nConstraint Violation (T+R-1)²:")
print(f"  Baseline: {constraint_violation_baseline:.6f}")
print(f"  Physics:  {constraint_violation_physics:.6f}")
print(f"  Reduction: {(1 - constraint_violation_physics/constraint_violation_baseline)*100:.1f}%")

if constraint_violation_physics < constraint_violation_baseline:
    print("\n✓ Physics-informed loss REDUCED constraint violations!")
else:
    print("\n⚠ Physics-informed loss did not reduce violations (may need tuning)")

print("\n" + "=" * 80)
print("Physics-Informed Training Test Complete!")
print("=" * 80)
