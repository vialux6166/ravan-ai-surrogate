#!/usr/bin/env python3
"""
Quick Validation - Fast version with 1000 samples
Verifies normalization fixes work correctly
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import time
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from schrodinger_solver import SchrodingerSolver
from mlp_regressor import MLPRegressor
from xgboost_regressor import XGBoostRegressor
from model_base import ModelConfig

print("\n" + "="*70)
print("QUICK VALIDATION - Verifying Normalization Fixes")
print("="*70)
print("\nUsing 1000 samples for faster validation...")

# Generate small dataset
print("\n[1/5] Generating dataset...")
simulator = SchrodingerSolver()

n_samples = 1000
X = []
y = []

for i in range(n_samples):
    if i % 200 == 0:
        print(f"  Progress: {i}/{n_samples}")
    
    # Random parameters
    params = {
        'V0': np.random.uniform(2.0, 8.0),
        'barrier_width': np.random.uniform(1.0, 2.0),
        'k0': np.random.uniform(3.0, 6.0),
        'sigma': np.random.uniform(0.8, 1.2),
        'x0': np.random.uniform(-6.0, -4.0)
    }
    
    result = simulator.run(params)
    
    X.append(list(params.values()))
    y.append([
        result.observables['transmission'],
        result.observables['reflection'],
        result.observables['total_probability'],
        result.observables['energy']
    ])

X = np.array(X)
y = np.array(y)

print(f"✓ Generated {len(X)} samples")
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")

# Split data
print("\n[2/5] Splitting and normalizing data...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nBefore normalization:")
print(f"  X_train mean: {X_train.mean(axis=0)}")
print(f"  X_train std: {X_train.std(axis=0)}")

# Normalize - CRITICAL FIX!
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_val_norm = scaler.transform(X_val)

print(f"\nAfter normalization:")
print(f"  X_train mean: {X_train_norm.mean(axis=0)}")
print(f"  X_train std: {X_train_norm.std(axis=0)}")

# Verify normalization
if np.allclose(X_train_norm.mean(axis=0), 0, atol=1e-10) and \
   np.allclose(X_train_norm.std(axis=0), 1, atol=1e-10):
    print("✓ Normalization verified!")
else:
    print("❌ Normalization failed!")
    exit(1)

# Train MLP
print("\n[3/5] Training MLP...")
mlp_config = ModelConfig(
    model_type='mlp',
    input_dim=X_train_norm.shape[1],
    output_dim=y_train.shape[1],
    hyperparameters={
        'hidden_dims': [256, 128, 64],
        'dropout': 0.2,
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 50,  # Reduced for speed
        'early_stopping_patience': 10
    }
)

mlp_model = MLPRegressor(mlp_config)

start = time.time()
history = mlp_model.train(X_train_norm, y_train, X_val_norm, y_val)
mlp_time = time.time() - start

print(f"✓ MLP trained in {mlp_time:.2f}s")
print(f"  Final train loss: {history['train_loss'][-1]:.6f}")
print(f"  Final val loss: {history['val_loss'][-1]:.6f}")

# Train XGBoost
print("\n[4/5] Training XGBoost...")
xgb_config = ModelConfig(
    model_type='xgboost',
    input_dim=X_train_norm.shape[1],
    output_dim=y_train.shape[1],
    hyperparameters={
        'n_estimators': 300,  # Reduced for speed
        'max_depth': 7,
        'learning_rate': 0.05,
        'tree_method': 'gpu_hist'
    }
)

xgb_model = XGBoostRegressor(xgb_config)

start = time.time()
xgb_model.train(X_train_norm, y_train, X_val_norm, y_val)
xgb_time = time.time() - start

print(f"✓ XGBoost trained in {xgb_time:.2f}s")

# Evaluate
print("\n[5/5] Evaluating models...")

# MLP
mlp_pred = mlp_model.predict(X_val_norm)
mlp_mae = mean_absolute_error(y_val, mlp_pred)
mlp_r2 = r2_score(y_val, mlp_pred)
mlp_mape = np.mean(np.abs((y_val - mlp_pred) / (y_val + 1e-10))) * 100

print(f"\nMLP Results:")
print(f"  MAE: {mlp_mae:.6f}")
print(f"  R²: {mlp_r2:.6f}")
print(f"  MAPE: {mlp_mape:.2f}%")

# XGBoost
xgb_pred = xgb_model.predict(X_val_norm)
xgb_mae = mean_absolute_error(y_val, xgb_pred)
xgb_r2 = r2_score(y_val, xgb_pred)
xgb_mape = np.mean(np.abs((y_val - xgb_pred) / (y_val + 1e-10))) * 100

print(f"\nXGBoost Results:")
print(f"  MAE: {xgb_mae:.6f}")
print(f"  R²: {xgb_r2:.6f}")
print(f"  MAPE: {xgb_mape:.2f}%")

# Check targets
print("\n" + "="*70)
print("VALIDATION RESULTS")
print("="*70)

mlp_pass = mlp_mape < 5.0 and mlp_r2 > 0.9
xgb_pass = xgb_mape < 5.0

print(f"\nMLP Accuracy: {'✅ PASS' if mlp_pass else '❌ FAIL'}")
print(f"  Target: MAPE < 5%, R² > 0.9")
print(f"  Actual: MAPE = {mlp_mape:.2f}%, R² = {mlp_r2:.4f}")

print(f"\nXGBoost Accuracy: {'✅ PASS' if xgb_pass else '❌ FAIL'}")
print(f"  Target: MAPE < 5%")
print(f"  Actual: MAPE = {xgb_mape:.2f}%")

# Inference latency
print(f"\nInference Latency Test:")
latencies = []
for i in range(100):
    sample = X_val_norm[i:i+1]
    start = time.perf_counter()
    _ = mlp_model.predict(sample)
    latencies.append((time.perf_counter() - start) * 1000)

p95 = np.percentile(latencies, 95)
latency_pass = p95 < 1.0

print(f"  {'✅ PASS' if latency_pass else '❌ FAIL'}")
print(f"  Target: P95 < 1ms")
print(f"  Actual: P95 = {p95:.4f}ms")

# Overall
print("\n" + "="*70)
all_pass = mlp_pass and xgb_pass and latency_pass

if all_pass:
    print("🎉 ALL TESTS PASSED - Normalization fix successful!")
    print("="*70)
    print("\nKey Improvements:")
    print(f"  ✓ MLP now learning correctly (R² = {mlp_r2:.4f})")
    print(f"  ✓ Both models meet <5% error target")
    print(f"  ✓ Inference latency excellent ({p95:.4f}ms)")
    exit(0)
else:
    print("⚠️  Some tests failed - review results above")
    print("="*70)
    exit(1)
