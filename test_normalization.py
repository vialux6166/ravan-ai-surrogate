#!/usr/bin/env python3
"""
Quick test to verify data normalization is working correctly
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

# Generate test data similar to our parameters
np.random.seed(42)
X = np.random.randn(100, 5)

# Scale to realistic parameter ranges (like our actual data)
X[:, 0] = X[:, 0] * 2 + 5      # V0: 3-7
X[:, 1] = X[:, 1] * 0.3 + 1.5  # barrier_width: 1.2-1.8
X[:, 2] = X[:, 2] * 1 + 4      # k0: 3-5
X[:, 3] = X[:, 3] * 0.2 + 1    # sigma: 0.8-1.2
X[:, 4] = X[:, 4] * 0.5 - 5    # x0: -5.5 to -4.5

print("Before Normalization:")
print(f"  Mean: {X.mean(axis=0)}")
print(f"  Std:  {X.std(axis=0)}")
print(f"  Min:  {X.min(axis=0)}")
print(f"  Max:  {X.max(axis=0)}")

# Normalize
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

print("\nAfter Normalization:")
print(f"  Mean: {X_norm.mean(axis=0)}")
print(f"  Std:  {X_norm.std(axis=0)}")
print(f"  Min:  {X_norm.min(axis=0)}")
print(f"  Max:  {X_norm.max(axis=0)}")

# Check if normalization is correct
mean_check = np.allclose(X_norm.mean(axis=0), 0, atol=1e-10)
std_check = np.allclose(X_norm.std(axis=0), 1, atol=1e-10)

print(f"\n✓ Mean ~0: {mean_check}")
print(f"✓ Std ~1: {std_check}")

if mean_check and std_check:
    print("\n✅ Normalization is working correctly!")
else:
    print("\n❌ Normalization has issues!")
