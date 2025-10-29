#!/usr/bin/env python3
"""
Export Trained Model to ONNX Format
Converts PyTorch model to ONNX for TensorRT optimization
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import torch
from mlp_regressor import MLPRegressor
from model_base import ModelConfig

print("="*70)
print("STARTING ONNX EXPORT")
print("="*70)

MODEL_DIR = "models/worldclass_quantum_ai"
MODEL_PATH = f"{MODEL_DIR}/model.pt"
CONFIG_PATH = f"{MODEL_DIR}/config.json"
ONNX_PATH = f"{MODEL_DIR}/model.onnx"

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model not found at {MODEL_PATH}")
    print("Please run train_worldclass_model.py first to create the model.")
    sys.exit(1)

# 1. Load the model configuration
print(f"\n[1/4] Loading config from {CONFIG_PATH}...")
with open(CONFIG_PATH, 'r') as f:
    config_dict = json.load(f)

input_dim = config_dict.get('input_dim', 3)
output_dim = config_dict.get('output_dim', 2)
hyperparams = config_dict.get('hyperparameters', {})

print(f"  Input dimension: {input_dim}")
print(f"  Output dimension: {output_dim}")
print(f"  Hyperparameters: {list(hyperparams.keys())}")

# Re-create config and model objects
config = ModelConfig(
    model_type='mlp',
    input_dim=input_dim,
    output_dim=output_dim,
    hyperparameters=hyperparams
)

model = MLPRegressor(config)

# 2. Load the trained weights
print(f"\n[2/4] Loading trained weights from {MODEL_PATH}...")
model.load(MODEL_DIR)
model.model.eval()  # Set model to evaluation mode (very important!)
print(f"  [OK] Model loaded successfully")

# 2.5. Move model to CPU for ONNX export (ONNX prefers CPU)
print(f"\n[2.5/4] Moving model to CPU for ONNX export...")
model.model = model.model.cpu()
print(f"  [OK] Model moved to CPU")

# 3. Create a dummy input tensor (on CPU for ONNX export)
print(f"\n[3/4] Creating dummy input tensor...")
dummy_input = torch.randn(1, input_dim, device='cpu')
print(f"  Input shape: {dummy_input.shape}")
print(f"  Input dtype: {dummy_input.dtype}")
print(f"  Device: cpu (for ONNX export)")

# Test the model first
with torch.no_grad():
    test_output = model.model(dummy_input)
    print(f"  Test output shape: {test_output.shape}")
    print(f"  [OK] Model forward pass successful")

# 4. Export the model to ONNX
print(f"\n[4/4] Exporting model to {ONNX_PATH}...")
torch.onnx.export(
    model.model,               # The PyTorch model
    dummy_input,               # Model input (already on CPU)
    ONNX_PATH,                 # Where to save the model
    export_params=True,        # Store the trained weights
    opset_version=11,          # The ONNX version to use
    do_constant_folding=True,  # Perform optimizations
    input_names=['parameters'], # The model's input names
    output_names=['observables'], # The model's output names
    dynamic_axes={            # Allow variable batch sizes
        'parameters': {0: 'batch_size'},
        'observables': {0: 'batch_size'}
    }
)

print("\n" + "="*70)
print("ONNX EXPORT SUCCESSFUL!")
print("="*70)
print(f"Model saved to: {ONNX_PATH}")

# Verify the export
if os.path.exists(ONNX_PATH):
    file_size = os.path.getsize(ONNX_PATH) / (1024 ** 2)  # Convert to MB
    print(f"File size: {file_size:.2f} MB")
    print("\nNext step: Convert to TensorRT")
    print("  Run: python convert_to_tensorrt.py")
    print("="*70)
else:
    print("[ERROR] ONNX file was not created")
    sys.exit(1)

