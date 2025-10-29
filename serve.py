#!/usr/bin/env python3
"""
FastAPI Server for Optimized ONNX Model
High-speed inference API for quantum ML predictions
"""

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List
import os
import json

# Optional PyTorch model for UQ and explainability
try:
    from mlp_regressor import MLPRegressor
    from model_base import ModelConfig
    HAS_TORCH_MODEL = True
except Exception:
    HAS_TORCH_MODEL = False

# --- Configuration ---
MODEL_DIR = "models/worldclass_quantum_ai"
MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")
MODEL_QUANT_PATH = os.path.join(MODEL_DIR, "model_quant.onnx")
INPUT_NAME = "parameters"  # Matches the ONNX model's input name
PYTORCH_DIR = "models/worldclass_quantum_ai"

# --- 1. Define the API data structures ---
class ModelInput(BaseModel):
    """Input schema for predictions"""
    # Example: [[apply_hadamard, apply_cnot, shots_norm]]
    parameters: List[List[float]] = Field(
        ..., 
        description="List of parameter vectors. Each vector has 3 elements: [apply_hadamard, apply_cnot, shots_norm]",
        min_length=1,
        json_schema_extra={"example": [[1, 1, 0.5]]}
    )

class PredictionResponse(BaseModel):
    """Output schema for predictions"""
    predictions: List[List[float]] = Field(
        ...,
        description="Predicted observables: [entropy, fidelity_to_ghz] for each input"
    )

# --- 2. Load the ONNX model and create the session ---
print("="*70)
print("INITIALIZING ONNX INFERENCE SERVER")
print("="*70)

use_quant = os.environ.get("USE_QUANT", "0") == "1" and os.path.exists(MODEL_QUANT_PATH)
model_to_load = MODEL_QUANT_PATH if use_quant else MODEL_PATH

if not os.path.exists(model_to_load):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please run export_onnx.py first.")

# Prioritize CUDAExecutionProvider (or TensorRT if requested), then fallback to CPU
use_trt = os.environ.get("USE_TRT", "0") == "1"
providers = []
if use_trt:
    providers.append('TensorrtExecutionProvider')
# Ensure CUDAExecutionProvider is listed first (or second if using TRT)!
providers += ['CUDAExecutionProvider', 'CPUExecutionProvider']

try:
    session = ort.InferenceSession(model_to_load, providers=providers)
    print(f"Model loaded from: {model_to_load}")
    print(f"Available providers: {session.get_providers()}")
    active_provider = session.get_providers()[0]
    print(f"Active provider: {active_provider}")
    if active_provider == 'CUDAExecutionProvider' or active_provider == 'TensorrtExecutionProvider':
        print("[OK] GPU acceleration enabled")
    elif active_provider == 'CPUExecutionProvider':
        print("[WARNING] Using CPU (GPU not available or failed to load)")
except Exception as e:
    print(f"[WARNING] Failed to load with GPU providers: {e}")
    print("Falling back to CPU only...")
    try:
        session = ort.InferenceSession(model_to_load, providers=['CPUExecutionProvider'])
        print("[OK] Model loaded with CPUExecutionProvider")
    except Exception as e2:
        raise RuntimeError(f"Failed to load model even with CPU: {e2}")

# Get model info
input_shape = session.get_inputs()[0].shape
output_shape = session.get_outputs()[0].shape
print(f"Input shape: {input_shape}")
print(f"Output shape: {output_shape}")

# Optionally load PyTorch model for MC Dropout UQ and gradients
mlp_model = None
if HAS_TORCH_MODEL:
    try:
        config_path = os.path.join(PYTORCH_DIR, 'config.json')
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        config = ModelConfig(
            model_type='mlp',
            input_dim=cfg.get('input_dim', 3),
            output_dim=cfg.get('output_dim', 2),
            hyperparameters=cfg.get('hyperparameters', {})
        )
        mlp_model = MLPRegressor(config)
        mlp_model.load(PYTORCH_DIR)
        print("Loaded PyTorch model for UQ/explainability")
    except Exception as e:
        print(f"[WARNING] PyTorch model unavailable for UQ: {e}")
        mlp_model = None

# --- 3. Create the FastAPI app ---
app = FastAPI(
    title="Quantum AI Simulator API",
    description="High-speed PINN quantum surrogate model inference API",
    version="1.0.0"
)

@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "status": "running",
        "model": "worldclass_quantum_ai",
        "format": "ONNX",
        "endpoint": "/predict"
    }

@app.get("/health")
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "providers": session.get_providers(),
        "input_name": INPUT_NAME,
        "input_shape": input_shape,
        "output_shape": output_shape
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(data: ModelInput, uq: bool = Query(default=False), samples: int = Query(default=30, ge=5, le=200)):
    """
    Run predictions using the optimized ONNX model.
    
    **Input**: List of parameter vectors [[apply_hadamard, apply_cnot, shots_norm], ...]
    
    **Output**: List of predicted observables [[entropy, fidelity], ...]
    
    **Example**:
        Input: {"parameters": [[1, 1, 0.5]]}
        Output: {"predictions": [[0.9999, 0.5]]}
    """
    try:
        # 1. Convert the input to NumPy array
        input_data = np.array(data.parameters, dtype=np.float32)
        
        # Validate input shape
        if len(input_data.shape) != 2 or input_data.shape[1] != 3:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid input shape. Expected (N, 3), got {input_data.shape}"
            )
        
        if uq:
            if mlp_model is None:
                raise HTTPException(status_code=400, detail="UQ not available on this server (PyTorch model missing)")
            mean_pred, std_pred = mlp_model.predict_with_uncertainty(input_data, n_samples=samples)
            return {"predictions": mean_pred.tolist(), "std": std_pred.tolist()}
        else:
            # ONNX path (fast)
            outputs = session.run(
                None,
                {INPUT_NAME: input_data}
            )
            predictions = outputs[0].tolist()
            return {"predictions": predictions}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.get("/model/info")
def model_info():
    """Get detailed model information"""
    return {
        "path": MODEL_PATH,
        "format": "ONNX",
        "providers": session.get_providers(),
        "input_name": session.get_inputs()[0].name,
        "input_shape": session.get_inputs()[0].shape,
        "input_type": str(session.get_inputs()[0].type),
        "output_name": session.get_outputs()[0].name,
        "output_shape": session.get_outputs()[0].shape,
        "output_type": str(session.get_outputs()[0].type)
    }

# --- Explainability endpoint ---
@app.post("/explain")
def explain(data: ModelInput):
    """Return basic input sensitivities via input gradients (abs dY/dX)."""
    if mlp_model is None:
        raise HTTPException(status_code=400, detail="Explainability not available (PyTorch model missing)")
    try:
        X = np.array(data.parameters, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != 3:
            raise HTTPException(status_code=400, detail=f"Invalid input shape. Expected (N, 3), got {X.shape}")

        import torch
        mlp = mlp_model.model
        mlp.eval()
        X_t = torch.tensor(X, requires_grad=True)
        outputs = mlp(X_t)
        # Compute gradients for each output separately; return mean abs grad over batch
        sensitivities = []
        for out_idx in range(outputs.shape[1]):
            mlp.zero_grad(set_to_none=True)
            grads = torch.autograd.grad(outputs[:, out_idx].sum(), X_t, retain_graph=True)[0]
            sensitivities.append(grads.abs().mean(dim=0).detach().cpu().numpy().tolist())
        return {
            "inputs": ["apply_hadamard", "apply_cnot", "shots_norm"],
            "outputs": ["entropy", "fidelity"],
            "sensitivities": sensitivities  # list of two lists (per output)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explainability error: {str(e)}")

# --- Start message ---
print("\n" + "="*70)
print("FastAPI SERVER READY")
print("="*70)
print("Run with: uvicorn serve:app --host 0.0.0.0 --port 8000 --reload")
print("API docs: http://localhost:8000/docs")
print("="*70 + "\n")

