#!/usr/bin/env python3
"""
Quantize ONNX model to INT8 for faster CPU inference.
Generates models/worldclass_quantum_ai/model_quant.onnx
"""

import os
from pathlib import Path

from onnxruntime.quantization import quantize_dynamic, QuantType

MODEL_DIR = Path("models/worldclass_quantum_ai")
ONNX_IN = MODEL_DIR / "model.onnx"
ONNX_OUT = MODEL_DIR / "model_quant.onnx"

if not ONNX_IN.exists():
    raise FileNotFoundError(f"Missing ONNX model: {ONNX_IN}")

print(f"Quantizing {ONNX_IN} -> {ONNX_OUT}")
quantize_dynamic(
    model_input=str(ONNX_IN),
    model_output=str(ONNX_OUT),
    weight_type=QuantType.QInt8,
    optimize_model=True
)
print("Done.")
