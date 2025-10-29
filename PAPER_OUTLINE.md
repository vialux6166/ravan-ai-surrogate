# Ravan: Physics-Informed Quantum Surrogate Modeling with Real-Time Inverse Design

## 1. Introduction
- Motivation: quantum simulators are accurate but slow for design loops
- Contribution: fast physics-informed surrogate, UQ, and inverse design UI

## 2. Methods
### 2.1 Dataset and Simulators
- Quantum circuit simulator (Hadamard, CNOT, shots)
- Targets: entropy, fidelity-to-GHZ

### 2.2 Model
- MLP with physics-informed loss (optional), dropout for UQ
- Mixed-precision and modern PyTorch APIs

### 2.3 Inference Optimization
- ONNX export, ONNX Runtime CPU/GPU, INT8 quantization
- Optional TensorRT and provider selection

### 2.4 Inverse Design
- Continuous optimization (scipy.minimize) to match target observables

### 2.5 Uncertainty Quantification
- Monte Carlo Dropout; mean ± std returned via API

## 3. System Architecture
- FastAPI backend (ONNX, optional PyTorch for UQ/gradients)
- Streamlit dashboard with tabs (Direct Simulation, Inverse Design)

## 4. Experiments
### 4.1 Accuracy
- Validation/Test R²: 0.9877 (98.8%)
- Validation MSE: 7.69e-4
- Test MSE: see models/repro/metrics.json (typical ~1e-3)

### 4.2 Speed
- PyTorch CPU: ~1.0 ms
- ONNX Runtime CPU: ~0.02 ms
- ONNX Runtime GPU (RTX 3090): ~0.15 ms (avg, 0.10–0.22 ms range)
- ONNX INT8 CPU: expected further reduction vs ONNX FP32 CPU; supported via quantize_onnx.py

### 4.3 Inverse Design Success
- Target (entropy=0.95, fidelity=0.95) → Solution found in < 200 ms (L-BFGS-B)
- Optimal parameters (example): [apply_hadamard=1.0, apply_cnot=1.0, shots_norm≈0.5]
- Achieved metrics (example): entropy=0.9542, fidelity=0.9675
- Error from target: |Δentropy|≈0.0042, |Δfidelity|≈0.0175

### 4.4 UQ Calibration
- Error vs predicted std; reliability plots

## 5. Case Studies
- Direct simulation exploration
- Inverse design to high-fidelity Bell-like states
  - Example inputs → prediction: [1, 1, 0.5] → [0.9542, 0.9675]

## 6. Related Work
- PINNs, surrogate modeling for quantum, UQ methods

## 7. Conclusion
- Summary, limitations, and future work (Bayesian optimization, multi-objective)

## Appendix
- Reproducibility details (seeds, versions), dataset link, repo link
