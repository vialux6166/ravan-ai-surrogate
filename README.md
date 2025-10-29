# Ravan AI: Quantum Surrogate & Inverse Design Engine

[![Python CI](https://github.com/vialux6166/ravan-ai-surrogate/actions/workflows/ci.yml/badge.svg)](https://github.com/vialux6166/ravan-ai-surrogate/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/ravan-sdk.svg)](https://badge.fury.io/py/ravan-sdk)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub issues](https://img.shields.io/github/issues/vialux6166/ravan-ai-surrogate)](https://github.com/vialux6166/ravan-ai-surrogate/issues)

Ravan is a high-performance AI system designed to accelerate quantum physics simulations. It uses a Physics-Informed Neural Network (PINN) to create a surrogate model that predicts quantum circuit outcomes thousands of times faster than traditional simulators, enabling real-time analysis and optimization.

The system's **"killer feature"** is an **Inverse Design Engine** that finds the optimal circuit parameters to achieve your desired physical outcomes (e.g., "find parameters for 0.95 fidelity") in milliseconds.

This repository contains the full source code, physics simulators, AI models, and the **`ravan-sdk`** Python package.

## Performance

- Accuracy: R² 0.987
- Inference Speed: ~0.02 ms (ONNX CPU)
- Inverse Design: ~38 ms (Avg. Solution Time)

## 🚀 World-Class Performance

| Metric | Result | Notes |
|--------|--------|-------|
| **AI Model Accuracy** | **98.7% (R²)** | Trained on a 1000-sample test set vs. original simulator. |
| **Inference Speed** | **~0.02 ms** | Single prediction on ONNX (CPU). ~8.4x faster than PyTorch. |
| **Inverse Design Speed** | **~36 ms** | Avg. time to find an optimal solution (83% success rate). |
| **UQ Calibration** | **100% Coverage** | Uncertainty model (MC Dropout) is robust and conservative. |

## ⚡ Quick Start (using the SDK)

Ravan is available as a Python package.

### 1. Installation

```bash
pip install ravan-sdk
```

*(Note: The SDK's "local" mode requires the full project contents, but "http" mode works standalone).*

### 2. Example Usage

Create a Python file (e.g., `test_ravan.py`) and run the following. This example assumes you have the API server running locally (see Step 3).

```python
from ravan_sdk import Ravan

try:
    # Initialize in "auto" mode.
    # It will try to connect to http://localhost:8000
    # If it fails, it will fall back to local models.
    client = Ravan(mode="auto")

    # --- 1. Fast AI Prediction ---
    # Input params: [apply_hadamard, apply_cnot, shots_norm]
    params = [1.0, 1.0, 0.5]
    res = client.predict(params)
    print(f"Prediction: {res['predictions'][0]}")

    # --- 2. Inverse Design ---
    # Find parameters to match a target
    print("\nRunning Inverse Design...")
    solution = client.inverse.run(
        target_entropy=0.95,
        target_fidelity=0.95,
        niter=25  # Number of global optimization steps
    )
    
    print(f"Inverse Design Results:")
    print(f"  Optimal params: {solution['x']}")
    print(f"  Final Error: {solution['fun']:.6f}")

    # --- 3. Run Original Simulator (Slower) ---
    print("\nRunning full (slow) simulation...")
    qc_params = {
        "n_qubits": 2,
        "shots": 4096,
        "apply_hadamard": 1,
        "apply_cnot": 1
    }
    sim_results = client.simulate_quantum_circuit(qc_params)
    print(f"Full Sim Results: {sim_results}")

except Exception as e:
    print(f"Error: {e}")
```

### 3. Running the (Optional) API Server

The SDK can run in a fast, local-only mode. However, you can also run the full FastAPI server to provide Ravan as a service.

```bash
# From the root of this project
cd "C:\Users\windows\Documents\rishi - professor"

# Start the server (uses fast ONNX CPU model)
uvicorn serve:app --host 0.0.0.0 --port 8000
```

## 📈 Benchmarks

The project includes a comprehensive benchmark suite (`run_benchmarks.py`) to validate performance.

### Accuracy
- **R²**: 0.9870 (on 1000-sample test set)
- **RMSE**: 0.0326
- **MAE**: 0.0276

### Latency (per single inference)
- **ONNX CPU**: ~0.018 ms (p50: 0.018 ms, p95: 0.019 ms)
- **PyTorch CPU**: ~0.153 ms (p50: 0.145 ms, p95: 0.199 ms)
- **Speedup**: ONNX is **~8.4x faster** than PyTorch CPU

### Inverse Design (100 runs; Basin-Hopping, niter=25)
- **Success Rate**: 83% at error < 0.01 (83/100 runs)
- **Mean Error**: 0.01023
- **Median Error**: 0.00179 (much tighter than mean, showing robust performance)
- **Mean Time**: 38.83 ms per optimization
- **Median Time**: 32.12 ms per optimization

### UQ Calibration (MC Dropout)
- **Coverage @ 1-sigma**: 100% (entropy and fidelity)
- **Mean Std**: Entropy 0.0913, Fidelity 0.1119

**Full benchmark results**: See `benchmark_results.json` for complete metrics.

## 🏗️ Project Architecture

This repository contains the full system:

- **`ravan_sdk/`**: The distributable `pip install ravan-sdk` package and client code.
- **`serve.py`**: The FastAPI server that serves the optimized ONNX model.
- **`streamlit_dashboard_v2.py`**: The interactive web UI for demos.
- **`models/`**: (Not in Git) Storage for trained `.pt`, `.onnx`, and `.engine` files.
- **`simulators/`**: The original, high-fidelity Python physics simulators.
- **`tests/`**: The complete suite of 210+ pytest unit, integration, and physics tests.
- **`scripts/`**: (e.g., `run_benchmarks.py`, `export_onnx.py`)
- **`docs/`**: Detailed documentation, case studies, and the arXiv paper draft.
- **`.github/workflows/`**: CI pipeline definition for automated testing.

## 📖 Documentation

- **[PAPER_OUTLINE.md](PAPER_OUTLINE.md)**: Full draft paper with all benchmark results
- **[CASE_STUDIES.md](CASE_STUDIES.md)**: Real-world use cases and examples
- **[README_DEPLOYMENT.md](README_DEPLOYMENT.md)**: Deployment guide for API server and dashboard

## 🤝 Contributing

Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request. All PRs must pass the build (pytest) status check to be merged.

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🔗 Links

- **PyPI Package**: https://pypi.org/project/ravan-sdk/
- **GitHub Repository**: https://github.com/vialux6166/ravan-ai-surrogate
- **Issues**: https://github.com/vialux6166/ravan-ai-surrogate/issues

---

**Built with ❤️ for quantum computing and machine learning**

**Version**: 0.1.0  
**Status**: ✅ Published on PyPI
