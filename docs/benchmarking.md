# Benchmarking RAVAN Inverse Design

This document explains how to reproduce and cite the inverse-design benchmark for RAVAN's AI surrogate model.

## Overview

The benchmark demonstrates the performance of RAVAN's inverse design system using a **Basin-Hopping + L-BFGS-B hybrid optimizer** with local ONNX Runtime inference.

| Metric | Result |
|--------|--------|
| **Success Rate** | **83.0% (83/100)** |
| **Mean Error** | **0.010229** |
| **Mean Time** | **35.8 ms per run** |
| **Method** | Basin-Hopping (niter=25) + L-BFGS-B (maxiter=50) |
| **Threshold** | Error < 0.01 |

### Key Improvements

- **Success Rate**: 0% → 83% (strict 0.01 error threshold)
- **Runtime**: 44 seconds → 35.8 ms per run (~1,230× speedup)
- **Strategy**: HTTP API calls → Local ONNX inference (~18× faster)
- **Optimizer**: Single L-BFGS-B → Basin-Hopping + L-BFGS-B hybrid

## Quick Start

### Basic Run

Run the full benchmark suite (all 4 sections):

```bash
python run_benchmarks.py
```

### Quick Test

Run a reduced 10-run test for faster validation:

```bash
python run_benchmarks.py --quick
```

### Analyze Results

Generate publication-ready visualizations:

```bash
python analyze_benchmarks.py --input benchmark_results.json --outdir reports/figures
```

## Output Files

- **`benchmark_results.json`** – Complete raw results and summary statistics
- **`reports/figures/`** – Generated figures:
  - `error_histogram.png` – Error distribution
  - `time_vs_error.png` – Speed vs accuracy trade-off
  - `success_rate.png` – Success rate visualization
  - `benchmarks_summary.csv` – Tabular summary data

## Benchmark Sections

The full benchmark includes:

1. **Accuracy Benchmarks** – R² score, RMSE, MAE (forward prediction)
2. **Latency Benchmarks** – Inference time comparison (PyTorch CPU, ONNX CPU, ONNX GPU)
3. **Inverse Design** – Optimization success rate and performance
4. **UQ Calibration** – Uncertainty quantification metrics

## Environment Setup

### Required Dependencies

```bash
pip install numpy scipy onnxruntime pandas matplotlib scikit-learn
```

### Model Requirements

- Trained model at `models/worldclass_quantum_ai/model.onnx`
- PyTorch model at `models/worldclass_quantum_ai/model.pt` (for UQ calibration)

### Hardware

- **Recommended**: CPU execution provider (optimal for single predictions)
- **Optional**: GPU execution provider for batch processing

## Reproducing Results

### Step 1: Ensure Models are Trained

```bash
python train_models_script.py
```

### Step 2: Run Benchmarks

```bash
python run_benchmarks.py
```

### Step 3: Analyze Results

```bash
python analyze_benchmarks.py
```

### Step 4: View Figures

Open `benchmark_plots/` directory to view generated visualizations.

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{ravan_benchmark_2025,
  title={RAVAN Quantum-ML Inverse Design Benchmark},
  author={Your Name},
  year={2025},
  note={Basin-Hopping + L-BFGS-B Hybrid Optimizer, 83% Success @ 35.8ms (CPU)}
}
```

Or in text:

> "RAVAN Inverse Design Benchmark – Basin-Hopping + L-BFGS-B Hybrid Optimizer, achieving 83% success rate with 35.8 ms mean runtime on CPU (2025)."

## Technical Details

### Optimization Strategy

The benchmark uses a **hybrid optimization approach**:

1. **Global Exploration**: Basin-Hopping with 25 iterations
   - Random "hops" to explore parameter space
   - Jumps out of local minima
   - Uses `stepsize=0.1`, `T=1.0`

2. **Local Refinement**: L-BFGS-B with bounded optimization
   - Fast gradient-based local search
   - Maximum 50 iterations per basin
   - Convergence tolerance: `ftol=1e-6`

### Inference Setup

- **Provider**: CPU-only (`CPUExecutionProvider`)
- **Why**: 18× faster for single predictions (~0.02ms vs ~0.38ms)
- **Model**: Quantized ONNX Runtime (if available)

### Success Criteria

- **Threshold**: Mean squared error < 0.01
- **Validation**: 100 random target configurations
- **Range**: Entropy [0.3, 1.0], Fidelity [0.3, 1.0]

## Performance Characteristics

### Latency Comparison

| Method | Mean Time | Use Case |
|--------|-----------|----------|
| ONNX CPU | 0.0207 ms | **Single predictions (optimal)** |
| PyTorch CPU | 0.1544 ms | Development/debugging |
| ONNX GPU | 0.3811 ms | Batch processing |

### Accuracy Metrics

- **R² Score**: 0.9873 (98.73% variance explained)
- **RMSE**: 0.0324
- **MAE**: 0.0274

### Inverse Design Performance

- **83% success rate** meeting strict 0.01 error threshold
- **Median time**: 31.7 ms (even faster than mean)
- **Median error**: 0.0018 (better than mean due to outliers)

## Troubleshooting

### Slow Performance

If inverse design is slow:

1. Check execution provider: Should show `CPUExecutionProvider`
2. Disable unnecessary logging
3. Reduce `niter` if testing (default is 25)

### Low Success Rate

If success rate is lower than expected:

1. Check threshold: Should be 0.01
2. Verify model accuracy (R² > 0.98)
3. Try increasing `niter` in basin-hopping

### Import Errors

If encountering import errors:

```bash
pip install --upgrade numpy scipy onnxruntime pandas matplotlib scikit-learn
```

## Contributing

To add new benchmark sections or improve existing ones:

1. Modify `run_benchmarks.py`
2. Add visualization in `visualize_benchmarks.py`
3. Update this documentation
4. Submit a pull request

## License

See main project LICENSE file.

## Contact

For questions or issues with the benchmark suite, please open an issue on GitHub.

