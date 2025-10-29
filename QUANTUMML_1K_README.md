# QuantumML-1K Benchmark Dataset

**Version:** 1.0  
**Release Date:** 2025-10-19  
**DOI:** [Placeholder - To be assigned upon publication]

## Overview

QuantumML-1K is a standardized benchmark dataset for evaluating machine learning models on quantum simulation tasks. The dataset contains 1,000 carefully curated samples from quantum circuit simulations, designed to test model performance across diverse parameter regimes.

## Dataset Composition

| Category | Samples | Description |
|----------|---------|-------------|
| **Uniform Coverage** | 500 | Latin Hypercube Sampling across full parameter space |
| **Edge Cases** | 200 | Boundary conditions and extreme parameter values |
| **Challenging Regimes** | 200 | High entanglement, deep circuits, low shot counts |
| **Validation** | 100 | Simple cases with known analytical solutions |
| **Total** | **1000** | |

## Features (Input Parameters)

The dataset includes 10 input parameters for quantum circuit configuration:

1. **n_qubits** (2-6): Number of qubits in the circuit
2. **circuit_depth** (1-5): Depth of the quantum circuit
3. **rx_angle_0** (0-2π): Rotation angle around X-axis (qubit 0)
4. **ry_angle_0** (0-2π): Rotation angle around Y-axis (qubit 0)
5. **rz_angle_0** (0-2π): Rotation angle around Z-axis (qubit 0)
6. **rx_angle_1** (0-2π): Rotation angle around X-axis (qubit 1)
7. **ry_angle_1** (0-2π): Rotation angle around Y-axis (qubit 1)
8. **rz_angle_1** (0-2π): Rotation angle around Z-axis (qubit 1)
9. **entangling_pattern** (0-3): Type of entangling gate pattern
10. **shots** (1024-8192): Number of measurement shots

## Targets (Quantum Observables)

The dataset includes 7 quantum observables:

1. **entanglement_entropy**: Von Neumann entropy (measure of entanglement)
2. **fidelity_to_ghz**: Fidelity to GHZ state
3. **expect_x**: Expectation value of Pauli X operator
4. **expect_y**: Expectation value of Pauli Y operator
5. **expect_z**: Expectation value of Pauli Z operator
6. **circuit_depth_actual**: Actual circuit depth after compilation
7. **two_qubit_gate_count**: Number of two-qubit gates

## Difficulty Scores

Each sample includes a difficulty score computed using model uncertainty (MC Dropout). Higher scores indicate samples that are more challenging for ML models to predict accurately.

**Difficulty Distribution:**
- Easy (score < 33rd percentile): ~333 samples
- Medium (33rd-66th percentile): ~333 samples
- Hard (score > 66th percentile): ~334 samples

## File Format

The dataset is provided in HDF5 format for efficient storage and loading:

```
quantumml_1k.h5
├── metadata/
│   ├── name: "QuantumML-1K"
│   ├── version: "1.0"
│   ├── creation_date
│   ├── simulator: "EnhancedQuantumCircuitSimulator"
│   └── ...
├── parameters/
│   ├── n_qubits: float[1000]
│   ├── circuit_depth: float[1000]
│   └── ... (10 parameters total)
└── observables/
    ├── entanglement_entropy: float[1000]
    ├── fidelity_to_ghz: float[1000]
    └── ... (7 observables total)
```

## Usage

### Loading the Dataset (Python)

```python
from pipeline.hdf5_storage import HDF5Storage

# Load dataset
storage = HDF5Storage()
dataset = storage.load_dataset('quantumml_1k.h5')

print(f"Samples: {dataset.n_samples}")
print(f"Features: {dataset.n_features}")
print(f"Targets: {dataset.n_targets}")

# Access data
X = dataset.X  # Input parameters (1000, 10)
y = dataset.y  # Quantum observables (1000, 7)
```

### Training a Model

```python
from pipeline.training import TrainingPipeline

# Initialize pipeline
pipeline = TrainingPipeline()

# Prepare data (80/20 split)
train_dataset, val_dataset = pipeline.prepare_data(
    dataset, 
    train_ratio=0.8, 
    normalize=True
)

# Train model
model, metrics = pipeline.train_model('mlp', train_dataset, val_dataset)

print(f"R² Score: {metrics.r2_score:.4f}")
print(f"MSE: {metrics.val_mse:.6f}")
```

## Evaluation Metrics

Models should be evaluated using the following metrics:

1. **R² Score**: Coefficient of determination (higher is better)
2. **MSE**: Mean Squared Error (lower is better)
3. **MAE**: Mean Absolute Error (lower is better)
4. **Physics Constraint Violation**: For applicable observables
5. **Per-Observable Performance**: Separate metrics for each target

## Leaderboard

Submit your results to be included in the official leaderboard!

| Rank | Model | R² Score | MSE | MAE | Date | Reference |
|------|-------|----------|-----|-----|------|-----------|
| 1 | MLP (256-128-64) | 0.297 | 0.703 | 0.600 | 2025-10-19 | Baseline |
| 2 | XGBoost (500 trees) | 0.214 | 0.786 | 0.609 | 2025-10-19 | Baseline |
| - | *Your model here* | - | - | - | - | - |

### Submission Guidelines

To submit your results:

1. Train your model on the training split (80%)
2. Evaluate on the validation split (20%)
3. Report all metrics (R², MSE, MAE)
4. Include model architecture and hyperparameters
5. Submit via GitHub issue or email

## Physics Validation

All samples in the dataset have been validated against known physics constraints:

- ✓ Probability conservation
- ✓ Entanglement bounds
- ✓ Measurement statistics
- ✓ Gate count consistency

**Validation Pass Rate:** 100% (all 1000 samples)

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{quantumml1k2025,
  title={QuantumML-1K: A Benchmark Dataset for Quantum Machine Learning},
  author={[Authors]},
  year={2025},
  publisher={[Publisher]},
  doi={[DOI - To be assigned]},
  url={[URL]}
}
```

## License

This dataset is released under the [MIT License / CC BY 4.0 / Your chosen license].

## Contact

For questions, issues, or contributions:
- GitHub: [Repository URL]
- Email: [Contact email]
- Issues: [GitHub Issues URL]

## Changelog

### Version 1.0 (2025-10-19)
- Initial release
- 1,000 samples from quantum circuit simulations
- 10 input parameters, 7 quantum observables
- Difficulty scores included
- Full physics validation

## Acknowledgments

This dataset was generated using:
- **Qiskit**: Quantum circuit simulation
- **PyTorch**: Machine learning framework
- **NumPy/SciPy**: Numerical computations

## Future Versions

Planned additions for future versions:
- Additional simulators (Schrödinger, Harmonic Oscillator)
- Noise models and error mitigation
- Time-series quantum dynamics
- Larger parameter spaces (10K, 100K samples)

---

**QuantumML-1K** - Advancing Quantum Machine Learning Research
