# Ravan Hybrid Quantum-ML System

A production-ready physics simulation and machine learning pipeline for accelerating quantum mechanics research through AI-powered predictions.

## 🎯 What It Does

Ravan combines three quantum simulators with machine learning to predict quantum behaviors in milliseconds instead of hours:

1. **Quantum Circuit Simulator** - Entanglement and quantum gates
2. **Schrödinger Solver** - Quantum tunneling through barriers
3. **Harmonic Oscillator** - Energy quantization and coherent states

## ✅ Validated Performance

All simulators pass strict physics requirements:
- Quantum Circuit: Entropy within 0.001 bits, Chi² < 1.1, KL < 0.001
- Schrödinger: T+R = 1.0 within 0.001, probability conserved
- Harmonic Oscillator: Energy spacing = 1.0 within 0.001

## 🚀 Quick Start

### 1. Generate Training Data

```bash
cd ~/ravan-quantum-ml
source venv/bin/activate

# Generate 500 quantum circuit samples
python scripts/generate_data.py --simulator quantum_circuit --n-samples 500 --output data/raw/qc_dataset.h5

# Generate 500 Schrödinger samples
python scripts/generate_data.py --simulator schrodinger --n-samples 500 --output data/raw/schrodinger_dataset.h5

# Generate 500 harmonic oscillator samples
python scripts/generate_data.py --simulator harmonic --n-samples 500 --output data/raw/harmonic_dataset.h5
```

### 2. Train ML Models

```bash
# Train MLP and XGBoost on quantum circuit data
python scripts/train_models.py --dataset data/raw/qc_dataset.h5 --models mlp xgboost --output results/qc

# Train on Schrödinger data
python scripts/train_models.py --dataset data/raw/schrodinger_dataset.h5 --models mlp xgboost --output results/schrodinger
```

### 3. Use Trained Models for Predictions

```python
from pipeline.hdf5_storage import HDF5Storage
from models.mlp import MLPRegressor
from models.base import ModelConfig

# Load trained model
config = ModelConfig(
    model_type='mlp',
    hyperparameters={},
    input_dim=4,
    output_dim=3,
    save_path='./data/models/mlp_model'
)
model = MLPRegressor(config)
model.load('./data/models/mlp_model')

# Make predictions
import numpy as np
new_params = np.array([[2, 4096, 1, 1]])  # Example parameters
predictions = model.predict(new_params)
print(f"Predictions: {predictions}")
```

## 📊 System Architecture

```
Quantum Simulators → Parameter Sweeps → Parallel Execution
         ↓
   HDF5 Dataset (compressed, with metadata)
         ↓
   ML Training (MLP + XGBoost on GPU)
         ↓
   Trained Models (<1ms inference)
```

## 🔧 Features

- **GPU-Accelerated**: CUDA support for PyTorch and XGBoost
- **VRAM Management**: Intelligent memory management for RTX 3090
- **Parallel Execution**: Multi-core CPU parallelization
- **Efficient Sampling**: Latin Hypercube Sampling for optimal coverage
- **Physics Validation**: Automatic constraint checking
- **Uncertainty Quantification**: MC Dropout for confidence intervals
- **Reproducible**: SHA256 hashing and random seed tracking

## 📁 Project Structure

```
~/ravan-quantum-ml/
├── src/
│   ├── simulators/     # Quantum simulators
│   ├── models/         # ML models
│   ├── pipeline/       # Training pipeline
│   ├── gpu/            # GPU management
│   └── utils/          # Utilities
├── scripts/            # Executable scripts
├── data/               # Data storage
└── results/            # Training results
```

## 🎓 Documentation

See `.kiro/specs/ravan-hybrid-quantum-ml-system/` for:
- `requirements.md` - Detailed requirements
- `design.md` - System architecture
- `tasks.md` - Implementation tasks

## 🔬 Performance

- **Data Generation**: 10-200 samples/second
- **MLP Training**: ~5 minutes for 10k samples
- **XGBoost Training**: ~3 minutes for 10k samples
- **Inference**: <1ms per prediction

## 💻 Hardware Requirements

- NVIDIA RTX 3090 (24 GB VRAM)
- 32 GB System RAM
- WSL2 Ubuntu 22.04/24.04
- CUDA 12.x or later

## 📝 Citation

TBD - Research publication in progress
