# Configuration Guide

## Overview

This guide documents all configuration options for the Ravan Quantum-ML System.

## Table of Contents

1. [Environment Configuration](#environment-configuration)
2. [Model Configuration](#model-configuration)
3. [Training Configuration](#training-configuration)
4. [Simulation Configuration](#simulation-configuration)
5. [VRAM Management](#vram-management)
6. [LLM Configuration](#llm-configuration)
7. [Best Practices](#best-practices)

---

## Environment Configuration

### GPU Setup

**File**: System environment variables

```bash
# CUDA Configuration
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export CUDA_LAUNCH_BLOCKING=1  # For debugging

# PyTorch Configuration
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Python Environment

**File**: `requirements.txt`

```txt
torch>=2.0.0
qiskit>=0.45.0
pennylane>=0.33.0
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0
xgboost>=2.0.0
matplotlib>=3.7.0
h5py>=3.9.0
transformers>=4.35.0
bitsandbytes>=0.41.0
accelerate>=0.24.0
```

**Installation**:
```bash
pip install -r requirements.txt
```

---

## Model Configuration

### MLP Regressor

**Class**: `MLPRegressor`

```python
from model_base import ModelConfig

config = ModelConfig(
    model_type='mlp',
    input_dim=5,           # Number of input parameters
    output_dim=4,          # Number of observables
    hyperparameters={
        'hidden_dims': [256, 128, 64],  # Layer sizes
        'dropout': 0.2,                  # Dropout rate
        'learning_rate': 0.001,          # Adam learning rate
        'batch_size': 128,               # Training batch size
        'epochs': 100,                   # Maximum epochs
        'early_stopping_patience': 10    # Early stopping
    }
)
```

**Parameters**:
- `hidden_dims`: List of hidden layer sizes
  - Default: `[256, 128, 64]`
  - Larger = more capacity, more memory
  - Recommended: 2-4 layers

- `dropout`: Dropout probability
  - Default: `0.2`
  - Range: `0.0` - `0.5`
  - Higher = more regularization

- `learning_rate`: Optimizer learning rate
  - Default: `0.001`
  - Range: `0.0001` - `0.01`
  - Use learning rate scheduler for best results

- `batch_size`: Training batch size
  - Default: `128`
  - Larger = faster training, more memory
  - Adjust based on available VRAM

- `epochs`: Maximum training epochs
  - Default: `100`
  - Early stopping usually triggers before this

- `early_stopping_patience`: Epochs without improvement
  - Default: `10`
  - Prevents overfitting

### XGBoost Regressor

**Class**: `XGBoostRegressor`

```python
config = ModelConfig(
    model_type='xgboost',
    input_dim=5,
    output_dim=4,
    hyperparameters={
        'n_estimators': 500,        # Number of trees
        'max_depth': 7,             # Tree depth
        'learning_rate': 0.05,      # Boosting learning rate
        'tree_method': 'gpu_hist',  # GPU acceleration
        'early_stopping_rounds': 20 # Early stopping
    }
)
```

**Parameters**:
- `n_estimators`: Number of boosting rounds
  - Default: `500`
  - More = better fit, longer training

- `max_depth`: Maximum tree depth
  - Default: `7`
  - Range: `3` - `10`
  - Deeper = more complex, risk overfitting

- `learning_rate`: Boosting learning rate
  - Default: `0.05`
  - Range: `0.01` - `0.3`
  - Lower = better generalization, slower

- `tree_method`: Training algorithm
  - `'gpu_hist'`: GPU accelerated (recommended)
  - `'hist'`: CPU histogram
  - `'exact'`: Exact greedy (slow)

---

## Training Configuration

### Dataset Generation

**File**: `training_pipeline.py`

```python
from training_pipeline import TrainingPipeline

pipeline = TrainingPipeline()

# Register simulator
pipeline.register_simulator('schrodinger', SchrodingerSolver())

# Define parameter ranges
param_ranges = {
    'V0': (2.0, 8.0),           # Barrier height
    'barrier_width': (1.0, 2.0), # Barrier width
    'k0': (3.0, 6.0),           # Initial momentum
    'sigma': (0.8, 1.2),        # Wavepacket width
    'x0': (-6.0, -4.0)          # Initial position
}

# Generate dataset
dataset = pipeline.generate_dataset(
    simulator_name='schrodinger',
    param_ranges=param_ranges,
    n_samples=1000,              # Number of samples
    sampling_strategy='lhs',     # 'lhs' or 'grid'
    n_jobs=4                     # Parallel workers
)
```

**Sampling Strategies**:
- `'lhs'`: Latin Hypercube Sampling (recommended)
  - Better coverage of parameter space
  - Fewer samples needed

- `'grid'`: Grid search
  - Systematic coverage
  - More samples required

### Training Pipeline

```python
# Split data
train_dataset, val_dataset = dataset.split(
    test_size=0.2,
    random_state=42
)

# Normalize
train_norm = train_dataset.normalize()
val_norm = val_dataset.normalize(scaler=train_norm.metadata['scaler'])

# Train model
model = MLPRegressor(config)
history = model.train(
    train_norm.X,
    train_norm.y,
    val_norm.X,
    val_norm.y
)

# Save model
model.save('models/my_model.pt')
```

---

## Simulation Configuration

### Quantum Circuit Simulator

**Class**: `QuantumCircuitSimulator`

```python
params = {
    'n_qubits': 2,              # Number of qubits
    'shots': 1000,              # Measurement shots
    'gate_sequence': 'bell'     # 'bell', 'ghz', or 'custom'
}

result = simulator.run(params)
```

**Parameters**:
- `n_qubits`: Number of qubits
  - Range: `2` - `10`
  - More = exponentially more memory

- `shots`: Number of measurements
  - Default: `1000`
  - More = better statistics

- `gate_sequence`: Circuit type
  - `'bell'`: Bell state
  - `'ghz'`: GHZ state
  - `'custom'`: Custom gates

### Schrödinger Solver

**Class**: `SchrodingerSolver`

```python
params = {
    'V0': 5.0,                  # Barrier height
    'barrier_width': 1.5,       # Barrier width
    'k0': 4.0,                  # Initial momentum
    'sigma': 1.0,               # Wavepacket width
    'x0': -5.0                  # Initial position
}

result = solver.run(params)
```

**Parameters**:
- `V0`: Potential barrier height
  - Range: `1.0` - `10.0`
  - Higher = less tunneling

- `barrier_width`: Barrier width
  - Range: `0.5` - `3.0`
  - Wider = less tunneling

- `k0`: Initial momentum
  - Range: `2.0` - `8.0`
  - Higher = more tunneling

### Harmonic Oscillator

**Class**: `HarmonicOscillator`

```python
params = {
    'oscillator_length': 1.0,   # Characteristic length
    'basis_size': 10,           # Number of basis states
    'initial_n': 2              # Initial quantum number
}

result = oscillator.run(params)
```

---

## VRAM Management

### VRAM Manager

**Class**: `VRAMManager`

```python
from vram_manager import VRAMManager, WorkloadMode

manager = VRAMManager()

# Request mode
with manager.request_mode(WorkloadMode.TRAINING):
    # Training code here
    model.train(X, y)

# Modes automatically released
```

**Workload Modes**:
- `SIMULATION`: For running quantum simulations
- `TRAINING`: For ML model training
- `INFERENCE`: For model predictions
- `LLM`: For LLM operations

**Configuration**:
```python
# Check available VRAM
free_vram = manager.gpu_accelerator.get_free_vram_gb()
print(f"Free VRAM: {free_vram:.2f} GB")

# Optimize memory
manager.gpu_accelerator.optimize_memory()
```

### Memory Optimization

**Enable gradient checkpointing**:
```python
model = MemoryEfficientModel(
    input_dim=2048,
    output_dim=1024,
    use_checkpointing=True  # Enable checkpointing
)
```

**Enable mixed precision**:
```python
scaler = torch.amp.GradScaler('cuda')

with torch.amp.autocast('cuda'):
    output = model(X)
    loss = criterion(output, y)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## LLM Configuration

### LLM Adapter

**Class**: `LLMAdapter`

```python
from llm_adapter import LLMAdapter

adapter = LLMAdapter(
    model_name='Qwen/Qwen2.5-32B-Instruct',
    load_in_4bit=True,          # 4-bit quantization
    device_map='auto',          # Automatic device placement
    max_memory={'cuda:0': '22GB'}  # VRAM limit
)
```

**Parameters**:
- `model_name`: HuggingFace model identifier
  - Default: `'Qwen/Qwen2.5-32B-Instruct'`
  - Must be downloaded locally

- `load_in_4bit`: Enable 4-bit quantization
  - Default: `True`
  - Reduces VRAM by ~75%

- `device_map`: Device placement strategy
  - `'auto'`: Automatic
  - `'cuda:0'`: Specific GPU
  - `'cpu'`: CPU only (slow)

- `max_memory`: VRAM limits per device
  - Prevents OOM errors
  - Leave headroom for other operations

### Generation Parameters

```python
response = adapter.query(
    prompt="Explain quantum tunneling",
    max_new_tokens=100,         # Maximum response length
    temperature=0.7,            # Sampling temperature
    top_p=0.9,                  # Nucleus sampling
    top_k=50                    # Top-k sampling
)
```

**Parameters**:
- `max_new_tokens`: Maximum tokens to generate
  - Default: `100`
  - More = longer responses, slower

- `temperature`: Sampling randomness
  - Range: `0.0` - `2.0`
  - Lower = more deterministic
  - Higher = more creative

- `top_p`: Nucleus sampling threshold
  - Range: `0.0` - `1.0`
  - Default: `0.9`

- `top_k`: Top-k sampling
  - Default: `50`
  - Limits vocabulary per step

---

## Best Practices

### Development

1. **Start Small**
   - Use small datasets (100-1000 samples)
   - Test with small models first
   - Validate before scaling up

2. **Monitor Resources**
   - Track VRAM usage
   - Monitor training time
   - Check GPU utilization

3. **Use Version Control**
   - Track model configurations
   - Save hyperparameters
   - Document experiments

### Production

1. **Optimize Memory**
   - Enable gradient checkpointing
   - Use mixed precision
   - Clear cache between operations

2. **Error Handling**
   - Catch OOM errors
   - Implement fallbacks
   - Log all operations

3. **Validation**
   - Verify physics constraints
   - Check prediction ranges
   - Monitor uncertainty

### Performance

1. **GPU Utilization**
   - Use appropriate batch sizes
   - Enable GPU acceleration
   - Profile bottlenecks

2. **Data Loading**
   - Use persistent workers
   - Enable pin_memory
   - Prefetch data

3. **Model Optimization**
   - Use TorchScript compilation
   - Quantize for inference
   - Prune unnecessary layers

---

## Example Configurations

### Quick Start (Development)

```python
# Small model for testing
config = ModelConfig(
    model_type='mlp',
    input_dim=5,
    output_dim=4,
    hyperparameters={
        'hidden_dims': [128, 64],
        'batch_size': 64,
        'epochs': 50
    }
)
```

### Production (High Accuracy)

```python
# Large model for production
config = ModelConfig(
    model_type='mlp',
    input_dim=5,
    output_dim=4,
    hyperparameters={
        'hidden_dims': [512, 256, 128, 64],
        'dropout': 0.3,
        'batch_size': 256,
        'epochs': 200,
        'early_stopping_patience': 20
    }
)
```

### Memory Constrained (8GB GPU)

```python
# Optimized for limited VRAM
config = ModelConfig(
    model_type='mlp',
    input_dim=5,
    output_dim=4,
    hyperparameters={
        'hidden_dims': [256, 128],
        'batch_size': 32,  # Smaller batch
        'epochs': 100
    }
)

# Enable optimizations
model.use_checkpointing = True
use_mixed_precision = True
```

---

## Troubleshooting

### Out of Memory (OOM)

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size
2. Enable gradient checkpointing
3. Use mixed precision
4. Clear cache: `torch.cuda.empty_cache()`

### Slow Training

**Symptoms**: Training takes too long

**Solutions**:
1. Increase batch size
2. Use GPU acceleration
3. Enable mixed precision
4. Use persistent DataLoader workers

### Poor Accuracy

**Symptoms**: High validation error

**Solutions**:
1. Generate more training data
2. Increase model capacity
3. Adjust learning rate
4. Use physics-informed loss

---

## Configuration Files

### Example: `config.yaml`

```yaml
model:
  type: mlp
  input_dim: 5
  output_dim: 4
  hidden_dims: [256, 128, 64]
  dropout: 0.2

training:
  batch_size: 128
  learning_rate: 0.001
  epochs: 100
  early_stopping_patience: 10

data:
  n_samples: 1000
  sampling_strategy: lhs
  test_size: 0.2

vram:
  enable_checkpointing: true
  enable_mixed_precision: true
  max_vram_gb: 20
```

### Loading Configuration

```python
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Use configuration
model_config = ModelConfig(
    model_type=config['model']['type'],
    input_dim=config['model']['input_dim'],
    output_dim=config['model']['output_dim'],
    hyperparameters={
        'hidden_dims': config['model']['hidden_dims'],
        'dropout': config['model']['dropout'],
        'batch_size': config['training']['batch_size'],
        'learning_rate': config['training']['learning_rate'],
        'epochs': config['training']['epochs']
    }
)
```

---

## Support

For questions or issues:
1. Check this documentation
2. Review example configurations
3. See tutorial notebooks
4. Check GitHub issues

**Last Updated**: 2025-10-19
