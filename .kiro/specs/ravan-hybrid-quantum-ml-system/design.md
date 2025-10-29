# Design Document

## Overview

The Ravan Hybrid Quantum-ML System is a three-layer architecture combining quantum physics simulations, machine learning prediction models, and an optional natural language interface. The system is designed for single RTX 3090 GPU deployment on WSL (Ubuntu 22.04 LTS) and follows a modular plugin-based architecture for extensibility.

**Core Design Principles:**
- Separation of concerns: simulation, training, and inference are independent modules
- GPU-first architecture: all compute-intensive operations leverage CUDA acceleration
- Data-driven workflow: simulations generate training data, ML models learn patterns
- Offline-capable: no cloud dependencies, fully local execution
- Extensible: plugin architecture for new simulation types and ML models
- VRAM-aware: intelligent memory management to prevent GPU OOM errors

## Hardware Specifications

| Component        | Specification           | Usage                                    |
| ---------------- | ----------------------- | ---------------------------------------- |
| GPU              | NVIDIA GeForce RTX 3090 | ML training, simulation acceleration     |
| **VRAM (GPU)**   | **24 GB GDDR6X**        | **GPU computations, model weights**      |
| **System RAM**   | **32 GB DDR4**          | **Data loading, preprocessing, CPU tasks** |
| CUDA Cores       | 10,496                  | Parallel computation                     |
| Tensor Cores     | 3rd gen (Ampere)        | Mixed precision training                 |
| FP32 Performance | ~35.6 TFLOPS            | Single precision operations              |
| Memory Bandwidth | 936 GB/s                | GPU memory throughput                    |

**Critical Distinction:**
- **VRAM (24 GB)**: GPU memory for PyTorch, XGBoost GPU, Qiskit Aer GPU, and LLM - **this is the bottleneck**
- **System RAM (32 GB)**: CPU memory for data loading, multiprocessing, preprocessing - **separate from GPU**

**VRAM Budget Allocation:**
```
Standard Workload (Simulation + ML Training):
├── PyTorch MLP/LSTM:        2-4 GB
├── XGBoost GPU:             2-3 GB  
├── Qiskit Aer GPU:          1-5 GB (depends on qubit count)
├── CuPy arrays:             1-3 GB
├── System overhead:         1-2 GB
└── Available buffer:        8-15 GB
Total: ~10-17 GB used, 7-14 GB free

LLM Workload (Mutually Exclusive):
├── Qwen 30B (4-bit):        20-22 GB
├── System overhead:         1-2 GB
└── Available buffer:        0-2 GB
Total: ~22-24 GB used, 0-2 GB free
```

**Design Implication:** LLM must be **strictly optional and mutually exclusive** with simulation/ML workloads due to VRAM constraints.

## Architecture

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   CLI Tool   │  │  Python API  │  │  LLM Interface   │  │
│  │              │  │              │  │   (Optional)     │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                   Orchestration Layer                        │
│  ┌────────────────────────────────────────────────────────┐ │
│  │           Training Pipeline Controller                  │ │
│  │  - Data generation coordination                        │ │
│  │  - Model training workflow                             │ │
│  │  - Validation and metrics                              │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
┌───────────────────┐                  ┌────────────────────┐
│ Simulation Engine │                  │   ML Predictor     │
│                   │                  │                    │
│ ┌───────────────┐ │                  │ ┌────────────────┐ │
│ │ Quantum       │ │                  │ │ MLP Regressor  │ │
│ │ Circuit       │ │                  │ │                │ │
│ │ Simulator     │ │                  │ └────────────────┘ │
│ └───────────────┘ │                  │                    │
│                   │                  │ ┌────────────────┐ │
│ ┌───────────────┐ │                  │ │ XGBoost        │ │
│ │ Schrödinger   │ │                  │ │ Regressor      │ │
│ │ Solver        │ │                  │ └────────────────┘ │
│ └───────────────┘ │                  │                    │
│                   │                  │ ┌────────────────┐ │
│ ┌───────────────┐ │                  │ │ LSTM/CNN       │ │
│ │ Harmonic      │ │                  │ │ (Time Series)  │ │
│ │ Oscillator    │ │                  │ └────────────────┘ │
│ └───────────────┘ │                  └────────────────────┘
└───────────────────┘                           │
        │                                       │
        └───────────────────┬───────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                  GPU Acceleration Layer                      │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  PyTorch CUDA Backend / NumPy GPU / Qiskit Aer GPU     │ │
│  │  - CUDA 12.x                                           │ │
│  │  - cuDNN for neural networks                           │ │
│  │  - Mixed precision (FP16/FP32)                         │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                    Hardware Layer                            │
│         NVIDIA RTX 3090 (24GB VRAM) on WSL Ubuntu           │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow Architecture

```
Parameter Space Definition
         │
         ▼
┌─────────────────────┐
│  Parameter Sweep    │ ──► Parallel execution on CPU/GPU
│  Generator          │     (multiprocessing + CUDA)
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Simulation Engine  │ ──► Quantum computations
│  (Qiskit/NumPy)     │     Generate observables
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Training Dataset   │ ──► HDF5 or Parquet format
│  (params → obs)     │     10k-100k samples
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  ML Training        │ ──► GPU-accelerated training
│  (PyTorch/XGBoost)  │     80/20 train/val split
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Trained Model      │ ──► Serialized weights
│  + Metrics          │     (PyTorch .pt / pickle)
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Inference Engine   │ ──► <1ms predictions
│  (Production)       │     New parameters → observables
└─────────────────────┘
```

## Components and Interfaces

### 1. Simulation Engine

**Purpose:** Generate quantum simulation data through parameter sweeps

**Interface:**
```python
class SimulationModule(ABC):
    @abstractmethod
    def run(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Execute simulation with given parameters, return observables"""
        pass
    
    @abstractmethod
    def get_parameter_space(self) -> Dict[str, Tuple[float, float]]:
        """Return valid parameter ranges for sweeps"""
        pass
    
    @abstractmethod
    def validate_output(self, result: Dict[str, float]) -> bool:
        """Verify physics constraints (conservation laws, etc.)"""
        pass
```

**Implementations:**

#### QuantumCircuitSimulator
- **Dependencies:** Qiskit, Qiskit Aer (GPU backend)
- **Input Parameters:**
  - `n_qubits`: Number of qubits (default: 2)
  - `gate_sequence`: List of gate operations
  - `shots`: Number of measurements (default: 8192)
  - `noise_model`: Optional noise parameters
- **Output Observables:**
  - `entropy`: Von Neumann entropy (bits)
  - `chi_squared`: Goodness-of-fit metric
  - `kl_divergence`: KL divergence from ideal
  - `measurement_probs`: Probability distribution
- **Validation:** Entropy ≤ log₂(dim), probabilities sum to 1.0

#### SchrödingerSolver
- **Dependencies:** NumPy, SciPy (sparse matrices), CuPy (GPU arrays)
- **Input Parameters:**
  - `V0`: Barrier height (eV)
  - `barrier_width`: Barrier width (nm)
  - `k0`: Initial momentum
  - `sigma`: Wavepacket width
  - `x0`: Initial position
  - `L`: System size
  - `Nx`: Grid points
  - `dt`: Time step
  - `n_steps`: Number of time steps
- **Output Observables:**
  - `transmission`: Transmission coefficient T
  - `reflection`: Reflection coefficient R
  - `total_probability`: Should equal 1.0
  - `energy`: Average energy
- **Validation:** T + R = 1.0 ± 0.001, probability conservation

#### HarmonicOscillatorModule
- **Dependencies:** NumPy, SciPy (eigensolvers)
- **Input Parameters:**
  - `oscillator_length`: Characteristic length a
  - `basis_size`: Number of basis states N
  - `initial_n`: Initial photon number
  - `time_points`: Array of time values
  - `driving_field`: Optional external field
- **Output Observables:**
  - `energy_levels`: Eigenvalues Eₙ
  - `photon_number`: Expected ⟨n⟩(t)
  - `coherence`: Coherence measure
  - `position_variance`: ⟨x²⟩
- **Validation:** Eₙ₊₁ - Eₙ = 1.0 ± 0.001, ⟨n⟩ conservation

### 2. Training Pipeline

**Purpose:** Orchestrate data generation, model training, and validation

**Interface:**
```python
class TrainingPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.simulators: List[SimulationModule] = []
        self.models: Dict[str, MLModel] = {}
    
    def register_simulator(self, name: str, simulator: SimulationModule):
        """Add simulation module to pipeline"""
        pass
    
    def generate_dataset(self, n_samples: int, output_path: str) -> Dataset:
        """Run parameter sweeps and save training data"""
        pass
    
    def train_models(self, dataset: Dataset, model_types: List[str]):
        """Train specified ML models on dataset"""
        pass
    
    def validate(self, test_dataset: Dataset) -> Dict[str, Metrics]:
        """Evaluate models on held-out test set"""
        pass
    
    def run_full_pipeline(self):
        """Execute end-to-end: generate → train → validate"""
        pass
```

**Key Methods:**

- **generate_dataset():**
  - Uses Latin Hypercube Sampling or grid search for initial parameter space coverage
  - Implements adaptive sampling / active learning to prioritize high-uncertainty regions
  - Logs edge cases separately (high decoherence, resonant tunneling, near-critical behavior)
  - Parallelizes simulations using `multiprocessing.Pool` or `joblib`
  - Saves to HDF5 format with compression for efficient I/O
  - Includes metadata: parameter ranges, simulation versions, timestamps, random seed, dataset version hash

- **train_models():**
  - Splits data 80/20 train/validation
  - Normalizes inputs using StandardScaler
  - Trains multiple model types in parallel where possible
  - Uses early stopping and cross-validation
  - Saves best models based on validation MSE

- **validate():**
  - Computes MSE, MAE, R², max absolute error
  - Generates prediction vs. actual scatter plots
  - Checks physics constraints on predictions (e.g., probability conservation)
  - Creates residual analysis plots
  - Computes uncertainty calibration metrics
  - Generates SHAP values for model interpretability (feature importance)

### 3. ML Predictor

**Purpose:** Fast inference on trained models

**Interface:**
```python
class MLModel(ABC):
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        """Train model on data"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions"""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Serialize model to disk"""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load model from disk"""
        pass
```

**Implementations:**

#### MLPRegressor (PyTorch)
- **Architecture:**
  - Input layer: variable size based on parameter count
  - Hidden layers: [256, 128, 64] neurons with ReLU activation
  - Dropout: 0.2 for regularization
  - Output layer: variable size based on observable count
  - Optional: Physics-informed layers to enforce constraints (e.g., R + T = 1)
- **Training:**
  - Optimizer: Adam with learning rate 0.001
  - Loss: MSE + optional physics constraint penalty
  - Batch size: 128
  - Epochs: 100 with early stopping (patience=10)
  - GPU acceleration via PyTorch CUDA
  - Mixed precision (FP16) with gradient scaling
- **Uncertainty Quantification:**
  - Monte Carlo dropout for prediction confidence intervals
  - Optional: Ensemble of 3-5 models for robust uncertainty estimates
- **Inference:** Batch prediction on GPU, <1ms per sample

#### XGBoostRegressor
- **Hyperparameters:**
  - `n_estimators`: 500
  - `max_depth`: 7
  - `learning_rate`: 0.05
  - `tree_method`: 'gpu_hist' for GPU acceleration
  - `objective`: 'reg:squarederror'
- **Training:** GPU-accelerated tree building
- **Inference:** CPU-based prediction (very fast)

#### LSTMPredictor (for time-series)
- **Architecture:**
  - LSTM layers: 2 layers with 128 hidden units
  - Fully connected output layer
  - Sequence length: 50 time steps
- **Use case:** Predicting time evolution in harmonic oscillator
- **Training:** Similar to MLP with sequence batching

### 4. GPU Accelerator

**Purpose:** Manage GPU resources and provide acceleration utilities with VRAM contention management

**Implementation:**
```python
class GPUAccelerator:
    def __init__(self, vram_limit_gb: float = 24.0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vram_limit_gb = vram_limit_gb
        self.gpu_info = self._get_gpu_info()
        self.llm_loaded = False
    
    def _get_gpu_info(self) -> Dict:
        """Query GPU properties"""
        if torch.cuda.is_available():
            return {
                'name': torch.cuda.get_device_name(0),
                'vram_total_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3),
                'vram_allocated_gb': torch.cuda.memory_allocated(0) / (1024**3),
                'vram_reserved_gb': torch.cuda.memory_reserved(0) / (1024**3),
                'cuda_version': torch.version.cuda
            }
        return {'name': 'CPU', 'vram_total_gb': 0}
    
    def get_free_vram_gb(self) -> float:
        """Get available VRAM in GB"""
        if not torch.cuda.is_available():
            return 0.0
        free, total = torch.cuda.mem_get_info(0)
        return free / (1024**3)
    
    def check_vram_available(self, required_gb: float, operation: str) -> bool:
        """Check if sufficient VRAM is available for operation"""
        free_vram = self.get_free_vram_gb()
        if free_vram < required_gb:
            raise RuntimeError(
                f"Insufficient VRAM for {operation}. "
                f"Required: {required_gb:.1f} GB, Available: {free_vram:.1f} GB. "
                f"Close other GPU processes or reduce batch size."
            )
        return True
    
    def prevent_llm_conflict(self):
        """Ensure LLM is not loaded when running simulations/training"""
        if self.llm_loaded:
            raise RuntimeError(
                "LLM is currently loaded and using ~20-22 GB VRAM. "
                "Unload LLM before running simulations or training. "
                "LLM and simulation/training are mutually exclusive."
            )
    
    def to_gpu(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Move data to GPU with VRAM check"""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        return data.to(self.device)
    
    def enable_mixed_precision(self) -> torch.cuda.amp.GradScaler:
        """Enable FP16 training to reduce VRAM by ~30%"""
        return torch.cuda.amp.GradScaler()
    
    def optimize_memory(self):
        """Clear cache and optimize VRAM usage"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def log_vram_usage(self, stage: str):
        """Log current VRAM usage for monitoring"""
        info = self._get_gpu_info()
        logging.info(
            f"[{stage}] VRAM: {info['vram_allocated_gb']:.2f} GB allocated, "
            f"{info['vram_reserved_gb']:.2f} GB reserved, "
            f"{self.get_free_vram_gb():.2f} GB free"
        )
```

**GPU Utilization Strategy:**
- Simulations: Use Qiskit Aer GPU backend for circuit simulation, CuPy for Schrödinger solver
- Training: PyTorch CUDA for neural networks, XGBoost GPU for gradient boosting
- Inference: Batch predictions on GPU for neural networks
- Memory management: Monitor VRAM before each major operation, use mixed precision when needed
- **LLM Isolation:** LLM loading is gated by VRAM checks and marked as mutually exclusive with other GPU workloads

**VRAM Management Rules:**
1. Check available VRAM before loading LLM (requires 22+ GB free)
2. Prevent simulation/training when LLM is loaded
3. Log VRAM usage at key pipeline stages
4. Automatically enable mixed precision if VRAM is constrained
5. Clear GPU cache between major operations

### 5. VRAM Contention Management

**Problem:** RTX 3090 has 24 GB VRAM (not 32 GB). LLM requires 20-22 GB, leaving insufficient memory for concurrent simulation/training.

**Solution:** Implement strict VRAM management with mutually exclusive workload modes.

**Workload Modes:**

```python
class WorkloadMode(Enum):
    SIMULATION = "simulation"  # Quantum simulations + data generation
    TRAINING = "training"      # ML model training
    INFERENCE = "inference"    # ML model inference (lightweight)
    LLM = "llm"               # LLM operations (mutually exclusive)

class VRAMManager:
    def __init__(self, gpu_accelerator: GPUAccelerator):
        self.gpu = gpu_accelerator
        self.current_mode = None
        self.mode_vram_requirements = {
            WorkloadMode.SIMULATION: 8.0,   # GB
            WorkloadMode.TRAINING: 10.0,    # GB
            WorkloadMode.INFERENCE: 3.0,    # GB
            WorkloadMode.LLM: 22.0          # GB
        }
    
    def request_mode(self, mode: WorkloadMode) -> bool:
        """Request to switch to a workload mode"""
        required_vram = self.mode_vram_requirements[mode]
        
        # Check if switching from LLM to other modes
        if self.current_mode == WorkloadMode.LLM and mode != WorkloadMode.LLM:
            raise RuntimeError(
                "Cannot switch from LLM mode to other modes without unloading LLM. "
                "Call unload_llm() first."
            )
        
        # Check if switching to LLM from other modes
        if mode == WorkloadMode.LLM and self.current_mode is not None:
            free_vram = self.gpu.get_free_vram_gb()
            if free_vram < required_vram:
                raise RuntimeError(
                    f"Cannot load LLM. Required: {required_vram} GB, "
                    f"Available: {free_vram:.1f} GB. "
                    f"Close simulation/training processes first."
                )
        
        # Verify VRAM availability
        self.gpu.check_vram_available(required_vram, mode.value)
        self.current_mode = mode
        self.gpu.log_vram_usage(f"Mode: {mode.value}")
        return True
    
    def unload_llm(self):
        """Explicitly unload LLM to free VRAM"""
        if self.current_mode == WorkloadMode.LLM:
            # Trigger garbage collection and clear VRAM
            import gc
            gc.collect()
            self.gpu.optimize_memory()
            self.current_mode = None
            logging.info("LLM unloaded, VRAM freed")
```

**Usage Pattern:**
```python
# Initialize
gpu = GPUAccelerator()
vram_mgr = VRAMManager(gpu)

# Simulation workflow
vram_mgr.request_mode(WorkloadMode.SIMULATION)
# ... run simulations ...

# Training workflow
vram_mgr.request_mode(WorkloadMode.TRAINING)
# ... train models ...

# LLM workflow (separate session)
vram_mgr.request_mode(WorkloadMode.LLM)
# ... use LLM ...
vram_mgr.unload_llm()  # Must explicitly unload before other work
```

**Design Constraints:**
- LLM and simulation/training are **mutually exclusive** - cannot run simultaneously
- User must explicitly choose between ML pipeline or LLM interface
- System will raise clear errors if VRAM limits are exceeded
- Automatic fallback to CPU for LLM if GPU memory insufficient (with performance warning)

### 6. LLM Adapter (Optional)

**Purpose:** Natural language interface and code generation

**⚠️ VRAM Warning:** LLM requires 20-22 GB VRAM and is **mutually exclusive** with simulation/training workloads.

**Architecture:**
```python
class LLMAdapter:
    def __init__(self, model_path: str, use_4bit: bool = True):
        self.model = self._load_model(model_path, use_4bit)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.lora_adapters = None
    
    def query(self, prompt: str, max_tokens: int = 512) -> str:
        """Process natural language query"""
        pass
    
    def generate_simulation_config(self, description: str) -> Dict:
        """Parse NL description into simulation parameters"""
        pass
    
    def explain_results(self, results: Dict) -> str:
        """Generate plain language explanation of simulation results"""
        pass
    
    def generate_code(self, task_description: str) -> str:
        """Generate Python/Qiskit code for quantum experiments"""
        pass
    
    def fine_tune(self, dataset_path: str, output_path: str):
        """QLoRA fine-tuning on domain data"""
        pass
```

**Implementation Details:**
- **Model Loading:**
  - Use `bitsandbytes` for 4-bit quantization
  - Load with `device_map="auto"` for automatic GPU placement
  - Expected VRAM: ~20-22 GB for Qwen 30B in 4-bit
  - Fallback to CPU inference if GPU memory exceeded (with performance warning)
- **Fine-Tuning:**
  - QLoRA with rank=4 or 8 (rank=4 for smaller VRAM footprint)
  - Alpha=16 for rank=8, alpha=8 for rank=4
  - Target modules: q_proj, v_proj
  - Training data: simulation configs, results, explanations, code examples
  - Training time: ~3-4 hours on RTX 3090
  - Use gradient accumulation if VRAM constrained
- **Inference:**
  - Response time: <5 seconds per query
  - Batch size: 1 (interactive use)
  - Temperature: 0.7 for balanced creativity/accuracy
  - Implement prompt validation and input sanitization
  - Cache repeated queries for efficiency
- **Code Generation Safety:**
  - Sandboxed execution environment for generated code
  - Syntax validation before execution
  - Whitelist allowed imports and functions

## Data Models

### Configuration Schema

```python
@dataclass
class SimulationConfig:
    """Configuration for a single simulation run"""
    simulator_type: str  # 'quantum_circuit', 'schrodinger', 'harmonic_oscillator'
    parameters: Dict[str, Any]  # Simulator-specific parameters
    output_path: Optional[str] = None
    validate: bool = True

@dataclass
class PipelineConfig:
    """Configuration for training pipeline"""
    n_samples: int = 10000
    output_dir: str = './data'
    train_split: float = 0.8
    model_types: List[str] = field(default_factory=lambda: ['mlp', 'xgboost'])
    gpu_enabled: bool = True
    mixed_precision: bool = False
    random_seed: int = 42
    
@dataclass
class ModelConfig:
    """Configuration for ML model"""
    model_type: str  # 'mlp', 'xgboost', 'lstm'
    hyperparameters: Dict[str, Any]
    save_path: str
```

### Dataset Schema

**HDF5 Structure:**
```
dataset.h5
├── metadata/
│   ├── simulator_type: str
│   ├── n_samples: int
│   ├── parameter_names: List[str]
│   ├── observable_names: List[str]
│   ├── timestamp: str
│   ├── version: str
│   ├── dataset_hash: str (SHA256 for reproducibility)
│   ├── random_seed: int
│   ├── grid_resolution: Dict (solver-specific)
│   └── sampling_strategy: str (LHS, grid, adaptive)
├── parameters/
│   ├── V0: float[n_samples]
│   ├── barrier_width: float[n_samples]
│   ├── k0: float[n_samples]
│   └── ... (other parameters)
├── observables/
│   ├── transmission: float[n_samples]
│   ├── reflection: float[n_samples]
│   ├── entropy: float[n_samples]
│   └── ... (other observables)
└── flags/
    ├── is_edge_case: bool[n_samples]
    ├── validation_passed: bool[n_samples]
    └── uncertainty_score: float[n_samples] (for adaptive sampling)
```

**In-Memory Representation:**
```python
@dataclass
class Dataset:
    X: np.ndarray  # Shape: (n_samples, n_parameters)
    y: np.ndarray  # Shape: (n_samples, n_observables)
    parameter_names: List[str]
    observable_names: List[str]
    metadata: Dict[str, Any]
    
    def split(self, train_ratio: float) -> Tuple['Dataset', 'Dataset']:
        """Split into train and validation sets"""
        pass
    
    def normalize(self) -> Tuple['Dataset', StandardScaler]:
        """Normalize features"""
        pass
```

### Model Output Schema

```python
@dataclass
class PredictionResult:
    """Result from ML model inference"""
    predictions: np.ndarray  # Predicted observables
    confidence: Optional[np.ndarray] = None  # Uncertainty estimates
    inference_time_ms: float = 0.0
    
@dataclass
class ValidationMetrics:
    """Metrics from model validation"""
    mse: float
    mae: float
    r2_score: float
    max_error: float
    predictions_vs_actual: np.ndarray  # For plotting
    residuals: np.ndarray
```

## Error Handling

### Simulation Errors

**Physics Validation Failures:**
- Check conservation laws after each simulation
- If validation fails, log warning and optionally retry with adjusted parameters
- Store failed simulations separately for debugging

**Numerical Instabilities:**
- Catch overflow/underflow in Schrödinger solver
- Reduce time step or increase grid resolution automatically
- Implement adaptive time stepping

**GPU Memory Errors:**
- Catch CUDA out-of-memory exceptions
- Fall back to CPU computation with warning
- Reduce batch size automatically

### Training Errors

**Data Quality Issues:**
- Check for NaN or Inf values in dataset
- Remove or impute invalid samples
- Log data quality report

**Training Failures:**
- Implement gradient clipping to prevent exploding gradients
- Use learning rate scheduling
- Save checkpoints every N epochs for recovery

**Model Convergence Issues:**
- Monitor validation loss for divergence
- Implement early stopping
- Suggest hyperparameter adjustments

### LLM Errors

**VRAM Overflow:**
- Catch OOM errors during model loading
- Suggest using smaller model or more aggressive quantization
- Provide fallback to CPU inference (slow but functional)

**Generation Failures:**
- Implement timeout for long-running generations
- Validate generated code/configs before execution
- Provide error messages with suggestions

## Strategic Enhancements

### 1. Adaptive Sampling Strategy

**Problem:** Uniform parameter sweeps waste computation on well-understood regions while under-sampling critical edge cases.

**Solution:**
- **Phase 1:** Generate initial dataset with Latin Hypercube Sampling (1000-2000 samples)
- **Phase 2:** Train preliminary ML model
- **Phase 3:** Use model uncertainty (MC dropout variance) to identify high-uncertainty regions
- **Phase 4:** Generate additional samples in high-uncertainty regions (adaptive refinement)
- **Phase 5:** Retrain model and repeat until convergence

**Implementation:**
```python
def adaptive_sampling(simulator, model, n_iterations=5, samples_per_iteration=500):
    # Initial LHS sampling
    X_init = latin_hypercube_sample(simulator.get_parameter_space(), n=2000)
    y_init = parallel_simulate(simulator, X_init)
    
    X_train, y_train = X_init, y_init
    
    for iteration in range(n_iterations):
        # Train model
        model.train(X_train, y_train)
        
        # Generate candidate points
        X_candidates = latin_hypercube_sample(simulator.get_parameter_space(), n=10000)
        
        # Compute uncertainty (MC dropout)
        uncertainty = model.predict_uncertainty(X_candidates)
        
        # Select high-uncertainty points
        top_indices = np.argsort(uncertainty)[-samples_per_iteration:]
        X_new = X_candidates[top_indices]
        
        # Simulate new points
        y_new = parallel_simulate(simulator, X_new)
        
        # Augment dataset
        X_train = np.vstack([X_train, X_new])
        y_train = np.vstack([y_train, y_new])
    
    return X_train, y_train
```

**Expected Impact:** 50-70% reduction in total simulations for equivalent model accuracy.

### 2. Physics-Informed Machine Learning

**Problem:** Standard ML models may violate fundamental physics constraints (e.g., probability conservation).

**Solution:** Integrate physics constraints directly into model architecture and loss function.

**Constraint Examples:**
- Schrödinger solver: R + T = 1 (probability conservation)
- Harmonic oscillator: Energy level spacing = ℏω (quantization)
- Quantum circuits: Trace(ρ) = 1 (density matrix normalization)

**Implementation:**
```python
class PhysicsInformedLoss(nn.Module):
    def __init__(self, base_loss=nn.MSELoss(), constraint_weight=0.1):
        super().__init__()
        self.base_loss = base_loss
        self.constraint_weight = constraint_weight
    
    def forward(self, predictions, targets, simulator_type):
        # Standard MSE loss
        mse = self.base_loss(predictions, targets)
        
        # Physics constraint penalty
        if simulator_type == 'schrodinger':
            # R + T should equal 1
            R_pred, T_pred = predictions[:, 0], predictions[:, 1]
            conservation_penalty = torch.mean((R_pred + T_pred - 1.0)**2)
        elif simulator_type == 'harmonic_oscillator':
            # Energy spacing should be uniform
            E_pred = predictions  # Energy levels
            spacing = E_pred[:, 1:] - E_pred[:, :-1]
            spacing_penalty = torch.var(spacing)
        else:
            conservation_penalty = 0.0
        
        total_loss = mse + self.constraint_weight * conservation_penalty
        return total_loss
```

**Expected Impact:** Improved generalization, especially in extrapolation regimes; reduced unphysical predictions.

### 3. Uncertainty Quantification

**Problem:** Point predictions don't convey model confidence, making it hard to identify when simulations are needed.

**Solution:** Implement multiple UQ methods for robust uncertainty estimates.

**Methods:**
1. **Monte Carlo Dropout:** Enable dropout at inference time, run N forward passes, compute variance
2. **Ensemble Models:** Train 3-5 models with different initializations, compute prediction variance
3. **Bayesian Neural Networks:** Use variational inference for posterior over weights (more expensive)

**Implementation:**
```python
class UncertaintyQuantifier:
    def __init__(self, model, method='mc_dropout', n_samples=50):
        self.model = model
        self.method = method
        self.n_samples = n_samples
    
    def predict_with_uncertainty(self, X):
        if self.method == 'mc_dropout':
            # Enable dropout at inference
            self.model.train()  # Keep dropout active
            predictions = []
            for _ in range(self.n_samples):
                with torch.no_grad():
                    pred = self.model(X)
                predictions.append(pred.cpu().numpy())
            predictions = np.array(predictions)
            
            mean = predictions.mean(axis=0)
            std = predictions.std(axis=0)
            return mean, std
        
        elif self.method == 'ensemble':
            # Use multiple trained models
            predictions = [model(X).cpu().numpy() for model in self.model.ensemble]
            predictions = np.array(predictions)
            
            mean = predictions.mean(axis=0)
            std = predictions.std(axis=0)
            return mean, std
```

**Expected Impact:** Enables adaptive sampling; provides confidence intervals for predictions; identifies when to fall back to full simulation.

### 4. Model Interpretability

**Problem:** Black-box ML models don't provide physical insight into parameter-observable relationships.

**Solution:** Use SHAP (SHapley Additive exPlanations) to identify which parameters most influence predictions.

**Implementation:**
```python
import shap

def explain_predictions(model, X_train, X_test):
    # Create SHAP explainer
    explainer = shap.KernelExplainer(model.predict, X_train[:100])
    
    # Compute SHAP values for test samples
    shap_values = explainer.shap_values(X_test)
    
    # Visualize feature importance
    shap.summary_plot(shap_values, X_test, feature_names=parameter_names)
    
    # For a single prediction
    shap.force_plot(explainer.expected_value, shap_values[0], X_test[0])
    
    return shap_values
```

**Expected Impact:** Physical insight into parameter sensitivities; validation of model behavior; identification of dominant physics mechanisms.

### 5. Benchmark Dataset: QuantumML-1K

**Purpose:** Standardized benchmark for comparing ML models and validating new approaches.

**Composition:**
- 1000 carefully selected samples covering:
  - Uniform coverage of parameter space (500 samples)
  - Edge cases: high barriers, resonant tunneling, critical oscillator states (200 samples)
  - Challenging regimes: high entanglement, strong decoherence (200 samples)
  - Validation samples with analytical solutions (100 samples)

**Metadata:**
- All simulation parameters
- Ground truth observables
- Physics validation flags
- Difficulty scores (based on model uncertainty)

**Distribution:**
- HDF5 format with full provenance
- Hosted on GitHub/Zenodo with DOI
- Leaderboard for model comparison

**Expected Impact:** Community adoption; reproducible research; standardized evaluation metrics.

### 6. Containerization and Deployment

**Docker Configuration:**
```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3.10 python3-pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install qiskit pennylane numpy scipy scikit-learn xgboost

# Copy application
COPY . /app
WORKDIR /app

# Install Ravan
RUN pip install -e .

# Expose API port
EXPOSE 8000

# Run API server
CMD ["python", "scripts/api_server.py"]
```

**Kubernetes Deployment:**
- Horizontal pod autoscaling based on GPU utilization
- Persistent volumes for model weights and datasets
- Load balancer for inference API

**Expected Impact:** Easy deployment on cloud or on-premises clusters; reproducible environments; scalable inference.

## Implementation Priorities

### Phase 1: Core Infrastructure (Weeks 1-2)
1. Set up WSL environment with CUDA support
2. Implement base simulation modules (quantum circuit, Schrödinger, harmonic oscillator)
3. Create dataset generation pipeline with HDF5 storage
4. Implement GPU accelerator utilities
5. Validate simulations against analytical solutions

**Success Criteria:** Generate 500-sample benchmark dataset with all three simulators; validate physics constraints.

### Phase 2: ML Pipeline (Weeks 3-4)
1. Implement MLP and XGBoost models with GPU acceleration
2. Create training pipeline with train/val split
3. Implement basic validation metrics (MSE, MAE, R²)
4. Add physics constraint validation
5. Generate prediction vs. actual plots

**Success Criteria:** Train models on 500-sample dataset; achieve <5% prediction error on validation set.

### Phase 3: Advanced Features (Weeks 5-6)
1. Implement uncertainty quantification (MC dropout)
2. Add adaptive sampling loop
3. Integrate physics-informed loss functions
4. Implement SHAP-based interpretability
5. Create QuantumML-1K benchmark dataset

**Success Criteria:** Demonstrate 50% reduction in samples needed via adaptive sampling; provide uncertainty estimates for all predictions.

### Phase 4: LLM Integration (Weeks 7-8) - Optional
1. Set up Qwen 30B with 4-bit quantization
2. Implement natural language query interface
3. Create fine-tuning pipeline with QLoRA
4. Add code generation with sandboxed execution
5. Implement query caching

**Success Criteria:** LLM responds to queries in <5 seconds; generates valid simulation configs from natural language.

### Phase 5: Production Readiness (Weeks 9-10)
1. Comprehensive testing (unit, integration, physics validation)
2. Performance optimization and profiling
3. Documentation and tutorials
4. Docker containerization
5. API server for remote inference

**Success Criteria:** Full test coverage; <1ms inference latency; production-ready deployment.

## Testing Strategy

### Unit Tests

**Simulation Modules:**
- Test each simulator with known analytical solutions
- Verify conservation laws on simple cases
- Test parameter validation and error handling

**ML Models:**
- Test training on synthetic data with known patterns
- Verify save/load functionality
- Test prediction shapes and ranges

**Pipeline Components:**
- Test dataset generation with small sample counts
- Test train/validation splitting
- Test metric computation

### Integration Tests

**End-to-End Pipeline:**
- Run full pipeline with small dataset (100 samples)
- Verify all outputs are generated
- Check file formats and data integrity

**GPU Acceleration:**
- Test GPU detection and initialization
- Verify CUDA operations produce correct results
- Test fallback to CPU when GPU unavailable

### Physics Validation Tests

**Quantum Mechanics Constraints:**
- Verify probability normalization in all simulations
- Check energy conservation in time evolution
- Validate uncertainty principle relations
- Test entanglement measures against theoretical bounds

**ML Prediction Validation:**
- Ensure predictions satisfy physics constraints
- Test on edge cases (extreme parameters)
- Compare against analytical solutions where available

### Performance Tests

**Benchmarks:**
- Measure simulation time for 10k samples
- Measure training time for each model type
- Measure inference latency (should be <1ms)
- Monitor GPU utilization and VRAM usage

**Scalability Tests:**
- Test with 100k samples
- Test with larger neural networks
- Test parallel simulation execution

## Deployment Considerations

### WSL Setup

**Prerequisites:**
- Windows 11 with WSL2 enabled
- NVIDIA GPU driver installed on Windows (version 525.60 or later)
- WSL Ubuntu 22.04 LTS distribution

**Installation Steps:**
1. Install CUDA Toolkit in WSL: `sudo apt install nvidia-cuda-toolkit`
2. Verify GPU access: `nvidia-smi` should show RTX 3090
3. Install conda: `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`
4. Create environment: `conda create -n ravan python=3.10`
5. Install dependencies: `pip install -r requirements.txt`

**GPU Passthrough Verification:**
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True

# Check GPU name
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Should print: NVIDIA GeForce RTX 3090
```

### Directory Structure

```
ravan-quantum-ml/
├── src/
│   ├── simulators/
│   │   ├── __init__.py
│   │   ├── base.py              # SimulationModule ABC
│   │   ├── quantum_circuit.py   # QuantumCircuitSimulator
│   │   ├── schrodinger.py       # SchrödingerSolver
│   │   └── harmonic.py          # HarmonicOscillatorModule
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py              # MLModel ABC
│   │   ├── mlp.py               # MLPRegressor
│   │   ├── xgboost_model.py     # XGBoostRegressor
│   │   └── lstm.py              # LSTMPredictor
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── training.py          # TrainingPipeline
│   │   ├── dataset.py           # Dataset class
│   │   └── metrics.py           # Validation metrics
│   ├── gpu/
│   │   ├── __init__.py
│   │   └── accelerator.py       # GPUAccelerator
│   ├── llm/
│   │   ├── __init__.py
│   │   └── adapter.py           # LLMAdapter (optional)
│   └── utils/
│       ├── __init__.py
│       ├── config.py            # Configuration classes
│       ├── logging.py           # Logging setup
│       └── visualization.py     # Plotting utilities
├── configs/
│   ├── simulation_defaults.yaml
│   ├── pipeline_config.yaml
│   └── model_configs.yaml
├── data/
│   ├── raw/                     # Generated simulation data
│   ├── processed/               # Normalized datasets
│   └── models/                  # Trained model weights
├── notebooks/
│   ├── 01_simulation_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_results_analysis.ipynb
├── tests/
│   ├── test_simulators.py
│   ├── test_models.py
│   ├── test_pipeline.py
│   └── test_physics_validation.py
├── scripts/
│   ├── run_pipeline.py          # Main entry point
│   ├── generate_data.py         # Data generation only
│   ├── train_models.py          # Training only
│   └── inference.py             # Inference on new parameters
├── requirements.txt
├── setup.py
└── README.md
```

### Configuration Management

**YAML Configuration Example:**
```yaml
# pipeline_config.yaml
pipeline:
  n_samples: 10000
  output_dir: ./data/raw
  train_split: 0.8
  random_seed: 42
  
gpu:
  enabled: true
  mixed_precision: false
  device_id: 0
  
simulators:
  - type: quantum_circuit
    parameter_ranges:
      n_qubits: [2, 5]
      shots: [4096, 8192]
  
  - type: schrodinger
    parameter_ranges:
      V0: [1.0, 10.0]
      barrier_width: [0.5, 2.0]
      k0: [5.0, 15.0]
      sigma: [0.5, 1.5]
  
  - type: harmonic_oscillator
    parameter_ranges:
      oscillator_length: [0.5, 2.0]
      basis_size: [20, 50]
      initial_n: [0, 10]

models:
  - type: mlp
    hyperparameters:
      hidden_layers: [256, 128, 64]
      dropout: 0.2
      learning_rate: 0.001
      batch_size: 128
      epochs: 100
  
  - type: xgboost
    hyperparameters:
      n_estimators: 500
      max_depth: 7
      learning_rate: 0.05
      tree_method: gpu_hist
```

### Logging and Monitoring

**Logging Strategy:**
- Use Python `logging` module with structured logging
- Log levels: DEBUG for development, INFO for production
- Separate log files for simulations, training, and inference
- Include timestamps, GPU memory usage, and performance metrics

**Monitoring Metrics:**
- Simulation throughput (samples/second)
- GPU utilization percentage
- VRAM usage (current/peak)
- Training loss curves
- Validation metrics over time
- Inference latency distribution

## Performance Optimization

### Simulation Optimization

**Parallelization:**
- Use `multiprocessing.Pool` for CPU-bound parameter sweeps
- Batch quantum circuit simulations on GPU to maximize occupancy
- Use CuPy for GPU-accelerated NumPy operations in Schrödinger solver with in-place operations

**Memory Optimization:**
- Stream large datasets from disk using HDF5 chunking
- Use memory-mapped arrays for very large parameter sweeps
- Clear GPU cache between large operations
- Implement memory reuse patterns in Schrödinger solver to avoid allocations
- Use gradient checkpointing for very deep networks to save VRAM

### Training Optimization

**Data Loading:**
- Use PyTorch DataLoader with `num_workers=4` and `pin_memory=True` for parallel data loading
- Enable prefetching to overlap data loading and computation
- Implement persistent workers to avoid process spawning overhead

**Model Training:**
- Enable mixed precision training (FP16) with `torch.cuda.amp` to reduce VRAM by ~30%
- Use gradient accumulation for effective larger batch sizes when VRAM is constrained
- Implement learning rate warmup and cosine annealing schedules
- Apply gradient clipping to prevent exploding gradients

**Hyperparameter Tuning:**
- Use Optuna for automated hyperparameter search with pruning
- Parallelize trials across multiple runs
- Use early stopping to avoid wasting compute on poor configurations
- Log hyperparameter experiments to MLflow or Weights & Biases

### Inference Optimization

**Batch Prediction:**
- Process multiple samples simultaneously on GPU
- Use TorchScript or ONNX for optimized inference
- Implement model quantization (INT8) for faster CPU inference if needed

**Caching:**
- Cache frequently requested predictions
- Use LRU cache for parameter → prediction mapping
- Store precomputed results for common parameter combinations

## Security and Privacy

**Local Execution:**
- All computations run locally on WSL
- No data sent to external services
- Model weights stored locally

**Data Protection:**
- Sensitive simulation parameters can be encrypted at rest
- Access control via file system permissions
- Audit logging for data access

**LLM Security:**
- Local model execution prevents data leakage
- Fine-tuned adapters stored separately from base model
- Input sanitization for code generation to prevent injection

## Future Extensibility

### Adding New Simulators

1. Implement `SimulationModule` interface
2. Define parameter space and observables
3. Implement physics validation
4. Register with pipeline via configuration
5. Add configuration schema
6. Provide template script: `templates/new_simulator_template.py`

### Adding New ML Models

1. Implement `MLModel` interface
2. Define hyperparameters
3. Implement GPU acceleration if applicable
4. Add to model registry via plugin system
5. Update configuration options
6. Provide template script: `templates/new_model_template.py`

### Scaling to Multiple GPUs

- Use PyTorch DistributedDataParallel for multi-GPU training
- Distribute parameter sweeps across GPUs using Ray or Dask
- Implement model parallelism for very large networks
- Support gradient accumulation across devices

### Cloud Deployment

- Containerize with Docker (NVIDIA CUDA base image)
- Deploy on cloud GPU instances (AWS p3, GCP A100, Azure NC-series)
- Implement REST API for remote inference using FastAPI
- Add authentication (JWT tokens) and rate limiting
- Support horizontal scaling with load balancing

### Advanced ML Techniques

**Physics-Informed Neural Networks (PINNs):**
- Integrate physics constraints directly into loss function
- Example: enforce Schrödinger equation residuals in training
- Improves generalization and reduces data requirements

**Adaptive Sampling:**
- Implement active learning loop: train model → identify high-uncertainty regions → generate new samples → retrain
- Reduces total simulation cost by 50-70% for equivalent accuracy

**Model Interpretability:**
- SHAP values for feature importance analysis
- Attention mechanisms for understanding model decisions
- Physics-based sanity checks on predictions

**Benchmark Dataset:**
- Create "QuantumML-1K" benchmark with 1000 diverse, validated samples
- Include edge cases and challenging parameter regimes
- Publish for community use and model comparison
