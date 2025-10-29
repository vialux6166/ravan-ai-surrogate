# Ravan Quick Start Guide

Get started with Ravan in 5 minutes!

## Installation

```bash
# Clone repository
git clone <repository-url>
cd ravan-quantum-ml

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt

# Optional: Install LLM (requires GPU with 20GB+ VRAM)
pip install -r requirements-llm.txt
```

## Basic Usage (No LLM Required)

### 1. Run a Simulation

```python
from ravan import Ravan

# Initialize
ravan = Ravan()

# Run quantum tunneling simulation
results = ravan.simulate({
    'simulator': 'schrodinger',
    'V0': 5.0,              # Barrier height (eV)
    'k0': 4.0,              # Initial momentum
    'barrier_width': 1.5    # Barrier width (nm)
})

print(f"Transmission: {results['transmission']:.4f}")
print(f"Reflection: {results['reflection']:.4f}")
```

### 2. Try Different Simulators

```python
# Quantum circuit
results = ravan.simulate({
    'simulator': 'quantum_circuit',
    'n_qubits': 2,
    'shots': 1000,
    'gate_sequence': 'bell'
})

# Harmonic oscillator
results = ravan.simulate({
    'simulator': 'harmonic',
    'oscillator_length': 1.0,
    'basis_size': 10,
    'initial_n': 2
})
```

## Advanced Usage (With LLM)

### 1. Natural Language Simulations

```python
from ravan import Ravan

# Initialize with LLM
with Ravan(use_llm=True) as ravan:
    # Describe simulation in plain English
    results = ravan.simulate(
        "quantum tunneling with high barrier",
        explain=True
    )
    
    print(results['explanation'])
```

### 2. Ask Questions

```python
with Ravan(use_llm=True) as ravan:
    answer = ravan.query("What is quantum entanglement?")
    print(answer)
```

### 3. Generate Code

```python
with Ravan(use_llm=True) as ravan:
    code = ravan.generate_code(
        "Create a Bell state circuit with measurement"
    )
    print(code)
```

## Command Line Interface

### Run Simulation

```bash
# Basic simulation
python ravan_cli.py simulate \
    --simulator schrodinger \
    --V0 5.0 \
    --k0 4.0

# With natural language (requires --use-llm)
python ravan_cli.py simulate \
    --nl-config "quantum tunneling with high barrier" \
    --explain \
    --use-llm
```

### Interactive Chat

```bash
python ravan_cli.py chat --use-llm
```

## Next Steps

1. **Explore Examples**: Check `examples/basic_usage.py`
2. **Read Documentation**: See `README.md` for full API
3. **Train Models**: Generate datasets and train ML models
4. **Customize**: Adjust LLM and simulation parameters

## Troubleshooting

### LLM Not Loading
- **Check GPU**: Requires 20GB+ VRAM (RTX 3090 or better)
- **Install Dependencies**: `pip install -r requirements-llm.txt`
- **Check CUDA**: Ensure CUDA 12.x is installed

### Slow Performance
- **Use GPU**: Enable CUDA for ML training
- **Reduce Samples**: Start with smaller datasets
- **Disable LLM**: Use `use_llm=False` for faster startup

### Import Errors
- **Activate venv**: `source venv/bin/activate`
- **Install deps**: `pip install -r requirements.txt`
- **Check Python**: Requires Python 3.8+

## Getting Help

- **Documentation**: `README.md`
- **Examples**: `examples/` directory
- **Issues**: GitHub Issues
- **Tests**: Run `pytest tests/` to verify installation

---

**Ready to explore quantum mechanics with AI? Let's go! 🚀**
