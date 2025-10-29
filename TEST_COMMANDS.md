# Test Commands Reference
Quick reference for running all tests in the Ravan Quantum-ML System

## Quick Start

### Run All Tests (Linux/WSL)
```bash
cd ~/ravan-quantum-ml
source venv/bin/activate
bash run_tests.sh
```

### Run All Tests (Windows PowerShell)
```powershell
wsl bash -c "cd ~/ravan-quantum-ml && source venv/bin/activate && bash run_tests.sh"
```

---

## Unit Tests (pytest-based)

### Task 3.4: Quantum Circuit Simulator Unit Tests
```bash
# Linux/WSL
cd ~/ravan-quantum-ml
source venv/bin/activate
python -m pytest tests/test_quantum_circuit_unit.py -v

# Windows PowerShell
wsl bash -c "cd ~/ravan-quantum-ml && source venv/bin/activate && python -m pytest tests/test_quantum_circuit_unit.py -v"
```

**Status:** ✅ Created (18 tests, 13 passing)

---

### Task 4.5: Schrödinger Solver Unit Tests
```bash
# Linux/WSL
cd ~/ravan-quantum-ml
source venv/bin/activate
python -m pytest tests/test_schrodinger_unit.py -v

# Windows PowerShell
wsl bash -c "cd ~/ravan-quantum-ml && source venv/bin/activate && python -m pytest tests/test_schrodinger_unit.py -v"
```

**Status:** ✅ Created (22 tests, 21 passing)

---

### Task 5.4: Harmonic Oscillator Unit Tests
```bash
# Linux/WSL
cd ~/ravan-quantum-ml
source venv/bin/activate
python -m pytest tests/test_harmonic_oscillator_unit.py -v

# Windows PowerShell
wsl bash -c "cd ~/ravan-quantum-ml && source venv/bin/activate && python -m pytest tests/test_harmonic_oscillator_unit.py -v"
```

**Status:** ✅ Created (22 tests, 22 passing - 100%)

---

### Task 7.4: MLP Model Unit Tests
```bash
# Linux/WSL
cd ~/ravan-quantum-ml
source venv/bin/activate
python -m pytest tests/test_mlp_unit.py -v

# Windows PowerShell
wsl bash -c "cd ~/ravan-quantum-ml && source venv/bin/activate && python -m pytest tests/test_mlp_unit.py -v"
```

**Status:** ⏳ To be created

---

### Task 8.4: XGBoost Model Unit Tests
```bash
# Linux/WSL
cd ~/ravan-quantum-ml
source venv/bin/activate
python -m pytest tests/test_xgboost_unit.py -v

# Windows PowerShell
wsl bash -c "cd ~/ravan-quantum-ml && source venv/bin/activate && python -m pytest tests/test_xgboost_unit.py -v"
```

**Status:** ⏳ To be created

---

## Integration Tests (existing scripts)

### Quantum Circuit Integration Test
```bash
# Linux/WSL
cd ~/ravan-quantum-ml
source venv/bin/activate
python test_quantum_circuit.py

# Windows PowerShell
wsl bash -c "cd ~/ravan-quantum-ml && source venv/bin/activate && python test_quantum_circuit.py"
```

---

### Enhanced Circuit Test
```bash
# Linux/WSL
cd ~/ravan-quantum-ml
source venv/bin/activate
python test_enhanced_circuit.py

# Windows PowerShell
wsl bash -c "cd ~/ravan-quantum-ml && source venv/bin/activate && python test_enhanced_circuit.py"
```

---

### Schrödinger Solver Integration Test
```bash
# Linux/WSL
cd ~/ravan-quantum-ml
source venv/bin/activate
python test_schrodinger.py

# Windows PowerShell
wsl bash -c "cd ~/ravan-quantum-ml && source venv/bin/activate && python test_schrodinger.py"
```

---

### Harmonic Oscillator Integration Test
```bash
# Linux/WSL
cd ~/ravan-quantum-ml
source venv/bin/activate
python test_harmonic.py

# Windows PowerShell
wsl bash -c "cd ~/ravan-quantum-ml && source venv/bin/activate && python test_harmonic.py"
```

---

### Base Classes Test
```bash
# Linux/WSL
cd ~/ravan-quantum-ml
source venv/bin/activate
python test_base_classes.py

# Windows PowerShell
wsl bash -c "cd ~/ravan-quantum-ml && source venv/bin/activate && python test_base_classes.py"
```

---

### GPU/VRAM Test
```bash
# Linux/WSL
cd ~/ravan-quantum-ml
source venv/bin/activate
python test_gpu_vram.py

# Windows PowerShell
wsl bash -c "cd ~/ravan-quantum-ml && source venv/bin/activate && python test_gpu_vram.py"
```

---

### XGBoost GPU Test
```bash
# Linux/WSL
cd ~/ravan-quantum-ml
source venv/bin/activate
python test_xgboost_gpu.py

# Windows PowerShell
wsl bash -c "cd ~/ravan-quantum-ml && source venv/bin/activate && python test_xgboost_gpu.py"
```

---

## Advanced Feature Tests

### Adaptive Sampling Quick Test
```bash
# Linux/WSL
cd ~/ravan-quantum-ml
source venv/bin/activate
python test_adaptive_sampling_quick.py

# Windows PowerShell
wsl bash -c "cd ~/ravan-quantum-ml && source venv/bin/activate && python test_adaptive_sampling_quick.py"
```

---

### Physics-Informed Training Test
```bash
# Linux/WSL
cd ~/ravan-quantum-ml
source venv/bin/activate
python test_physics_informed_training.py

# Windows PowerShell
wsl bash -c "cd ~/ravan-quantum-ml && source venv/bin/activate && python test_physics_informed_training.py"
```

---

### Uncertainty Quantification Test
```bash
# Linux/WSL
cd ~/ravan-quantum-ml
source venv/bin/activate
python test_uncertainty_quantification.py

# Windows PowerShell
wsl bash -c "cd ~/ravan-quantum-ml && source venv/bin/activate && python test_uncertainty_quantification.py"
```

---

### Model Interpretability Test
```bash
# Linux/WSL
cd ~/ravan-quantum-ml
source venv/bin/activate
python test_model_interpretability.py

# Windows PowerShell
wsl bash -c "cd ~/ravan-quantum-ml && source venv/bin/activate && python test_model_interpretability.py"
```

---

## Validation Tests

### Task 11.4: Adaptive Sampling Validation
```bash
# Linux/WSL
cd ~/ravan-quantum-ml
source venv/bin/activate
python validate_adaptive_sampling.py

# Windows PowerShell
wsl bash -c "cd ~/ravan-quantum-ml && source venv/bin/activate && python validate_adaptive_sampling.py"
```

**Note:** This test takes several minutes to complete.

**Status:** ✅ Completed

---

## Data Generation

### Generate QuantumML-1K Dataset (1000 samples)
```bash
# Linux/WSL
cd ~/ravan-quantum-ml
source venv/bin/activate
python generate_quantumml_1k.py

# Windows PowerShell
wsl bash -c "cd ~/ravan-quantum-ml && source venv/bin/activate && python generate_quantumml_1k.py"
```

**Status:** ✅ Completed (1000 samples generated)

---

## Pytest Options

### Run with verbose output
```bash
python -m pytest <test_file> -v
```

### Run with short traceback
```bash
python -m pytest <test_file> -v --tb=short
```

### Run with line-only traceback
```bash
python -m pytest <test_file> -v --tb=line
```

### Run specific test
```bash
python -m pytest <test_file>::<TestClass>::<test_method> -v
```

### Run and stop on first failure
```bash
python -m pytest <test_file> -x
```

### Run with coverage report
```bash
python -m pytest <test_file> --cov=src --cov-report=html
```

---

## Test Status Summary

| Task | Test File | Status | Pass Rate |
|------|-----------|--------|-----------|
| 3.4 | `tests/test_quantum_circuit_unit.py` | ✅ Created | 72% (13/18) |
| 4.5 | `tests/test_schrodinger_unit.py` | ✅ Created | 95% (21/22) |
| 5.4 | `tests/test_harmonic_oscillator_unit.py` | ✅ Created | 100% (22/22) |
| 7.4 | `tests/test_mlp_unit.py` | ⏳ Pending | - |
| 8.4 | `tests/test_xgboost_unit.py` | ⏳ Pending | - |
| 11.4 | `validate_adaptive_sampling.py` | ✅ Complete | - |

---

## File Locations

### Unit Test Files (pytest)
- `~/ravan-quantum-ml/tests/test_quantum_circuit_unit.py`
- `~/ravan-quantum-ml/tests/test_schrodinger_unit.py`
- `~/ravan-quantum-ml/tests/test_harmonic_oscillator_unit.py`
- `~/ravan-quantum-ml/tests/test_mlp_unit.py` (to be created)
- `~/ravan-quantum-ml/tests/test_xgboost_unit.py` (to be created)

### Integration Test Files
- `~/ravan-quantum-ml/test_quantum_circuit.py`
- `~/ravan-quantum-ml/test_enhanced_circuit.py`
- `~/ravan-quantum-ml/test_schrodinger.py`
- `~/ravan-quantum-ml/test_harmonic.py`
- `~/ravan-quantum-ml/test_base_classes.py`
- `~/ravan-quantum-ml/test_gpu_vram.py`
- `~/ravan-quantum-ml/test_xgboost_gpu.py`

### Validation Scripts
- `~/ravan-quantum-ml/validate_adaptive_sampling.py`

### Test Runner
- `~/ravan-quantum-ml/run_tests.sh`

---

## Notes

1. **Always activate the virtual environment** before running tests
2. **Unit tests use pytest** - they are in the `tests/` directory
3. **Integration tests are standalone scripts** - they are in the root directory
4. **Some tests require GPU** - they will fall back to CPU if GPU is not available
5. **Long-running tests** (like validation) should be run separately
6. **Test files in Windows** are located at: `C:\Users\windows\Documents\rishi - professor\tests\`
7. **Test files in WSL** are located at: `~/ravan-quantum-ml/tests/`

---

## Troubleshooting

### Tests not found
```bash
# Make sure you're in the right directory
cd ~/ravan-quantum-ml
pwd

# Check if test files exist
ls -la tests/
```

### Import errors
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Check Python path
python -c "import sys; print(sys.path)"
```

### pytest not found
```bash
# Install pytest
pip install pytest pytest-cov
```

---

**Last Updated:** October 19, 2025
