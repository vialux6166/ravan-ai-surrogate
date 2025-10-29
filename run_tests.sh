#!/bin/bash
# Test Runner Script for Ravan Quantum-ML System
# This script contains all test commands for easy execution

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Ravan Quantum-ML System - Test Runner"
echo "=========================================="
echo ""

# Activate virtual environment
source ~/ravan-quantum-ml/venv/bin/activate

# Change to project directory
cd ~/ravan-quantum-ml

# =========================================================================
# UNIT TESTS (pytest-based)
# =========================================================================

echo -e "${BLUE}=========================================="
echo "UNIT TESTS"
echo -e "==========================================${NC}"
echo ""

# Test 3.4: Quantum Circuit Simulator Unit Tests
echo -e "${GREEN}[1/4] Running Quantum Circuit Simulator Unit Tests...${NC}"
python -m pytest tests/test_quantum_circuit_unit.py -v --tb=short
echo ""

# Test 4.5: Schrödinger Solver Unit Tests
echo -e "${GREEN}[2/4] Running Schrödinger Solver Unit Tests...${NC}"
python -m pytest tests/test_schrodinger_unit.py -v --tb=short
echo ""

# Test 5.4: Harmonic Oscillator Unit Tests
echo -e "${GREEN}[3/4] Running Harmonic Oscillator Unit Tests...${NC}"
python -m pytest tests/test_harmonic_oscillator_unit.py -v --tb=short
echo ""

# Test 7.4: MLP Model Unit Tests (when created)
echo -e "${GREEN}[4/4] Running MLP Model Unit Tests...${NC}"
if [ -f "tests/test_mlp_unit.py" ]; then
    python -m pytest tests/test_mlp_unit.py -v --tb=short
else
    echo -e "${YELLOW}  Not yet created${NC}"
fi
echo ""

# Test 8.4: XGBoost Model Unit Tests (when created)
echo -e "${GREEN}[5/5] Running XGBoost Model Unit Tests...${NC}"
if [ -f "tests/test_xgboost_unit.py" ]; then
    python -m pytest tests/test_xgboost_unit.py -v --tb=short
else
    echo -e "${YELLOW}  Not yet created${NC}"
fi
echo ""

# =========================================================================
# INTEGRATION TESTS (existing test scripts)
# =========================================================================

echo -e "${BLUE}=========================================="
echo "INTEGRATION TESTS"
echo -e "==========================================${NC}"
echo ""

# Quantum Circuit Integration Test
echo -e "${GREEN}Running Quantum Circuit Integration Test...${NC}"
python test_quantum_circuit.py
echo ""

# Enhanced Circuit Test
echo -e "${GREEN}Running Enhanced Circuit Test...${NC}"
python test_enhanced_circuit.py
echo ""

# Schrödinger Solver Integration Test
echo -e "${GREEN}Running Schrödinger Integration Test...${NC}"
python test_schrodinger.py
echo ""

# Harmonic Oscillator Integration Test
echo -e "${GREEN}Running Harmonic Oscillator Integration Test...${NC}"
python test_harmonic.py
echo ""

# Base Classes Test
echo -e "${GREEN}Running Base Classes Test...${NC}"
python test_base_classes.py
echo ""

# GPU/VRAM Test
echo -e "${GREEN}Running GPU/VRAM Test...${NC}"
python test_gpu_vram.py
echo ""

# XGBoost GPU Test
echo -e "${GREEN}Running XGBoost GPU Test...${NC}"
python test_xgboost_gpu.py
echo ""

# =========================================================================
# ADVANCED FEATURE TESTS
# =========================================================================

echo -e "${BLUE}=========================================="
echo "ADVANCED FEATURE TESTS"
echo -e "==========================================${NC}"
echo ""

# Adaptive Sampling Test
echo -e "${GREEN}Running Adaptive Sampling Quick Test...${NC}"
python test_adaptive_sampling_quick.py
echo ""

# Physics-Informed Training Test
echo -e "${GREEN}Running Physics-Informed Training Test...${NC}"
python test_physics_informed_training.py
echo ""

# Uncertainty Quantification Test
echo -e "${GREEN}Running Uncertainty Quantification Test...${NC}"
python test_uncertainty_quantification.py
echo ""

# Model Interpretability Test
echo -e "${GREEN}Running Model Interpretability Test...${NC}"
python test_model_interpretability.py
echo ""

# =========================================================================
# VALIDATION TESTS
# =========================================================================

echo -e "${BLUE}=========================================="
echo "VALIDATION TESTS"
echo -e "==========================================${NC}"
echo ""

# Adaptive Sampling Validation (Task 11.4)
echo -e "${GREEN}Running Adaptive Sampling Validation...${NC}"
echo -e "${YELLOW}  (This takes several minutes)${NC}"
# python validate_adaptive_sampling.py
echo -e "${YELLOW}  Skipped (run manually if needed)${NC}"
echo ""

# =========================================================================
# SUMMARY
# =========================================================================

echo -e "${BLUE}=========================================="
echo "TEST EXECUTION COMPLETE"
echo -e "==========================================${NC}"
echo ""
echo "All tests have been executed."
echo "Review the output above for any failures."
echo ""
