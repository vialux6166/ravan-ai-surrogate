#!/bin/bash
# Script to install LLM dependencies for Ravan Quantum-ML System
# Task 15.1: Install LLM dependencies

set -e  # Exit on error

echo "=========================================="
echo "Installing LLM Dependencies"
echo "=========================================="
echo ""

# Activate virtual environment
source ~/ravan-quantum-ml/venv/bin/activate

# Change to project directory
cd ~/ravan-quantum-ml

echo "Step 1: Installing transformers library..."
pip install transformers>=4.35.0

echo ""
echo "Step 2: Installing PEFT (Parameter-Efficient Fine-Tuning)..."
pip install peft>=0.6.0

echo ""
echo "Step 3: Installing bitsandbytes for 4-bit quantization..."
pip install bitsandbytes>=0.41.0

echo ""
echo "Step 4: Installing accelerate for distributed training..."
pip install accelerate>=0.24.0

echo ""
echo "Step 5: Installing additional dependencies..."
pip install sentencepiece>=0.1.99  # For tokenization
pip install protobuf>=3.20.0       # For model serialization

echo ""
echo "=========================================="
echo "Verifying Installation"
echo "=========================================="
echo ""

python -c "
import transformers
import peft
import bitsandbytes
import accelerate
print('✓ transformers version:', transformers.__version__)
print('✓ peft version:', peft.__version__)
print('✓ bitsandbytes version:', bitsandbytes.__version__)
print('✓ accelerate version:', accelerate.__version__)
print('')
print('All LLM dependencies installed successfully!')
"

echo ""
echo "=========================================="
echo "Installation Complete"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Download Qwen 30B model weights (requires ~60GB disk space)"
echo "2. Verify 4-bit quantization loading"
echo "3. Test VRAM requirements (~22GB for inference)"
echo ""
