# Retrain Instructions - Fixed EOS Token Issue

## Quick Start

```bash
cd '/mnt/c/Users/windows/Documents/rishi - professor'
source venv_test/bin/activate

# Retrain with fixed code (35 minutes)
python trd_framework/run_phase2_training.py \
    --dataset trd_framework/large_dataset/physics_large_dataset.jsonl \
    --output-dir trd_framework/models/qwen2-1.5b-1k-fixed \
    --batch-size 2 \
    --gradient-accumulation 8 \
    --epochs 3
```

## What Was Fixed

1. **Added EOS tokens to training data** - Model now learns when to stop
2. **Fixed tokenizer configuration** - Proper pad_token = eos_token setup

## Test After Training

```bash
# Test with examples
python trd_framework/test_trd_model.py \
    --model-path trd_framework/models/qwen2-1.5b-1k-fixed \
    --mode examples \
    --domain physics

# Test interactively
python trd_framework/test_trd_model.py \
    --model-path trd_framework/models/qwen2-1.5b-1k-fixed \
    --mode interactive
```

## Expected Behavior

The model should now:
- Generate structured reasoning (Initial Thought → Chain → Flaws → Correction → Final Answer)
- **STOP after completing the response** (no hallucinated conversations)
- Wait for your next query in interactive mode

## Training Time

- Same as before: ~35 minutes
- No need to regenerate dataset
- Just retrain with fixed code
