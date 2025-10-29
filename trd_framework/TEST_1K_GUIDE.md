# Test with 1,000 Examples - Quick Guide

## Overview

Generate and train on 1,000 examples to test the complete TRD pipeline.

**Total Time**: ~4 hours
- Dataset Generation: ~3 hours
- Model Training: ~30-60 minutes

## Easiest Way

### Step 1: Generate Dataset (3 hours)

**Double-click**: `trd_framework\TEST_1K_DATASET.bat`

This will:
1. Start Ollama with qwen3:30b-a3b
2. Generate 1,000 physics examples
3. Save to `large_dataset/physics_large_dataset.jsonl`

### Step 2: Train Model (30-60 minutes)

**Double-click**: `trd_framework\TRAIN_ON_1K.bat`

This will:
1. Train Qwen2-1.5B on 1,000 examples
2. Save to `models/qwen2-1.5b-1k/`

## Manual Commands

### Generate Dataset

```cmd
# Terminal 1: Start teacher
ollama run qwen3:30b-a3b

# Terminal 2: Generate 1k examples
wsl bash -c "cd '/mnt/c/Users/windows/Documents/rishi - professor' && source venv_test/bin/activate && python trd_framework/generate_large_dataset.py --mode physics --target-size 1000 --teacher-model qwen3:30b-a3b 2>&1"
```

### Train Model

```cmd
wsl bash -c "cd '/mnt/c/Users/windows/Documents/rishi - professor' && source venv_test/bin/activate && python trd_framework/run_phase2_training.py --dataset trd_framework/large_dataset/physics_large_dataset.jsonl --output-dir trd_framework/models/qwen2-1.5b-1k --epochs 3 --batch-size 2 --gradient-accumulation 8 2>&1"
```

## What You'll See

### During Generation

```
======================================================================
LARGE-SCALE TRD DATASET GENERATION
======================================================================
Target size: 1,000 examples
Mode: physics
Teacher: qwen3:30b-a3b
======================================================================

Step 1: Generating base queries...
✓ Base queries: 69

Step 2: Generating 15 variations...
✓ Generated 1,035 total queries
✓ Total queries to process: 1,000

Step 3: Querying teacher model...
Estimated time: 3.3 hours

Generating dataset:   0%|                      | 0/1000 [00:00<?, ?it/s]
Generating dataset:  10%|██                    | 100/1000 [00:20:00<03:00:00]
✓ Checkpoint saved: 100/1000 examples

Generating dataset:  20%|████                  | 200/1000 [00:40:00<02:40:00]
✓ Checkpoint saved: 200/1000 examples

...

Generating dataset: 100%|██████████████████████| 1000/1000 [03:20:00<00:00]

======================================================================
GENERATION COMPLETE!
======================================================================
Total examples: 1,000
Output file: trd_framework/large_dataset/physics_large_dataset.jsonl
File size: 24.50 MB
======================================================================
```

### During Training

```
======================================================================
TRD PHASE 2: STUDENT MODEL TRAINING
======================================================================

Loading dataset from trd_framework/large_dataset/physics_large_dataset.jsonl...
✓ Loaded 1000 examples

======================================================================
STARTING TRAINING
======================================================================
Dataset size: 1000
Epochs: 3
Total steps: 187

Training: [████████████████████] 100%
Epoch 1/3: Loss: 2.145
Epoch 2/3: Loss: 1.687
Epoch 3/3: Loss: 1.423

======================================================================
TRAINING COMPLETE!
======================================================================
✓ Model saved to: trd_framework/models/qwen2-1.5b-1k
✓ Training time: 0.67 hours
```

## Monitoring Progress

### Check Generation Progress

```cmd
# Count generated examples
wsl wc -l trd_framework/large_dataset/physics_large_dataset.jsonl

# Check checkpoint
wsl cat trd_framework/large_dataset/physics_checkpoint.json
```

### Check Training Progress

```cmd
# Monitor GPU
wsl nvidia-smi

# Check VRAM usage
wsl nvidia-smi --query-gpu=memory.used --format=csv
```

## Pause and Resume

### To Pause Generation
Press `Ctrl+C` in the terminal

### To Resume
Just run the same command again - it automatically resumes!

## Expected Results

### Dataset Quality

With 1,000 examples, you'll get:
- ✅ Good reasoning ability
- ✅ Basic self-correction
- ✅ Decent physics knowledge
- ✅ Fast inference

### Model Performance

| Metric | Expected Value |
|--------|----------------|
| Training Loss | 1.4-1.8 |
| Reasoning Score | 3.0-3.5 / 5.0 |
| Self-Correction | 40-50% |
| Model Size | 3.2GB |
| Inference Speed | <100ms |

## After Testing

### If Results Look Good

Scale up to 10k or 100k:

```cmd
# 10k examples (overnight)
python trd_framework/generate_large_dataset.py --target-size 10000

# 100k examples (weekend)
python trd_framework/generate_large_dataset.py --target-size 100000
```

### Test the Model

```cmd
wsl bash -c "cd '/mnt/c/Users/windows/Documents/rishi - professor' && source venv_test/bin/activate && python trd_framework/test_trd_model.py --model-path trd_framework/models/qwen2-1.5b-1k --mode interactive 2>&1"
```

Then ask:
- "Explain quantum tunneling through a barrier"
- "What is the Von Neumann entropy of a Bell state?"
- "Derive the energy eigenvalues of a harmonic oscillator"

## Troubleshooting

### Generation Stops

**Solution**: Just restart - it resumes automatically
```cmd
python trd_framework/generate_large_dataset.py --target-size 1000
```

### Training OOM

**Solution**: Reduce batch size
```cmd
python trd_framework/run_phase2_training.py ... --batch-size 1 --gradient-accumulation 16
```

### Slow Generation

**Normal!** Each example takes ~12 seconds because:
1. Teacher reads query
2. Generates initial thought
3. Creates reasoning chain
4. Identifies flaws
5. Self-corrects
6. Provides final answer

## Timeline

```
Hour 0:00 - Start generation
Hour 0:20 - 100 examples (10%)
Hour 0:40 - 200 examples (20%)
Hour 1:00 - 300 examples (30%)
Hour 1:20 - 400 examples (40%)
Hour 1:40 - 500 examples (50%)
Hour 2:00 - 600 examples (60%)
Hour 2:20 - 700 examples (70%)
Hour 2:40 - 800 examples (80%)
Hour 3:00 - 900 examples (90%)
Hour 3:20 - 1000 examples (100%) ✓

Hour 3:20 - Start training
Hour 4:00 - Training complete ✓
```

## Summary

**To test with 1,000 examples:**

1. ✅ Double-click `TEST_1K_DATASET.bat`
2. ⏳ Wait ~3 hours (can pause/resume)
3. ✅ Double-click `TRAIN_ON_1K.bat`
4. ⏳ Wait ~30-60 minutes
5. 🎉 Test your model!

**Result**: A working TRD model trained on 1,000 structured reasoning examples!
