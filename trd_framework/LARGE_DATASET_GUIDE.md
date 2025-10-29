# Generate 100,000+ Training Examples

## Overview

Generate a large-scale TRD dataset (100k+ examples) using Qwen 30B teacher model for comprehensive training.

## Time Estimates

| Dataset Size | Estimated Time | Can Pause/Resume |
|--------------|----------------|------------------|
| 10,000 | ~3.3 hours | ✅ Yes |
| 50,000 | ~16.7 hours | ✅ Yes |
| 100,000 | ~33.3 hours | ✅ Yes |

**Note**: ~12 seconds per query (teacher generates structured reasoning)

## Quick Start

### Step 1: Start Teacher Model

```cmd
ollama run qwen3:30b-a3b
```

Keep this running in Terminal 1.

### Step 2: Generate Large Dataset

**For 100k Physics Examples:**
```cmd
wsl bash -c "cd '/mnt/c/Users/windows/Documents/rishi - professor' && source venv_test/bin/activate && python trd_framework/generate_large_dataset.py --mode physics --target-size 100000 --teacher-model qwen3:30b-a3b 2>&1"
```

**For 100k General Examples:**
```cmd
wsl bash -c "cd '/mnt/c/Users/windows/Documents/rishi - professor' && source venv_test/bin/activate && python trd_framework/generate_large_dataset.py --mode general --target-size 100000 --teacher-model qwen3:30b-a3b 2>&1"
```

## Features

### ✅ Checkpoint System
- Saves progress every 1,000 examples
- Can pause (Ctrl+C) and resume anytime
- No data loss on interruption

### ✅ Progress Tracking
- Real-time progress bar
- Examples per second
- Estimated time remaining

### ✅ Automatic Variation
- **Physics Mode**: Generates varied simulation parameters
- **General Mode**: Creates question variations
- Ensures diverse training data

## Recommended Approach

### Option 1: Overnight Generation (Recommended)

Generate while you sleep:

```cmd
# Start before bed (~8 hours)
# Generates ~24,000 examples overnight

python trd_framework/generate_large_dataset.py \
    --mode physics \
    --target-size 25000 \
    --teacher-model qwen3:30b-a3b
```

### Option 2: Weekend Generation

Run over weekend:

```cmd
# Friday evening to Sunday evening (~48 hours)
# Generates ~144,000 examples

python trd_framework/generate_large_dataset.py \
    --mode physics \
    --target-size 100000 \
    --teacher-model qwen3:30b-a3b
```

### Option 3: Incremental Generation

Generate in chunks:

```cmd
# Day 1: 10k examples (~3 hours)
python trd_framework/generate_large_dataset.py \
    --mode physics \
    --target-size 10000

# Day 2: Continue to 20k
python trd_framework/generate_large_dataset.py \
    --mode physics \
    --target-size 20000

# Automatically resumes from checkpoint!
```

## What You'll See

```
======================================================================
LARGE-SCALE TRD DATASET GENERATION
======================================================================
Target size: 100,000 examples
Mode: physics
Teacher: qwen3:30b-a3b
======================================================================

Step 1: Generating base queries...
✓ Base queries: 69

Step 2: Generating 1450 variations...
✓ Generated 100,050 total queries
✓ Total queries to process: 100,000

Step 3: Querying teacher model...
Estimated time: 33.3 hours
(~12 seconds per query)

======================================================================
LARGE-SCALE DATASET GENERATION
======================================================================
Mode: physics
Teacher Model: qwen3:30b-a3b
Total Queries: 100000
Starting from: 0
Remaining: 100000
Batch Size: 100
Output: trd_framework/large_dataset/physics_large_dataset.jsonl
======================================================================

Generating dataset: 1%|█▌                    | 1000/100000 [00:20:00<33:00:00]
✓ Checkpoint saved: 1000/100000 examples

Generating dataset: 2%|███                   | 2000/100000 [00:40:00<32:40:00]
✓ Checkpoint saved: 2000/100000 examples

...

Generating dataset: 100%|██████████████████| 100000/100000 [33:20:00<00:00]

======================================================================
GENERATION COMPLETE!
======================================================================
Total examples: 100,000
Output file: trd_framework/large_dataset/physics_large_dataset.jsonl
File size: 2,450.00 MB
======================================================================
```

## Pause and Resume

### To Pause
Press `Ctrl+C` at any time. Progress is saved automatically.

### To Resume
Just run the same command again:
```cmd
python trd_framework/generate_large_dataset.py \
    --mode physics \
    --target-size 100000 \
    --teacher-model qwen3:30b-a3b
```

It will automatically resume from the last checkpoint!

## Output Files

```
trd_framework/large_dataset/
├── base_queries.jsonl                    # Original 69 queries
├── physics_checkpoint.json               # Resume point
└── physics_large_dataset.jsonl           # 100k examples (~2.5GB)
```

## Training on Large Dataset

After generation completes:

```cmd
wsl bash -c "cd '/mnt/c/Users/windows/Documents/rishi - professor' && source venv_test/bin/activate && python trd_framework/run_phase2_training.py --dataset trd_framework/large_dataset/physics_large_dataset.jsonl --output-dir trd_framework/models/qwen2-1.5b-large --epochs 3 --batch-size 2 --gradient-accumulation 8 2>&1"
```

**Training Time**: ~10-15 hours for 100k examples

## Benefits of Large Dataset

| Dataset Size | Model Quality | Use Case |
|--------------|---------------|----------|
| 69 examples | Good | Quick prototype |
| 1,000 examples | Better | Development |
| 10,000 examples | Very Good | Production |
| 100,000 examples | Excellent | Research/Commercial |

### Why 100k Examples?

1. **Better Generalization**: Model sees more diverse scenarios
2. **Stronger Reasoning**: More examples of chain-of-thought
3. **Improved Self-Correction**: More error patterns learned
4. **Production Quality**: Comparable to commercial models

## Practical Recommendations

### For Testing (1-2 hours)
```cmd
--target-size 1000
```

### For Development (8 hours overnight)
```cmd
--target-size 10000
```

### For Production (weekend)
```cmd
--target-size 50000
```

### For Research (week)
```cmd
--target-size 100000
```

## Monitoring

### Check Progress
```cmd
# Count generated examples
wsl wc -l trd_framework/large_dataset/physics_large_dataset.jsonl

# Check checkpoint
wsl cat trd_framework/large_dataset/physics_checkpoint.json
```

### Check GPU/CPU Usage
```cmd
# GPU (if teacher uses GPU)
wsl nvidia-smi

# CPU
wsl htop
```

### Check Disk Space
```cmd
# ~25MB per 1000 examples
# 100k examples = ~2.5GB

wsl df -h
```

## Troubleshooting

### Issue: Generation Too Slow

**Solution**: Use faster teacher model
```cmd
--teacher-model qwen2.5:14b  # Smaller, faster
```

### Issue: Disk Space

**Solution**: Generate in chunks
```cmd
# Generate 10k at a time
--target-size 10000

# Then merge files later
cat dataset1.jsonl dataset2.jsonl > combined.jsonl
```

### Issue: Ollama Crashes

**Solution**: Automatic resume
```cmd
# Just restart Ollama and re-run
ollama run qwen3:30b-a3b

# Script resumes from checkpoint automatically
```

## Cost-Benefit Analysis

| Approach | Time | Quality | Best For |
|----------|------|---------|----------|
| Small (69) | 20 min | Good | Testing |
| Medium (1k) | 3 hours | Better | Development |
| Large (10k) | 1 day | Very Good | Production |
| Huge (100k) | 1 week | Excellent | Research |

## Summary

**To generate 100k examples:**

1. Start Ollama: `ollama run qwen3:30b-a3b`
2. Run generator: `python trd_framework/generate_large_dataset.py --target-size 100000`
3. Wait ~33 hours (can pause/resume)
4. Train model: `python trd_framework/run_phase2_training.py --dataset large_dataset.jsonl`

**Result**: World-class reasoning model trained on 100k structured examples! 🚀
