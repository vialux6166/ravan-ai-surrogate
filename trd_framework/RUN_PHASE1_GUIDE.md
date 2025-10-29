# Phase 1: Generate Resonance Dataset with Ollama

## Prerequisites

You have these Qwen models available:
- `qwen3-vl:235b-cloud` (cloud model)
- `qwen3:30b-a3b` (30B parameters, local)
- `deepseek-r1:latest` (8.2B parameters)

## Recommended: Use qwen3:30b-a3b

This is your best local model for the teacher role.

## Step-by-Step Instructions

### Option 1: Run in Windows PowerShell

#### Terminal 1: Start Ollama with Teacher Model
```powershell
# Start the teacher model
ollama run qwen3:30b-a3b
```

Keep this terminal open. The model will be ready when you see the prompt.

#### Terminal 2: Generate Resonance Dataset

**For Physics Mode** (69 queries, ~15-20 minutes):
```powershell
wsl bash -c "cd '/mnt/c/Users/windows/Documents/rishi - professor' && source venv_test/bin/activate && python trd_framework/run_phase1_complete.py --mode physics --teacher-model qwen3:30b-a3b --teacher-url http://host.docker.internal:11434 2>&1"
```

**For General Mode** (35 queries, ~8-12 minutes):
```powershell
wsl bash -c "cd '/mnt/c/Users/windows/Documents/rishi - professor' && source venv_test/bin/activate && python trd_framework/run_phase1_complete.py --mode general --teacher-model qwen3:30b-a3b --teacher-url http://host.docker.internal:11434 2>&1"
```

### Option 2: If WSL Can't Access Windows Ollama

Run everything in Windows PowerShell (not WSL):

#### Terminal 1: Start Teacher Model
```powershell
ollama run qwen3:30b-a3b
```

#### Terminal 2: Generate Dataset (Windows)
```powershell
# Activate conda/venv if needed
conda activate base

# Run Phase 1
python trd_framework/run_phase1_complete.py --mode physics --teacher-model qwen3:30b-a3b
```

### Option 3: Alternative - Pull qwen2.5:32b

If you want the exact model from the guide:

```powershell
# Pull the model (this will download ~20GB)
ollama pull qwen2.5:32b

# Then run Phase 1
ollama run qwen2.5:32b  # Terminal 1

# Terminal 2
python trd_framework/run_phase1_complete.py --mode physics --teacher-model qwen2.5:32b
```

## What Will Happen

### Phase 1 Output

You'll see:
```
======================================================================
TRD PHASE 1 COMPLETE: RESONANCE DATASET GENERATION
======================================================================

Mode: PHYSICS
Teacher Model: qwen3:30b-a3b
Output Directory: trd_framework

======================================================================
STEP 1: GENERATE PHYSICS-SPECIFIC QUERIES
======================================================================

[Ravan integration runs, generates 69 queries]

✓ Step 1 Complete: 69 queries generated

======================================================================
STEP 2: QUERY TEACHER MODEL
======================================================================

This will query qwen3:30b-a3b for each question...
Ready to query teacher model?
This will take approximately 10-14 minutes for 69 queries.

[1/69] Processing: A quantum wavepacket with initial momentum...
  ✓ Generated 1234 chars of reasoning

[2/69] Processing: A quantum wavepacket with initial momentum...
  ✓ Generated 987 chars of reasoning

...

✓ Step 2 Complete: 69 reasoning examples generated

======================================================================
STEP 3: MERGE METADATA
======================================================================

✓ Step 3 Complete: 69 complete examples

======================================================================
PHASE 1 COMPLETE!
======================================================================

Generated Files:
  1. Queries: trd_framework/ravan_physics_queries.jsonl
  2. Resonance: trd_framework/ravan_resonance_dataset.jsonl
  3. Complete: trd_framework/ravan_trd_complete_dataset.jsonl

Total Examples: 69

Next Step: Run Phase 2 (Student Model Training)
  python trd_framework/run_phase2_training.py
```

## After Phase 1 Completes

You'll have the file:
- `trd_framework/ravan_trd_complete_dataset.jsonl` (for physics mode)
- OR `trd_framework/general_trd_complete_dataset.jsonl` (for general mode)

## Then Run Phase 2 Training

```powershell
wsl bash -c "cd '/mnt/c/Users/windows/Documents/rishi - professor' && source venv_test/bin/activate && python trd_framework/run_phase2_training.py --dataset trd_framework/ravan_trd_complete_dataset.jsonl --output-dir trd_framework/models/qwen2-1.5b-ravan-physics --epochs 3 --batch-size 2 --gradient-accumulation 8 2>&1"
```

## Troubleshooting

### Issue: WSL can't connect to Ollama

Try using `host.docker.internal` instead of `localhost`:
```powershell
python trd_framework/run_phase1_complete.py --mode physics --teacher-url http://host.docker.internal:11434
```

### Issue: Teacher model not responding

Check if Ollama is running:
```powershell
curl http://localhost:11434/api/tags
```

Should return JSON with model list.

### Issue: Slow generation

This is normal! Each query takes 10-20 seconds because the teacher model is:
1. Reading the query
2. Generating initial thought
3. Creating chain of reasoning
4. Identifying flaws
5. Self-correcting
6. Providing final answer

Total time: ~10-20 minutes for 69 queries

## Quick Test First (Recommended)

Test with just 3 queries to verify everything works:

```powershell
# Terminal 1
ollama run qwen3:30b-a3b

# Terminal 2
python trd_framework/phase1_dataset_generation.py
# Then manually test with a single query
```

## Summary

1. ✅ Start Ollama with teacher model: `ollama run qwen3:30b-a3b`
2. ✅ Run Phase 1: `python trd_framework/run_phase1_complete.py --mode physics --teacher-model qwen3:30b-a3b`
3. ⏳ Wait 10-20 minutes
4. ✅ Check output: `trd_framework/ravan_trd_complete_dataset.jsonl`
5. ✅ Run Phase 2 training

You're ready to go! 🚀
