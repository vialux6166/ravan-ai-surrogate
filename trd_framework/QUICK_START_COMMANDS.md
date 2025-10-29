# TRD Quick Start Commands

## Your Setup
- **Teacher Model**: qwen3:30b-a3b (30B parameters, 18GB)
- **Student Model**: Qwen2-1.5B-Instruct (1.5B parameters, 3.2GB)
- **GPU**: RTX 3090/5090

## Phase 1: Generate Resonance Dataset (~15-20 minutes)

### Option 1: Use Batch File (Easiest)
```cmd
cd "C:\Users\windows\Documents\rishi - professor"
trd_framework\START_PHASE1.bat
```

### Option 2: Manual Commands

**Terminal 1 - Start Teacher Model:**
```cmd
ollama run qwen3:30b-a3b
```

**Terminal 2 - Generate Dataset:**
```cmd
wsl bash -c "cd '/mnt/c/Users/windows/Documents/rishi - professor' && source venv_test/bin/activate && python trd_framework/run_phase1_complete.py --mode physics --teacher-model qwen3:30b-a3b 2>&1"
```

**Output**: `trd_framework/ravan_trd_complete_dataset.jsonl` (69 examples)

---

## Phase 2: Train Student Model (~1-2 hours)

### Option 1: Use Batch File (Easiest)
```cmd
cd "C:\Users\windows\Documents\rishi - professor"
trd_framework\START_PHASE2.bat
```

### Option 2: Manual Command
```cmd
wsl bash -c "cd '/mnt/c/Users/windows/Documents/rishi - professor' && source venv_test/bin/activate && python trd_framework/run_phase2_training.py --dataset trd_framework/ravan_trd_complete_dataset.jsonl --output-dir trd_framework/models/qwen2-1.5b-ravan-physics --epochs 3 --batch-size 2 --gradient-accumulation 8 2>&1"
```

**Output**: `trd_framework/models/qwen2-1.5b-ravan-physics/`

---

## Phase 3: Test the Model

### Interactive Testing
```cmd
wsl bash -c "cd '/mnt/c/Users/windows/Documents/rishi - professor' && source venv_test/bin/activate && python trd_framework/test_trd_model.py --model-path trd_framework/models/qwen2-1.5b-ravan-physics --mode interactive 2>&1"
```

Then enter queries like:
- "Explain quantum tunneling through a barrier"
- "What is the Von Neumann entropy of a Bell state?"
- "Derive the energy eigenvalues of a harmonic oscillator"

---

## Quick Test (5 minutes)

Test the pipeline with 3 examples:
```cmd
wsl bash -c "cd '/mnt/c/Users/windows/Documents/rishi - professor' && source venv_test/bin/activate && python trd_framework/run_phase2_training.py --dataset trd_framework/test_dataset_small.jsonl --output-dir trd_framework/models/test-model --epochs 1 --batch-size 1 --gradient-accumulation 2 2>&1"
```

---

## Monitor Progress

### Check GPU Usage
```cmd
wsl nvidia-smi
```

### Check Ollama Status
```cmd
curl http://localhost:11434/api/tags
```

### Check Dataset Size
```cmd
wsl wc -l trd_framework/ravan_trd_complete_dataset.jsonl
```

---

## Expected Timeline

| Phase | Time | Output |
|-------|------|--------|
| Phase 1 | 15-20 min | Resonance dataset (69 examples) |
| Phase 2 | 1-2 hours | Trained model (~3.2GB) |
| Phase 3 | 5 min | Evaluation results |
| **Total** | **~2-3 hours** | **Production-ready model** |

---

## Files You'll Get

```
trd_framework/
├── ravan_physics_queries.jsonl              # 69 physics queries
├── ravan_resonance_dataset.jsonl            # Teacher responses
├── ravan_trd_complete_dataset.jsonl         # Complete training data
└── models/
    └── qwen2-1.5b-ravan-physics/
        ├── adapter_model.bin                # LoRA weights (~50MB)
        ├── adapter_config.json              # LoRA config
        └── trd_config.json                  # Training metadata
```

---

## Troubleshooting

### Ollama Not Responding
```cmd
# Check if running
curl http://localhost:11434/api/tags

# Restart if needed
taskkill /F /IM ollama.exe
ollama serve
```

### CUDA Out of Memory
```cmd
# Reduce batch size
python trd_framework/run_phase2_training.py ... --batch-size 1 --gradient-accumulation 16
```

### Training Too Slow
```cmd
# Check GPU is being used
wsl nvidia-smi

# Should show Python process using GPU
```

---

## Ready to Start!

**Simplest way:**
1. Double-click `trd_framework\START_PHASE1.bat`
2. Wait 15-20 minutes
3. Double-click `trd_framework\START_PHASE2.bat`
4. Wait 1-2 hours
5. Done! 🎉

**Your specialized quantum physics reasoning model is ready!**
