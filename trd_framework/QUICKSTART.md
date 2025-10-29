# TRD Framework Quick Start Guide

## 🚀 Get Started in 3 Steps

### Prerequisites

- ✅ Ravan v1.0.0 installed
- ✅ WSL with GPU access
- ✅ Python environment activated

### Step 1: Install Ollama (Teacher Model)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the teacher model (Qwen 32B)
ollama pull qwen2.5:32b

# Verify installation
ollama list
```

### Step 2: Choose Your Mode

#### Option A: Physics Mode (Ravan-Specific)

```bash
# Already generated: 69 physics queries ✅
# File: trd_framework/ravan_physics_queries.jsonl

# Start teacher model (in terminal 1)
ollama run qwen2.5:32b

# Generate resonance dataset (in terminal 2)
wsl bash -c "cd '/mnt/c/Users/windows/Documents/rishi - professor' && \
  source venv_test/bin/activate && \
  python trd_framework/run_phase1_complete.py --mode physics 2>&1"

# Time: ~10-15 minutes
# Output: trd_framework/ravan_trd_complete_dataset.jsonl
```

#### Option B: General Mode (Multi-Domain)

```bash
# Already generated: 35 general queries ✅
# File: trd_framework/general_purpose_queries.jsonl

# Start teacher model (in terminal 1)
ollama run qwen2.5:32b

# Generate resonance dataset (in terminal 2)
wsl bash -c "cd '/mnt/c/Users/windows/Documents/rishi - professor' && \
  source venv_test/bin/activate && \
  python trd_framework/run_phase1_complete.py --mode general 2>&1"

# Time: ~5-8 minutes
# Output: trd_framework/general_trd_complete_dataset.jsonl
```

#### Option C: Both Modes (Hybrid)

```bash
# Run both modes to create a hybrid model
# Physics mode first
python trd_framework/run_phase1_complete.py --mode physics

# Then general mode
python trd_framework/run_phase1_complete.py --mode general

# Combine datasets
cat trd_framework/ravan_trd_complete_dataset.jsonl \
    trd_framework/general_trd_complete_dataset.jsonl \
    > trd_framework/hybrid_trd_dataset.jsonl

# Time: ~15-23 minutes total
```

### Step 3: Train Student Model (Coming Next)

```bash
# After Phase 1 completes, train the student model
# (Phase 2 implementation - Task 24.3)

python trd_framework/run_phase2_training.py \
    --dataset trd_framework/ravan_trd_complete_dataset.jsonl \
    --output-dir trd_framework/models/qwen2-1.5b-ravan-trd \
    --epochs 3

# Time: ~2-4 hours on RTX 3090
```

## 📊 What You'll Get

### Physics Mode Output
- **Model**: Quantum physics reasoning expert
- **Size**: 3.2GB (vs 60GB teacher)
- **Capabilities**: 
  - Explain Ravan simulations
  - Generate quantum physics code
  - Validate ML predictions
  - Self-correct physics errors

### General Mode Output
- **Model**: Multi-domain reasoning assistant
- **Size**: 3.2GB
- **Capabilities**:
  - Math problem solving
  - Code debugging
  - Scientific explanations
  - Creative writing
  - Business analysis
  - Medical reasoning

### Hybrid Mode Output
- **Model**: Physics expert + general reasoning
- **Size**: 3.2GB
- **Capabilities**: Best of both worlds!

## 🔍 Verify Your Setup

### Check Query Datasets

```bash
# Physics queries (should exist)
wc -l trd_framework/ravan_physics_queries.jsonl
# Expected: 69 lines

# General queries (should exist)
wc -l trd_framework/general_purpose_queries.jsonl
# Expected: 35 lines
```

### Test Ollama Connection

```bash
# Test if Ollama is running
curl http://localhost:11434/api/tags

# Should return JSON with available models
```

### Check GPU Access

```bash
# In WSL
nvidia-smi

# Should show your GPU (RTX 3090/5090)
```

## ⚠️ Troubleshooting

### Issue: Ollama not found

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve
```

### Issue: Teacher model not responding

```bash
# Check if model is running
ollama ps

# If not, start it
ollama run qwen2.5:32b
```

### Issue: CUDA errors in Windows

```bash
# Always use WSL for GPU operations
wsl bash -c "cd '/mnt/c/Users/windows/Documents/rishi - professor' && \
  source venv_test/bin/activate && \
  python your_script.py 2>&1"
```

### Issue: Out of memory

```bash
# Reduce batch size in Phase 2
python trd_framework/run_phase2_training.py \
    --batch-size 1 \
    --gradient-accumulation 16
```

## 📚 Next Steps

1. ✅ **Phase 1 Complete**: Query datasets generated
2. ⏳ **Phase 1 Teacher**: Run teacher model to create resonance dataset
3. 🔜 **Phase 2**: Train student model (Task 24.3)
4. 🔜 **Phase 3**: Evaluate model (Task 24.4)
5. 🔜 **Deploy**: Integrate with Ravan (Task 24.8)

## 💡 Tips

- **Start with Physics Mode** if you're using Ravan
- **Start with General Mode** if you want a versatile assistant
- **Use Hybrid Mode** if you want both capabilities
- **Monitor VRAM** during training (use `nvidia-smi`)
- **Save checkpoints** during long training runs
- **Test incrementally** after each phase

## 📖 Documentation

- `README.md` - Complete framework documentation
- `GENERAL_VS_PHYSICS_COMPARISON.md` - Mode comparison
- `PHASE1_AND_GENERAL_COMPLETE.md` - Implementation summary
- `QUICKSTART.md` - This file

## 🎯 Current Status

```
✅ Phase 1: Dataset Generation - COMPLETE
   ├─ Physics queries: 69 ✅
   ├─ General queries: 35 ✅
   └─ Pipeline: Ready ✅

⏳ Phase 1: Teacher Querying - READY TO RUN
   └─ Waiting for: ollama run qwen2.5:32b

🔜 Phase 2: Student Training - NEXT
🔜 Phase 3: Evaluation - AFTER PHASE 2
🔜 Phase 4: Deployment - FINAL
```

## 🚀 Ready to Go!

You're all set! Just run:

```bash
# Terminal 1: Start teacher
ollama run qwen2.5:32b

# Terminal 2: Generate dataset
python trd_framework/run_phase1_complete.py --mode physics
```

And watch the magic happen! 🎉
