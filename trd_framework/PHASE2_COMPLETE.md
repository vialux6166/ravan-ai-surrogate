# TRD Phase 2 Complete: Student Model Training

## 🎉 Summary

Successfully implemented **Task 24.3**: Student Model Training with Stabilizer Loss!

The TRD framework now has a complete training pipeline that teaches a small student model (Qwen2-1.5B) to reason like a large teacher model (Qwen 32B) using structured chain-of-thought examples.

## ✅ What Was Built

### Core Implementation

#### 1. Student Training Pipeline (`phase2_student_training.py`)
- ✅ **TRDTrainingConfig**: Complete configuration dataclass
- ✅ **ResonanceDataset**: Custom dataset loader for JSONL format
- ✅ **TRDStabilizerLoss**: Multi-component loss function
- ✅ **TRDTrainer**: Complete training orchestrator
- ✅ **LoRA Integration**: Efficient fine-tuning with 4-bit quantization
- ✅ **PEFT Support**: Parameter-efficient training

#### 2. Training Script (`run_phase2_training.py`)
- ✅ Command-line interface with sensible defaults
- ✅ Support for physics, general, and hybrid datasets
- ✅ Comprehensive error handling and troubleshooting
- ✅ Progress tracking and logging
- ✅ Checkpoint saving and resumption

#### 3. Testing Utility (`test_trd_model.py`)
- ✅ Interactive testing mode
- ✅ Example query testing
- ✅ Model loading and inference
- ✅ Support for both physics and general domains

## 🔧 Technical Architecture

### Training Pipeline

```
Input: Resonance Dataset (JSONL)
├─ Query: "Explain quantum tunneling..."
├─ Initial Thought: "The wavepacket will..."
├─ Chain of Reasoning: "Step 1: Calculate..."
├─ Identified Flaws: "I initially assumed..."
├─ Self-Correction: "Actually, for thick barriers..."
└─ Final Answer: "The transmission coefficient..."

        ↓ TRD Training ↓

Student Model (Qwen2-1.5B + LoRA)
├─ Base Model: Qwen2-1.5B-Instruct
├─ LoRA Adapters: r=8, alpha=16
├─ Quantization: 4-bit (NF4)
├─ Training: Stabilizer Loss
└─ Output: Structured reasoning model
```

### Stabilizer Loss Function

```python
L_total = w1·L_final + w2·L_reasoning + w3·L_correction

Where:
- L_final (w1=1.0): Ensures correct final answers
- L_reasoning (w2=2.0): Teaches step-by-step process
- L_correction (w3=3.0): Teaches error detection

Analogy to TRA:
- L_reasoning ≈ Gauss's Law (logical charge conservation)
- L_correction ≈ Zero Flux (penalizes inconsistencies)
```

### LoRA Configuration

```python
LoRA Parameters:
├─ Rank (r): 8
├─ Alpha: 16
├─ Dropout: 0.1
├─ Target Modules: [q_proj, v_proj, k_proj, o_proj]
└─ Trainable Parameters: ~0.5% of total

Benefits:
✅ 200x fewer parameters to train
✅ Faster training (2-4 hours vs days)
✅ Lower memory requirements
✅ Easy to swap/merge adapters
```

### 4-bit Quantization

```python
Quantization Config:
├─ Type: NF4 (Normal Float 4-bit)
├─ Compute dtype: float16
├─ Double quantization: Enabled
└─ VRAM Savings: ~75%

Memory Usage:
- Base Model (FP16): ~3GB
- With 4-bit: ~2GB
- LoRA Adapters: ~50MB
- Total: ~2.5GB VRAM
```

## 📊 Training Configuration

### Default Settings (Optimized for RTX 3090)

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Student Model** | Qwen2-1.5B-Instruct | 1.5B parameters |
| **Batch Size** | 2 | Per device |
| **Gradient Accumulation** | 8 | Effective batch = 16 |
| **Learning Rate** | 2e-4 | AdamW optimizer |
| **Epochs** | 3 | Typically sufficient |
| **Max Length** | 2048 | Tokens |
| **LoRA Rank** | 8 | Balance efficiency/quality |
| **LoRA Alpha** | 16 | Scaling factor |
| **Quantization** | 4-bit NF4 | Memory efficient |
| **Optimizer** | Paged AdamW 8-bit | Memory efficient |

### Hardware Requirements

**Minimum**:
- GPU: RTX 3090 (24GB VRAM)
- RAM: 32GB
- Storage: 10GB free

**Recommended**:
- GPU: RTX 5090 (32GB VRAM)
- RAM: 64GB
- Storage: 20GB free

**Training Time**:
- Physics dataset (69 examples): ~1-2 hours
- General dataset (35 examples): ~30-60 minutes
- Hybrid dataset (104 examples): ~2-4 hours

## 🚀 Usage Guide

### Basic Training (Physics Mode)

```bash
# Prerequisites:
# 1. Phase 1 complete (resonance dataset exists)
# 2. WSL with GPU access
# 3. Python environment activated

# Train on physics dataset
wsl bash -c "cd '/mnt/c/Users/windows/Documents/rishi - professor' && \
  source venv_test/bin/activate && \
  python trd_framework/run_phase2_training.py \
    --dataset trd_framework/ravan_trd_complete_dataset.jsonl \
    --output-dir trd_framework/models/qwen2-1.5b-ravan-physics \
    --epochs 3 2>&1"

# Output: trd_framework/models/qwen2-1.5b-ravan-physics/
```

### Basic Training (General Mode)

```bash
# Train on general dataset
wsl bash -c "cd '/mnt/c/Users/windows/Documents/rishi - professor' && \
  source venv_test/bin/activate && \
  python trd_framework/run_phase2_training.py \
    --dataset trd_framework/general_trd_complete_dataset.jsonl \
    --output-dir trd_framework/models/qwen2-1.5b-general \
    --epochs 3 2>&1"

# Output: trd_framework/models/qwen2-1.5b-general/
```

### Advanced Training (Custom Parameters)

```bash
# Custom training with reduced memory usage
python trd_framework/run_phase2_training.py \
    --dataset trd_framework/ravan_trd_complete_dataset.jsonl \
    --output-dir trd_framework/models/qwen2-1.5b-custom \
    --batch-size 1 \
    --gradient-accumulation 16 \
    --epochs 5 \
    --learning-rate 1e-4 \
    --lora-r 16 \
    --lora-alpha 32 \
    --max-length 1024
```

### Resume Training from Checkpoint

```bash
# Resume if training was interrupted
python trd_framework/run_phase2_training.py \
    --dataset trd_framework/ravan_trd_complete_dataset.jsonl \
    --output-dir trd_framework/models/qwen2-1.5b-ravan-physics \
    --resume-from trd_framework/models/qwen2-1.5b-ravan-physics/checkpoint-100
```

## 🧪 Testing the Trained Model

### Interactive Testing

```bash
# Test model interactively
wsl bash -c "cd '/mnt/c/Users/windows/Documents/rishi - professor' && \
  source venv_test/bin/activate && \
  python trd_framework/test_trd_model.py \
    --model-path trd_framework/models/qwen2-1.5b-ravan-physics \
    --mode interactive 2>&1"

# Then enter queries:
# Query: Explain quantum tunneling through a barrier
# [Model generates structured response]
```

### Example Query Testing

```bash
# Test with predefined examples
python trd_framework/test_trd_model.py \
    --model-path trd_framework/models/qwen2-1.5b-ravan-physics \
    --mode examples \
    --domain physics
```

## 📁 Output Structure

After training, you'll have:

```
trd_framework/models/qwen2-1.5b-ravan-physics/
├── adapter_config.json          # LoRA configuration
├── adapter_model.bin            # LoRA weights (~50MB)
├── tokenizer_config.json        # Tokenizer config
├── tokenizer.json               # Tokenizer vocabulary
├── special_tokens_map.json      # Special tokens
├── trd_config.json              # Training metadata
└── checkpoint-*/                # Training checkpoints
```

**Key Files**:
- `adapter_model.bin`: The trained LoRA weights (this is what you deploy!)
- `trd_config.json`: Training statistics and configuration
- Base model (Qwen2-1.5B) is loaded separately

## 🎯 Expected Results

### Training Metrics

| Metric | Expected Value | Notes |
|--------|----------------|-------|
| Training Loss | 1.5-2.5 | Lower is better |
| Training Time | 1-4 hours | Depends on dataset size |
| VRAM Usage | 2-3GB | With 4-bit quantization |
| Model Size | ~50MB | LoRA adapters only |
| Total Size | ~3.2GB | Base + adapters |

### Model Capabilities

**Physics Mode**:
- ✅ Structured reasoning about quantum mechanics
- ✅ Step-by-step problem solving
- ✅ Self-correction of physics errors
- ✅ Conservation law awareness
- ✅ Natural language explanations

**General Mode**:
- ✅ Multi-domain reasoning
- ✅ Math problem solving
- ✅ Code debugging
- ✅ Scientific explanations
- ✅ Creative writing assistance

## ⚠️ Troubleshooting

### Issue: CUDA Out of Memory

```bash
# Solution 1: Reduce batch size
python trd_framework/run_phase2_training.py \
    --dataset your_dataset.jsonl \
    --batch-size 1 \
    --gradient-accumulation 16

# Solution 2: Reduce max length
python trd_framework/run_phase2_training.py \
    --dataset your_dataset.jsonl \
    --max-length 1024

# Solution 3: Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

### Issue: Model Download Fails

```bash
# Check internet connection
ping huggingface.co

# Set HuggingFace cache directory
export HF_HOME=/path/to/cache

# Download model manually
python -c "from transformers import AutoModelForCausalLM; \
  AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-1.5B-Instruct')"
```

### Issue: Training Very Slow

```bash
# Check GPU utilization
nvidia-smi

# Increase batch size if VRAM allows
python trd_framework/run_phase2_training.py \
    --dataset your_dataset.jsonl \
    --batch-size 4 \
    --gradient-accumulation 4

# Use mixed precision (enabled by default)
# Verify FP16 is working in logs
```

### Issue: Loss Not Decreasing

```bash
# Try lower learning rate
python trd_framework/run_phase2_training.py \
    --dataset your_dataset.jsonl \
    --learning-rate 1e-4

# Increase LoRA rank
python trd_framework/run_phase2_training.py \
    --dataset your_dataset.jsonl \
    --lora-r 16 \
    --lora-alpha 32

# Train for more epochs
python trd_framework/run_phase2_training.py \
    --dataset your_dataset.jsonl \
    --epochs 5
```

## 📈 Performance Optimization

### Memory Optimization

1. **4-bit Quantization** (default): Saves ~75% VRAM
2. **Gradient Checkpointing**: Trades compute for memory
3. **Paged AdamW**: Efficient optimizer
4. **LoRA**: Only train 0.5% of parameters

### Speed Optimization

1. **Increase Batch Size**: If VRAM allows
2. **Reduce Max Length**: If sequences are shorter
3. **Use FP16**: Enabled by default
4. **Persistent Workers**: For data loading

### Quality Optimization

1. **Increase LoRA Rank**: r=16 or r=32
2. **More Epochs**: 5-10 epochs
3. **Lower Learning Rate**: 1e-4 or 5e-5
4. **Larger Dataset**: More examples = better learning

## 🔬 Advanced Features

### Custom Loss Weights

Modify `TRDTrainingConfig` in code:

```python
config = TRDTrainingConfig(
    w_final=1.0,      # Final answer weight
    w_reasoning=3.0,  # Increase reasoning emphasis
    w_correction=5.0  # Increase correction emphasis
)
```

### Multi-GPU Training

```python
# Automatic with device_map="auto"
# Distributes model across available GPUs
```

### Gradient Accumulation

```bash
# Simulate larger batch size
# Effective batch = batch_size * gradient_accumulation
python trd_framework/run_phase2_training.py \
    --batch-size 1 \
    --gradient-accumulation 32  # Effective batch = 32
```

## 📚 Next Steps

### After Training Completes

1. **Test the Model**:
```bash
python trd_framework/test_trd_model.py \
    --model-path trd_framework/models/qwen2-1.5b-ravan-physics \
    --mode interactive
```

2. **Evaluate Performance** (Task 24.4):
```bash
python trd_framework/run_phase3_evaluation.py \
    --model-path trd_framework/models/qwen2-1.5b-ravan-physics
```

3. **Deploy to Ravan** (Task 24.8):
```bash
# Integration with Ravan system
# Use the trained model as LLM backend
```

## 🎓 Key Insights

### What Makes TRD Training Special

1. **Structured Learning**: Not just answers, but reasoning process
2. **Self-Correction**: Model learns to identify and fix errors
3. **Efficient**: LoRA + 4-bit = fast training on consumer GPUs
4. **Transferable**: Same framework works for any domain

### Comparison to Standard Fine-Tuning

| Aspect | Standard Fine-Tuning | TRD Training |
|--------|---------------------|--------------|
| **Focus** | Final answers | Reasoning process |
| **Dataset** | Q&A pairs | Structured reasoning |
| **Loss** | Simple CE | Stabilizer Loss |
| **Output** | Direct answers | Chain-of-thought |
| **Self-Correction** | No | Yes |
| **Compression** | Moderate | High |

## 🏆 Achievements

✅ **Task 24.3**: Student Model Training - COMPLETE  
✅ **LoRA Integration**: Efficient fine-tuning  
✅ **4-bit Quantization**: Memory optimization  
✅ **Stabilizer Loss**: Multi-component learning  
✅ **Testing Utilities**: Interactive and batch testing  
✅ **Documentation**: Comprehensive guides  

**Next**: Task 24.4 - Model Evaluation

## 📞 Support

If you encounter issues:

1. Check VRAM usage: `nvidia-smi`
2. Review training logs for errors
3. Try reduced batch size or max length
4. Verify dataset format (JSONL)
5. Ensure Phase 1 completed successfully

## 🎯 Conclusion

Phase 2 is complete! You now have:

- ✅ Complete training pipeline
- ✅ Efficient LoRA + 4-bit implementation
- ✅ Testing utilities
- ✅ Comprehensive documentation
- ✅ Ready for Phase 3 (Evaluation)

**The student is ready to learn from the teacher!**

---

**Date**: 2024-10-20  
**Status**: Phase 2 Complete ✅  
**Next**: Phase 3 - Model Evaluation (Task 24.4)  
**Files**: 3 Python files, comprehensive training pipeline
