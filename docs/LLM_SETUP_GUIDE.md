# LLM Integration Setup Guide

## Overview
This guide covers the setup and integration of Qwen 30B for natural language interaction with the Ravan Quantum-ML System.

**Task:** 15.1 - Install LLM dependencies  
**Status:** Ready for execution  
**Requirements:** RTX 3090 with 24GB VRAM, ~70GB disk space

---

## Prerequisites

### Hardware Requirements
- **GPU:** NVIDIA RTX 3090 (24GB VRAM) or equivalent
- **RAM:** 32GB system RAM recommended
- **Disk:** 70GB free space for model weights
- **CUDA:** Version 12.1 or higher

### Software Requirements
- Python 3.10+
- PyTorch 2.0+ with CUDA support
- WSL Ubuntu 22.04 (for Windows users)

---

## Installation Steps

### Step 1: Install LLM Dependencies

Run the installation script:

```bash
cd ~/ravan-quantum-ml
bash scripts/install_llm_dependencies.sh
```

This installs:
- **transformers** (≥4.35.0) - HuggingFace transformers library
- **peft** (≥0.6.0) - Parameter-Efficient Fine-Tuning
- **bitsandbytes** (≥0.41.0) - 4-bit quantization
- **accelerate** (≥0.24.0) - Distributed training utilities
- **sentencepiece** (≥0.1.99) - Tokenization
- **protobuf** (≥3.20.0) - Model serialization

**Expected time:** 2-5 minutes

### Step 2: Download Qwen 30B Model

Run the download script:

```bash
cd ~/ravan-quantum-ml
source venv/bin/activate
python scripts/download_qwen_model.py
```

**What it does:**
1. Checks disk space (requires ~70GB)
2. Checks GPU VRAM (requires ~22GB for inference)
3. Downloads Qwen 30B model weights (~60GB)
4. Verifies 4-bit quantization loading

**Expected time:** 30-60 minutes (depends on internet speed)

**Model location:** `~/.cache/huggingface/models--Qwen--Qwen-30B-Chat`

### Step 3: Verify Installation

Test that everything works:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Check CUDA
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen-30B-Chat",
    trust_remote_code=True
)

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-30B-Chat",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

print("✓ Model loaded successfully!")
```

---

## 4-Bit Quantization Details

### What is 4-bit Quantization?

4-bit quantization reduces model memory requirements by storing weights in 4 bits instead of 16 bits (FP16) or 32 bits (FP32).

**Benefits:**
- **Memory reduction:** ~75% less VRAM (60GB → 15GB)
- **Faster loading:** Reduced data transfer
- **Minimal accuracy loss:** NF4 quantization preserves model quality

**Configuration:**
```python
BitsAndBytesConfig(
    load_in_4bit=True,              # Enable 4-bit quantization
    bnb_4bit_quant_type="nf4",      # NormalFloat4 (optimal for LLMs)
    bnb_4bit_compute_dtype=torch.float16,  # Compute in FP16
    bnb_4bit_use_double_quant=True, # Double quantization for extra compression
)
```

### VRAM Requirements

| Configuration | VRAM Required | Notes |
|---------------|---------------|-------|
| FP32 (full precision) | ~120GB | Not feasible on single GPU |
| FP16 (half precision) | ~60GB | Requires A100 or multi-GPU |
| 4-bit (quantized) | ~22GB | ✓ Fits on RTX 3090 |
| 4-bit + offloading | ~15GB | Slower, uses CPU RAM |

**RTX 3090 (24GB):** Can run 4-bit Qwen 30B with ~2GB headroom

---

## VRAM Management Strategy

### Mutual Exclusion with Simulation/Training

The Ravan system uses **strict VRAM management** to prevent OOM errors:

```python
from src.utils.vram_manager import VRAMManager, WorkloadMode

vram_manager = VRAMManager()

# Request LLM mode (requires 22GB)
vram_manager.request_mode(WorkloadMode.LLM)

# Load LLM
llm = load_qwen_model()

# Use LLM
response = llm.query("Explain quantum tunneling")

# Unload LLM before simulation/training
vram_manager.unload_llm()
vram_manager.request_mode(WorkloadMode.SIMULATION)
```

**Workload Modes:**
- **SIMULATION:** ~1-2GB (quantum simulators)
- **TRAINING:** ~4-8GB (ML model training)
- **INFERENCE:** ~1-2GB (ML predictions)
- **LLM:** ~22GB (Qwen 30B in 4-bit)

**Rules:**
1. Only one mode active at a time
2. LLM must be unloaded before simulation/training
3. Automatic garbage collection on mode switch

---

## Alternative Models

If Qwen 30B is too large, consider these alternatives:

### Smaller Models (Fit with more headroom)

| Model | Parameters | 4-bit VRAM | Quality |
|-------|------------|------------|---------|
| Qwen-14B | 14B | ~10GB | Good |
| Qwen-7B | 7B | ~5GB | Decent |
| Mistral-7B | 7B | ~5GB | Good |
| Llama-2-13B | 13B | ~9GB | Good |

### Larger Models (Require offloading)

| Model | Parameters | 4-bit VRAM | Notes |
|-------|------------|------------|-------|
| Qwen-72B | 72B | ~45GB | Requires CPU offloading |
| Llama-2-70B | 70B | ~44GB | Requires CPU offloading |

**Recommendation:** Start with Qwen-14B for testing, upgrade to Qwen-30B for production.

---

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution:**
1. Check free VRAM: `nvidia-smi`
2. Unload other models: `vram_manager.unload_llm()`
3. Clear cache: `torch.cuda.empty_cache()`
4. Try smaller model (Qwen-14B)

### Issue: "Model download interrupted"

**Solution:**
- Downloads resume automatically
- Re-run: `python scripts/download_qwen_model.py`
- Check disk space: `df -h ~`

### Issue: "Slow inference on CPU"

**Solution:**
- Ensure CUDA is available: `torch.cuda.is_available()`
- Check GPU passthrough in WSL: `nvidia-smi`
- Verify model is on GPU: `model.device`

### Issue: "Import error: bitsandbytes"

**Solution:**
```bash
pip uninstall bitsandbytes
pip install bitsandbytes>=0.41.0
```

---

## Performance Benchmarks

### Loading Time
- **First load:** 2-5 minutes (loads from disk)
- **Subsequent loads:** 1-2 minutes (cached)
- **With offloading:** 3-7 minutes

### Inference Speed
- **Short queries (<50 tokens):** 1-3 seconds
- **Medium queries (50-200 tokens):** 3-10 seconds
- **Long queries (>200 tokens):** 10-30 seconds

### VRAM Usage
- **Idle:** ~22GB
- **During inference:** ~23GB (peak)
- **After unload:** <1GB

---

## Next Steps

After completing Task 15.1:

1. **Task 15.2:** Implement LLMAdapter class
   - Wrapper for model loading/unloading
   - Query interface
   - VRAM checking

2. **Task 15.3:** Integrate with VRAMManager
   - Add LLM mode
   - Implement mutual exclusion
   - Test mode transitions

3. **Task 16.1-16.3:** Natural Language Interface
   - Simulation config generation
   - Results explanation
   - Prompt caching

---

## References

- [Qwen Documentation](https://github.com/QwenLM/Qwen)
- [bitsandbytes Documentation](https://github.com/TimDettmers/bitsandbytes)
- [PEFT Documentation](https://github.com/huggingface/peft)
- [Transformers Documentation](https://huggingface.co/docs/transformers)

---

## Notes

- **Privacy:** All inference is local, no data sent to cloud
- **Cost:** Zero operational cost after initial setup
- **Offline:** Works without internet after download
- **Security:** No external API calls, fully sandboxed

**Status:** Ready for implementation ✓
