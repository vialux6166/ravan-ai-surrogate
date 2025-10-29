# Transformers Library Usage in Ravan Quantum-ML System

## Overview

The Ravan system uses the HuggingFace `transformers` library for LLM integration, providing natural language interfaces for quantum simulations.

## Implementation Details

### Core Components

#### 1. LLM Adapter (`src/llm/llm_adapter.py`)

The LLMAdapter class uses transformers for:

- **Model Loading**: `AutoModelForCausalLM.from_pretrained()`
- **Tokenization**: `AutoTokenizer.from_pretrained()`
- **4-bit Quantization**: `BitsAndBytesConfig` for memory efficiency
- **Generation**: `model.generate()` with configurable parameters

**Key Features:**
```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig
)

# 4-bit quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)
```

#### 2. Natural Language Interface (`src/llm/nl_interface.py`)

Uses transformers to convert natural language descriptions into simulation configurations:

- Detects simulator type from keywords
- Generates JSON parameters using LLM
- Validates and clamps parameters to valid ranges

**Example:**
```python
# Input: "Run quantum tunneling with high barrier"
# Output: {'simulator': 'schrodinger', 'parameters': {...}}
```

#### 3. Results Explainer (`src/llm/results_explainer.py`)

Uses transformers to generate human-readable explanations:

- Explains simulation results in plain language
- Provides physics context
- Compares multiple results
- Suggests next experiments

#### 4. Code Generator (`src/llm/code_generator.py`)

Uses transformers to generate executable Python/Qiskit code:

- Task description → Python code
- Syntax validation using AST
- Safety checks (no file I/O, system calls)
- Qiskit/NumPy/SciPy focus

## Dependencies

```txt
transformers>=4.35.0        # Core library
peft>=0.6.0                 # Parameter-Efficient Fine-Tuning
bitsandbytes>=0.41.0        # 4-bit/8-bit quantization
accelerate>=0.24.0          # Distributed inference
sentencepiece>=0.1.99       # Tokenization
```

## Memory Optimization

### 4-bit Quantization

The system uses 4-bit quantization to reduce VRAM requirements:

- **Full precision**: ~60GB VRAM
- **4-bit quantized**: ~22GB VRAM (RTX 3090 compatible)

### VRAM Management

```python
# Check available VRAM before loading
vram_ok, free_vram = llm.check_vram_available(required_gb=22.0)

# Automatic CPU fallback if insufficient VRAM
if not vram_ok:
    use_gpu = False  # Falls back to CPU
```

## Usage Examples

### Basic Query

```python
from src.llm.llm_adapter import LLMAdapter, LLMConfig

# Initialize adapter
config = LLMConfig(model_name="Qwen/Qwen-30B-Chat", load_in_4bit=True)
llm = LLMAdapter(config)

# Load model
llm.load()

# Query
response = llm.query("Explain quantum tunneling", max_new_tokens=200)

# Cleanup
llm.unload()
```

### Context Manager

```python
with LLMAdapter(config) as llm:
    response = llm.query("What is superposition?")
```

### Natural Language Config Generation

```python
from src.llm.nl_interface import SimulationConfigGenerator

generator = SimulationConfigGenerator(llm)
config = generator.generate_simulation_config(
    "Run quantum tunneling with high barrier and narrow wavepacket"
)
# Returns: {'simulator': 'schrodinger', 'parameters': {...}}
```

### Results Explanation

```python
from src.llm.results_explainer import ResultsExplainer

explainer = ResultsExplainer(llm)
explanation = explainer.explain_results(
    simulator_type='schrodinger',
    parameters={'V0': 5.0, 'k0': 4.0},
    results={'transmission': 0.65, 'reflection': 0.35}
)
```

### Code Generation

```python
from src.llm.code_generator import CodeGenerator

generator = CodeGenerator(llm)
code = generator.generate_code(
    "Create a Bell state circuit with measurement",
    code_type='quantum_circuit'
)
```

## Model Support

### Default Model
- **Qwen/Qwen-30B-Chat**: 30B parameter chat model optimized for technical tasks

### Compatible Models
Any HuggingFace causal language model:
- Qwen series (Qwen-7B, Qwen-14B, Qwen-30B)
- Llama 2/3 series
- Mistral series
- CodeLlama for code generation

### Local Models (Ollama)
The system also supports local Ollama models:
```python
# Tested with qwen3:30b-a3b
# 100% test pass rate for NL interface
```

## Performance

### Inference Speed
- **GPU (RTX 3090)**: ~50 tokens/sec
- **CPU**: ~2-5 tokens/sec (fallback mode)

### Memory Usage
- **Model**: ~22GB VRAM (4-bit quantized)
- **Inference**: +2-4GB VRAM per batch

## Safety Features

### Code Generation Safety
- Whitelist of allowed imports (qiskit, numpy, scipy, matplotlib)
- Blocked patterns (os, subprocess, eval, exec, file I/O)
- AST syntax validation
- No arbitrary code execution

### Prompt Validation
- Input sanitization
- Length limits
- Injection attack prevention

## Testing

All LLM components have been tested:
- ✅ Natural language interface (100% pass rate with Ollama)
- ✅ Results explanation
- ✅ Code generation with validation
- ✅ Prompt manager with caching

## Future Enhancements

1. **Fine-tuning**: Use PEFT/LoRA for domain-specific adaptation
2. **Streaming**: Implement token streaming for real-time responses
3. **Multi-modal**: Add support for diagram/circuit visualization
4. **Batch Processing**: Optimize for multiple queries
5. **Model Caching**: Implement model weight caching for faster loading

## References

- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers)
- [BitsAndBytes Quantization](https://github.com/TimDettmers/bitsandbytes)
- [PEFT Library](https://github.com/huggingface/peft)
- [Qwen Models](https://huggingface.co/Qwen)
