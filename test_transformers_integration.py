"""
Test Transformers Library Integration
Tests the HuggingFace transformers library with Ravan LLM components
"""

import sys
import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_transformers_import():
    """Test 1: Verify transformers library can be imported"""
    print("\n" + "="*70)
    print("TEST 1: Transformers Library Import")
    print("="*70)
    
    try:
        import transformers
        print(f"✓ transformers version: {transformers.__version__}")
        
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            GenerationConfig
        )
        print("✓ All required transformers components imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import transformers: {e}")
        return False


def test_torch_cuda():
    """Test 2: Check PyTorch and CUDA availability"""
    print("\n" + "="*70)
    print("TEST 2: PyTorch and CUDA")
    print("="*70)
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ GPU device: {torch.cuda.get_device_name(0)}")
            
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✓ Total VRAM: {total_vram:.2f} GB")
        else:
            print("⚠ CUDA not available - will use CPU mode")
        
        return True
    except Exception as e:
        print(f"✗ Error checking PyTorch/CUDA: {e}")
        return False


def test_llm_adapter_init():
    """Test 3: Initialize LLMAdapter"""
    print("\n" + "="*70)
    print("TEST 3: LLMAdapter Initialization")
    print("="*70)
    
    try:
        from src.llm.llm_adapter import LLMAdapter, LLMConfig
        
        # Test with small model for quick testing
        config = LLMConfig(
            model_name="gpt2",  # Small model for testing
            load_in_4bit=False,  # Disable quantization for CPU testing
            max_new_tokens=50
        )
        
        adapter = LLMAdapter(config)
        print(f"✓ LLMAdapter initialized with model: {config.model_name}")
        print(f"✓ Config: max_tokens={config.max_new_tokens}, temp={config.temperature}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to initialize LLMAdapter: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tokenizer_loading():
    """Test 4: Load tokenizer using transformers"""
    print("\n" + "="*70)
    print("TEST 4: Tokenizer Loading")
    print("="*70)
    
    try:
        from transformers import AutoTokenizer
        
        print("Loading GPT-2 tokenizer for testing...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        print("✓ Tokenizer loaded successfully")
        
        # Test tokenization
        test_text = "Quantum computing uses superposition"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"✓ Test tokenization: '{test_text}'")
        print(f"✓ Token IDs shape: {tokens['input_ids'].shape}")
        print(f"✓ Number of tokens: {len(tokens['input_ids'][0])}")
        
        # Test decoding
        decoded = tokenizer.decode(tokens['input_ids'][0])
        print(f"✓ Decoded text: '{decoded}'")
        
        return True
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading():
    """Test 5: Load small model using transformers"""
    print("\n" + "="*70)
    print("TEST 5: Model Loading (GPT-2 Small)")
    print("="*70)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print("Loading GPT-2 model (this may take a minute)...")
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        print("✓ Model loaded successfully")
        
        # Get model info
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model parameters: {num_params:,}")
        print(f"✓ Model device: {next(model.parameters()).device}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_text_generation():
    """Test 6: Generate text using transformers"""
    print("\n" + "="*70)
    print("TEST 6: Text Generation")
    print("="*70)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        print("Loading model and tokenizer...")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Test prompt
        prompt = "Quantum mechanics is"
        print(f"✓ Prompt: '{prompt}'")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        print(f"✓ Input tokens: {inputs['input_ids'].shape}")
        
        # Generate
        print("Generating text...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✓ Generated text: '{generated_text}'")
        
        return True
    except Exception as e:
        print(f"✗ Failed to generate text: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_llm_adapter_full():
    """Test 7: Full LLMAdapter workflow"""
    print("\n" + "="*70)
    print("TEST 7: LLMAdapter Full Workflow")
    print("="*70)
    
    try:
        from src.llm.llm_adapter import LLMAdapter, LLMConfig
        
        # Use small model for testing
        config = LLMConfig(
            model_name="gpt2",
            load_in_4bit=False,
            max_new_tokens=30,
            temperature=0.7
        )
        
        print("Initializing LLMAdapter...")
        adapter = LLMAdapter(config)
        
        print("Loading model...")
        success = adapter.load(force_cpu=True)
        if not success:
            print("✗ Failed to load model")
            return False
        
        print("✓ Model loaded successfully")
        
        # Get model info
        info = adapter.get_model_info()
        print(f"✓ Model info: {info}")
        
        # Test query
        prompt = "Quantum computing uses"
        print(f"✓ Querying with prompt: '{prompt}'")
        
        response = adapter.query(prompt, max_new_tokens=20)
        print(f"✓ Response: '{response}'")
        
        # Cleanup
        adapter.unload()
        print("✓ Model unloaded successfully")
        
        return True
    except Exception as e:
        print(f"✗ LLMAdapter workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_nl_interface():
    """Test 8: Natural Language Interface"""
    print("\n" + "="*70)
    print("TEST 8: Natural Language Interface")
    print("="*70)
    
    try:
        from src.llm.llm_adapter import LLMAdapter, LLMConfig
        from src.llm.nl_interface import SimulationConfigGenerator
        
        # Initialize with small model
        config = LLMConfig(model_name="gpt2", load_in_4bit=False)
        adapter = LLMAdapter(config)
        
        print("Loading model...")
        adapter.load(force_cpu=True)
        
        print("Creating SimulationConfigGenerator...")
        generator = SimulationConfigGenerator(adapter)
        print("✓ Generator created successfully")
        
        # Test simulator detection
        test_descriptions = [
            "quantum tunneling through barrier",
            "entangle two qubits",
            "harmonic oscillator ground state"
        ]
        
        for desc in test_descriptions:
            sim_type = generator._detect_simulator_type(desc)
            print(f"✓ '{desc}' → {sim_type}")
        
        # Test config generation (may use defaults if LLM fails)
        print("\nTesting config generation...")
        config = generator.generate_simulation_config(
            "Run quantum tunneling with high barrier"
        )
        print(f"✓ Generated config: {config}")
        
        adapter.unload()
        return True
    except Exception as e:
        print(f"✗ NL Interface test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quantization_config():
    """Test 9: BitsAndBytes Quantization Config"""
    print("\n" + "="*70)
    print("TEST 9: Quantization Configuration")
    print("="*70)
    
    try:
        from transformers import BitsAndBytesConfig
        import torch
        
        print("Creating 4-bit quantization config...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print("✓ BitsAndBytesConfig created successfully")
        print(f"✓ Quantization type: {bnb_config.bnb_4bit_quant_type}")
        print(f"✓ Compute dtype: {bnb_config.bnb_4bit_compute_dtype}")
        print(f"✓ Double quantization: {bnb_config.bnb_4bit_use_double_quant}")
        
        return True
    except Exception as e:
        print(f"✗ Quantization config test failed: {e}")
        print("Note: bitsandbytes may not be available on CPU-only systems")
        return False


def test_generation_config():
    """Test 10: Generation Configuration"""
    print("\n" + "="*70)
    print("TEST 10: Generation Configuration")
    print("="*70)
    
    try:
        from transformers import GenerationConfig
        
        print("Creating generation config...")
        gen_config = GenerationConfig(
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=50256  # GPT-2 EOS token
        )
        print("✓ GenerationConfig created successfully")
        print(f"✓ Max new tokens: {gen_config.max_new_tokens}")
        print(f"✓ Temperature: {gen_config.temperature}")
        print(f"✓ Top-p: {gen_config.top_p}")
        print(f"✓ Sampling: {gen_config.do_sample}")
        
        return True
    except Exception as e:
        print(f"✗ Generation config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all transformers integration tests"""
    print("\n" + "="*70)
    print("TRANSFORMERS LIBRARY INTEGRATION TEST SUITE")
    print("="*70)
    
    tests = [
        ("Import Test", test_transformers_import),
        ("PyTorch/CUDA Test", test_torch_cuda),
        ("LLMAdapter Init", test_llm_adapter_init),
        ("Tokenizer Loading", test_tokenizer_loading),
        ("Model Loading", test_model_loading),
        ("Text Generation", test_text_generation),
        ("LLMAdapter Workflow", test_llm_adapter_full),
        ("NL Interface", test_nl_interface),
        ("Quantization Config", test_quantization_config),
        ("Generation Config", test_generation_config),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! Transformers integration is working correctly.")
    elif passed >= total * 0.7:
        print("\n⚠ Most tests passed. Some optional features may not be available.")
    else:
        print("\n❌ Many tests failed. Check dependencies and installation.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
