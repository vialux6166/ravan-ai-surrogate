#!/usr/bin/env python3
"""
Download and Verify Qwen 30B Model
Task 15.1: Download Qwen 30B model weights locally

This script downloads the Qwen model and verifies it can load in 4-bit quantization.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import gc


def check_disk_space():
    """Check available disk space"""
    import shutil
    
    stats = shutil.disk_usage(os.path.expanduser("~"))
    free_gb = stats.free / (1024**3)
    
    print(f"Available disk space: {free_gb:.2f} GB")
    
    if free_gb < 70:
        print("⚠ Warning: Less than 70GB free. Qwen 30B requires ~60GB.")
        response = input("Continue anyway? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborting download.")
            sys.exit(0)
    else:
        print("✓ Sufficient disk space available")
    
    return free_gb


def check_gpu_vram():
    """Check available GPU VRAM"""
    if not torch.cuda.is_available():
        print("⚠ Warning: CUDA not available. Will use CPU (very slow).")
        return 0
    
    device = torch.device('cuda:0')
    total_vram = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    
    # Clear cache to get accurate free memory
    torch.cuda.empty_cache()
    gc.collect()
    
    free_vram = (torch.cuda.get_device_properties(device).total_memory - 
                 torch.cuda.memory_allocated(device)) / (1024**3)
    
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Total VRAM: {total_vram:.2f} GB")
    print(f"Free VRAM: {free_vram:.2f} GB")
    
    if free_vram < 22:
        print(f"⚠ Warning: Less than 22GB free VRAM. May not be able to load model.")
    else:
        print("✓ Sufficient VRAM for 4-bit quantization")
    
    return free_vram


def download_model(model_name="Qwen/Qwen-30B-Chat", cache_dir=None):
    """
    Download Qwen model
    
    Args:
        model_name: HuggingFace model identifier
        cache_dir: Directory to cache model (default: ~/.cache/huggingface)
    """
    print("\n" + "="*70)
    print("DOWNLOADING QWEN MODEL")
    print("="*70)
    
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/huggingface")
    
    print(f"\nModel: {model_name}")
    print(f"Cache directory: {cache_dir}")
    print("\nThis will download ~60GB of data. This may take 30-60 minutes.")
    print("The download will resume if interrupted.")
    
    response = input("\nProceed with download? (yes/no): ")
    if response.lower() != 'yes':
        print("Download cancelled.")
        return None
    
    print("\nDownloading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        print("✓ Tokenizer downloaded successfully")
    except Exception as e:
        print(f"✗ Error downloading tokenizer: {e}")
        return None
    
    print("\nDownloading model weights...")
    print("(This is the large download - ~60GB)")
    
    try:
        # Download model without loading (saves memory)
        from transformers import AutoConfig
        
        config = AutoConfig.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        print("✓ Model config downloaded")
        
        # This downloads the weights but doesn't load them into memory
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            device_map=None,  # Don't load to device yet
            low_cpu_mem_usage=True
        )
        print("✓ Model weights downloaded successfully")
        
        # Clean up
        del model
        gc.collect()
        
        return tokenizer
        
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        return None


def verify_4bit_loading(model_name="Qwen/Qwen-30B-Chat", cache_dir=None):
    """
    Verify model can load in 4-bit quantization
    
    Args:
        model_name: HuggingFace model identifier
        cache_dir: Directory where model is cached
    """
    print("\n" + "="*70)
    print("VERIFYING 4-BIT QUANTIZATION")
    print("="*70)
    
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/huggingface")
    
    if not torch.cuda.is_available():
        print("⚠ CUDA not available. Skipping 4-bit verification.")
        print("  4-bit quantization requires GPU.")
        return False
    
    print("\nConfiguring 4-bit quantization...")
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    print("✓ Quantization config created")
    print(f"  - Quantization: 4-bit NF4")
    print(f"  - Compute dtype: float16")
    print(f"  - Double quantization: enabled")
    
    print("\nLoading model in 4-bit mode...")
    print("(This may take 2-5 minutes)")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        # Load model in 4-bit
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("✓ Model loaded successfully in 4-bit mode")
        
        # Check VRAM usage
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated(0) / (1024**3)
            print(f"  VRAM used: {vram_used:.2f} GB")
        
        # Test inference
        print("\nTesting inference...")
        test_prompt = "What is quantum tunneling?"
        
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("✓ Inference test successful")
        print(f"\nTest prompt: {test_prompt}")
        print(f"Response: {response[:200]}...")
        
        # Clean up
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        
        print("\n✓ 4-bit quantization verification complete")
        return True
        
    except Exception as e:
        print(f"✗ Error loading model in 4-bit: {e}")
        return False


def main():
    """Main execution"""
    print("="*70)
    print("QWEN 30B MODEL SETUP")
    print("="*70)
    
    # Step 1: Check system resources
    print("\nStep 1: Checking system resources...")
    free_disk = check_disk_space()
    free_vram = check_gpu_vram()
    
    # Step 2: Download model
    print("\nStep 2: Download model")
    model_name = "Qwen/Qwen-30B-Chat"
    
    # Check if already downloaded
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    model_path = os.path.join(cache_dir, "models--Qwen--Qwen-30B-Chat")
    
    if os.path.exists(model_path):
        print(f"✓ Model already downloaded at: {model_path}")
        response = input("Re-download? (yes/no): ")
        if response.lower() != 'yes':
            print("Skipping download.")
        else:
            tokenizer = download_model(model_name, cache_dir)
    else:
        tokenizer = download_model(model_name, cache_dir)
    
    # Step 3: Verify 4-bit loading
    print("\nStep 3: Verify 4-bit quantization")
    response = input("Test 4-bit loading? (requires 22GB VRAM) (yes/no): ")
    
    if response.lower() == 'yes':
        success = verify_4bit_loading(model_name, cache_dir)
        
        if success:
            print("\n" + "="*70)
            print("✓ ALL CHECKS PASSED")
            print("="*70)
            print("\nQwen 30B is ready for use!")
            print(f"Model location: {cache_dir}")
            print("\nNext steps:")
            print("1. Implement LLMAdapter class (Task 15.2)")
            print("2. Integrate with VRAMManager (Task 15.3)")
        else:
            print("\n" + "="*70)
            print("⚠ VERIFICATION INCOMPLETE")
            print("="*70)
            print("\nModel downloaded but 4-bit loading failed.")
            print("Check VRAM availability and CUDA installation.")
    else:
        print("\nSkipping 4-bit verification.")
        print("You can test it later with:")
        print("  python scripts/download_qwen_model.py")


if __name__ == "__main__":
    main()
