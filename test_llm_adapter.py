#!/usr/bin/env python3
"""
Test script for LLMAdapter
Task 15.2: Test LLM adapter functionality
"""

import sys
sys.path.insert(0, '/home/windows/ravan-quantum-ml')

from src.llm.llm_adapter import LLMAdapter, LLMConfig, create_llm_adapter
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def test_vram_check():
    """Test VRAM checking"""
    print("\n" + "="*70)
    print("TEST 1: VRAM Checking")
    print("="*70)
    
    llm = LLMAdapter()
    is_available, free_vram = llm.check_vram_available(required_gb=22.0)
    
    print(f"\nVRAM Check:")
    print(f"  Available: {is_available}")
    print(f"  Free VRAM: {free_vram:.2f} GB")
    
    if is_available:
        print("✓ Sufficient VRAM for LLM loading")
    else:
        print("✗ Insufficient VRAM - will use CPU fallback")
    
    return is_available


def test_model_loading(use_gpu=True):
    """Test model loading"""
    print("\n" + "="*70)
    print("TEST 2: Model Loading")
    print("="*70)
    
    # Use smaller model for testing if specified
    config = LLMConfig(
        model_name="Qwen/Qwen-7B-Chat",  # Smaller for testing
        load_in_4bit=True
    )
    
    llm = LLMAdapter(config)
    
    print(f"\nLoading model: {config.model_name}")
    print("(Using Qwen-7B for faster testing)")
    
    success = llm.load(force_cpu=not use_gpu)
    
    if success:
        print("✓ Model loaded successfully")
        
        # Get model info
        info = llm.get_model_info()
        print("\nModel Info:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Unload
        llm.unload()
        print("\n✓ Model unloaded successfully")
        
        return True
    else:
        print("✗ Model loading failed")
        return False


def test_query():
    """Test querying the model"""
    print("\n" + "="*70)
    print("TEST 3: Query Functionality")
    print("="*70)
    
    # Use smaller model for testing
    config = LLMConfig(
        model_name="Qwen/Qwen-7B-Chat",
        load_in_4bit=True,
        max_new_tokens=100
    )
    
    llm = LLMAdapter(config)
    
    print("\nLoading model...")
    if not llm.load():
        print("✗ Failed to load model")
        return False
    
    # Test query
    print("\nTesting query...")
    prompt = "What is quantum tunneling? Explain in 2 sentences."
    
    print(f"\nPrompt: {prompt}")
    print("\nGenerating response...")
    
    try:
        response = llm.query(prompt, max_new_tokens=100)
        
        print(f"\nResponse:")
        print(f"{response}")
        print("\n✓ Query successful")
        
        success = True
    except Exception as e:
        print(f"\n✗ Query failed: {e}")
        success = False
    finally:
        llm.unload()
    
    return success


def test_context_manager():
    """Test context manager usage"""
    print("\n" + "="*70)
    print("TEST 4: Context Manager")
    print("="*70)
    
    config = LLMConfig(
        model_name="Qwen/Qwen-7B-Chat",
        load_in_4bit=True
    )
    
    print("\nUsing context manager...")
    
    try:
        with LLMAdapter(config) as llm:
            print("✓ Model loaded via context manager")
            
            # Quick query
            response = llm.query("What is 2+2?", max_new_tokens=20)
            print(f"\nTest query response: {response[:100]}...")
        
        print("✓ Model automatically unloaded")
        return True
        
    except Exception as e:
        print(f"✗ Context manager test failed: {e}")
        return False


def test_batch_query():
    """Test batch querying"""
    print("\n" + "="*70)
    print("TEST 5: Batch Query")
    print("="*70)
    
    config = LLMConfig(
        model_name="Qwen/Qwen-7B-Chat",
        load_in_4bit=True,
        max_new_tokens=50
    )
    
    llm = LLMAdapter(config)
    
    print("\nLoading model...")
    if not llm.load():
        print("✗ Failed to load model")
        return False
    
    # Test batch queries
    prompts = [
        "What is quantum entanglement?",
        "What is superposition?",
        "What is the Schrödinger equation?"
    ]
    
    print(f"\nTesting batch query with {len(prompts)} prompts...")
    
    try:
        responses = llm.batch_query(prompts, max_new_tokens=50)
        
        print("\n✓ Batch query successful")
        print(f"\nReceived {len(responses)} responses:")
        
        for i, (prompt, response) in enumerate(zip(prompts, responses), 1):
            print(f"\n{i}. {prompt}")
            print(f"   → {response[:100]}...")
        
        success = True
    except Exception as e:
        print(f"\n✗ Batch query failed: {e}")
        success = False
    finally:
        llm.unload()
    
    return success


def main():
    """Run all tests"""
    print("="*70)
    print("LLM ADAPTER TEST SUITE")
    print("="*70)
    print("\nNote: Using Qwen-7B for faster testing")
    print("      (Replace with Qwen-30B for production)")
    
    results = {}
    
    # Test 1: VRAM check
    vram_available = test_vram_check()
    results['vram_check'] = vram_available
    
    # Test 2: Model loading
    results['model_loading'] = test_model_loading(use_gpu=vram_available)
    
    # Test 3: Query (skip if loading failed)
    if results['model_loading']:
        results['query'] = test_query()
    else:
        print("\nSkipping query test (model loading failed)")
        results['query'] = False
    
    # Test 4: Context manager
    results['context_manager'] = test_context_manager()
    
    # Test 5: Batch query
    if results['model_loading']:
        results['batch_query'] = test_batch_query()
    else:
        print("\nSkipping batch query test (model loading failed)")
        results['batch_query'] = False
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED")
    else:
        print(f"\n⚠ {total - passed} test(s) failed")


if __name__ == "__main__":
    main()
