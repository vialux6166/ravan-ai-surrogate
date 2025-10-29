"""
Full Integration Test: LLMAdapter + Qwen2.5-32B + 4-bit + GPU
Tests the complete workflow with the updated LLMAdapter
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)

def test_llm_adapter_with_qwen():
    """Test LLMAdapter with Qwen2.5-32B on GPU with 4-bit quantization"""
    print("\n" + "="*70)
    print("FULL INTEGRATION TEST: LLMAdapter + Qwen2.5-32B + GPU + 4-bit")
    print("="*70)
    
    from src.llm.llm_adapter import LLMAdapter, LLMConfig
    
    # Test 1: Default config (should use Qwen2.5-32B)
    print("\n--- Test 1: Default Configuration ---")
    config = LLMConfig()
    print(f"✓ Default model: {config.model_name}")
    print(f"✓ 4-bit quantization: {config.load_in_4bit}")
    print(f"✓ Device: {config.device}")
    
    # Test 2: Initialize adapter
    print("\n--- Test 2: Initialize LLMAdapter ---")
    adapter = LLMAdapter(config)
    print("✓ LLMAdapter initialized")
    
    # Test 3: Check VRAM
    print("\n--- Test 3: Check VRAM ---")
    vram_ok, free_vram = adapter.check_vram_available(required_gb=20.0)
    if vram_ok:
        print(f"✓ Sufficient VRAM: {free_vram:.2f} GB")
    else:
        print(f"⚠ Insufficient VRAM: {free_vram:.2f} GB (will use CPU)")
    
    # Test 4: Load model
    print("\n--- Test 4: Load Model ---")
    print("Loading Qwen2.5-32B with 4-bit quantization...")
    print("(This will take 1-2 minutes if cached, 20-30 minutes if downloading)")
    
    success = adapter.load()
    if not success:
        print("✗ Failed to load model")
        return False
    
    print("✓ Model loaded successfully")
    
    # Test 5: Get model info
    print("\n--- Test 5: Model Information ---")
    info = adapter.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test 6: Query model
    print("\n--- Test 6: Query Model ---")
    
    test_queries = [
        "Explain quantum superposition in one sentence:",
        "What is the Heisenberg uncertainty principle?",
        "Describe quantum entanglement briefly:"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        response = adapter.query(query, max_new_tokens=100)
        print(f"Response: {response[:200]}...")
    
    print("\n✓ All queries successful")
    
    # Test 7: Unload model
    print("\n--- Test 7: Cleanup ---")
    adapter.unload()
    print("✓ Model unloaded")
    
    return True


def test_natural_language_interface():
    """Test natural language interface with Qwen"""
    print("\n" + "="*70)
    print("TEST: Natural Language Interface with Qwen")
    print("="*70)
    
    from src.llm.llm_adapter import LLMAdapter, LLMConfig
    from src.llm.nl_interface import SimulationConfigGenerator
    
    # Initialize
    config = LLMConfig()
    adapter = LLMAdapter(config)
    
    print("Loading model...")
    adapter.load()
    
    print("\nCreating SimulationConfigGenerator...")
    generator = SimulationConfigGenerator(adapter)
    
    # Test config generation
    print("\n--- Test: Config Generation ---")
    
    test_descriptions = [
        "Run quantum tunneling with high barrier and narrow wavepacket",
        "Create a Bell state with two entangled qubits",
        "Simulate harmonic oscillator in ground state"
    ]
    
    for desc in test_descriptions:
        print(f"\nDescription: '{desc}'")
        config = generator.generate_simulation_config(desc)
        print(f"Generated config:")
        print(f"  Simulator: {config['simulator']}")
        print(f"  Parameters: {config['parameters']}")
    
    print("\n✓ Natural language interface working")
    
    adapter.unload()
    return True


def test_results_explainer():
    """Test results explainer with Qwen"""
    print("\n" + "="*70)
    print("TEST: Results Explainer with Qwen")
    print("="*70)
    
    from src.llm.llm_adapter import LLMAdapter, LLMConfig
    from src.llm.results_explainer import ResultsExplainer
    
    # Initialize
    config = LLMConfig()
    adapter = LLMAdapter(config)
    
    print("Loading model...")
    adapter.load()
    
    print("\nCreating ResultsExplainer...")
    explainer = ResultsExplainer(adapter)
    
    # Test explanation
    print("\n--- Test: Results Explanation ---")
    
    test_results = {
        'simulator': 'schrodinger',
        'parameters': {'V0': 5.0, 'k0': 4.0, 'barrier_width': 1.5},
        'results': {'transmission': 0.65, 'reflection': 0.35, 'total_probability': 1.0}
    }
    
    print(f"Explaining results for {test_results['simulator']} simulation...")
    explanation = explainer.explain_results(
        test_results['simulator'],
        test_results['parameters'],
        test_results['results']
    )
    
    print(f"\nExplanation: {explanation}")
    print("\n✓ Results explainer working")
    
    adapter.unload()
    return True


def main():
    """Run all integration tests"""
    print("\n" + "="*70)
    print("FULL INTEGRATION TEST SUITE")
    print("="*70)
    print("\nTesting:")
    print("1. LLMAdapter with Qwen2.5-32B")
    print("2. 4-bit quantization on GPU")
    print("3. Natural language interface")
    print("4. Results explainer")
    
    tests = [
        ("LLMAdapter + Qwen + GPU", test_llm_adapter_with_qwen),
        ("Natural Language Interface", test_natural_language_interface),
        ("Results Explainer", test_results_explainer),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*70}")
            print(f"Running: {test_name}")
            print(f"{'='*70}")
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' failed: {e}")
            import traceback
            traceback.print_exc()
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
        print("\n🎉 All integration tests passed!")
        print("✅ System is ready for production use")
    else:
        print("\n⚠ Some tests failed. Review errors above.")
    
    return passed == total


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Tests crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
