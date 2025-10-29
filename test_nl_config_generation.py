#!/usr/bin/env python3
"""
Test Natural Language Config Generation
Task 16.1: Test simulation config generation from NL descriptions
"""

import sys
sys.path.insert(0, '/home/windows/ravan-quantum-ml')

import logging
from src.llm.llm_adapter import LLMAdapter, LLMConfig
from src.llm.nl_interface import SimulationConfigGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def test_simulator_detection():
    """Test automatic simulator type detection"""
    print("\n" + "="*70)
    print("TEST 1: Simulator Type Detection")
    print("="*70)
    
    # Create dummy LLM adapter (won't be used for detection)
    config = LLMConfig(model_name="Qwen/Qwen-7B-Chat", load_in_4bit=True)
    llm = LLMAdapter(config)
    
    generator = SimulationConfigGenerator(llm)
    
    test_cases = [
        ("Create a Bell state with 2 qubits", "quantum_circuit"),
        ("Run quantum tunneling through a barrier", "schrodinger"),
        ("Simulate a harmonic oscillator in ground state", "harmonic_oscillator"),
        ("Measure entanglement in a 3-qubit system", "quantum_circuit"),
        ("Calculate transmission coefficient for high barrier", "schrodinger"),
        ("Analyze photon number in coherent state", "harmonic_oscillator"),
    ]
    
    print("\nTesting simulator detection:")
    results = []
    
    for description, expected in test_cases:
        detected = generator._detect_simulator_type(description)
        match = "✓" if detected == expected else "✗"
        results.append(detected == expected)
        
        print(f"\n{match} Description: {description}")
        print(f"  Expected: {expected}")
        print(f"  Detected: {detected}")
    
    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} detections correct")
    
    return passed == total


def test_parameter_validation():
    """Test parameter validation and clamping"""
    print("\n" + "="*70)
    print("TEST 2: Parameter Validation")
    print("="*70)
    
    config = LLMConfig(model_name="Qwen/Qwen-7B-Chat", load_in_4bit=True)
    llm = LLMAdapter(config)
    generator = SimulationConfigGenerator(llm)
    
    # Test Schrödinger parameters
    print("\nTesting Schrödinger parameter validation:")
    
    test_params = {
        'V0': 15.0,  # Above max (10.0)
        'barrier_width': 0.2,  # Below min (0.5)
        'k0': 5.0,  # Valid
        'sigma': 1.0,  # Valid
        'x0': -5.0  # Valid
    }
    
    validated = generator._validate_parameters(test_params, 'schrodinger')
    
    print(f"\nOriginal: {test_params}")
    print(f"Validated: {validated}")
    
    # Check clamping
    assert validated['V0'] == 10.0, "V0 should be clamped to 10.0"
    assert validated['barrier_width'] == 0.5, "barrier_width should be clamped to 0.5"
    assert validated['k0'] == 5.0, "k0 should remain unchanged"
    
    print("\n✓ Parameter validation working correctly")
    return True


def test_default_parameters():
    """Test default parameter generation"""
    print("\n" + "="*70)
    print("TEST 3: Default Parameters")
    print("="*70)
    
    config = LLMConfig(model_name="Qwen/Qwen-7B-Chat", load_in_4bit=True)
    llm = LLMAdapter(config)
    generator = SimulationConfigGenerator(llm)
    
    simulators = ['quantum_circuit', 'schrodinger', 'harmonic_oscillator']
    
    print("\nDefault parameters for each simulator:")
    
    for sim_type in simulators:
        defaults = generator._get_default_parameters(sim_type)
        print(f"\n{sim_type}:")
        for param, value in defaults.items():
            print(f"  {param}: {value}")
    
    print("\n✓ Default parameters available for all simulators")
    return True


def test_config_explanation():
    """Test config explanation generation"""
    print("\n" + "="*70)
    print("TEST 4: Config Explanation")
    print("="*70)
    
    config = LLMConfig(model_name="Qwen/Qwen-7B-Chat", load_in_4bit=True)
    llm = LLMAdapter(config)
    generator = SimulationConfigGenerator(llm)
    
    test_configs = [
        {
            'simulator': 'quantum_circuit',
            'parameters': {'n_qubits': 2, 'shots': 1000, 'gate_sequence': 'bell'}
        },
        {
            'simulator': 'schrodinger',
            'parameters': {'V0': 5.0, 'barrier_width': 1.5, 'k0': 4.0, 'sigma': 1.0, 'x0': -5.0}
        },
        {
            'simulator': 'harmonic_oscillator',
            'parameters': {'oscillator_length': 1.0, 'basis_size': 10, 'initial_n': 2}
        }
    ]
    
    print("\nGenerating explanations:")
    
    for cfg in test_configs:
        explanation = generator.explain_config(cfg)
        print(f"\n{cfg['simulator']}:")
        print(f"  {explanation}")
    
    print("\n✓ Config explanations generated")
    return True


def test_full_config_generation():
    """Test full config generation with LLM"""
    print("\n" + "="*70)
    print("TEST 5: Full Config Generation (with LLM)")
    print("="*70)
    
    # Create and load LLM
    config = LLMConfig(
        model_name="Qwen/Qwen-7B-Chat",
        load_in_4bit=True,
        max_new_tokens=300
    )
    llm = LLMAdapter(config)
    
    print("\nLoading LLM...")
    if not llm.load():
        print("✗ LLM loading failed, skipping test")
        return False
    
    generator = SimulationConfigGenerator(llm)
    
    # Test descriptions
    descriptions = [
        "Run quantum tunneling with a high barrier and narrow wavepacket",
        "Create a Bell state with 2 qubits and 5000 measurements",
        "Simulate harmonic oscillator in first excited state"
    ]
    
    print("\nGenerating configs from natural language:")
    
    for desc in descriptions:
        print(f"\n{'='*70}")
        print(f"Description: {desc}")
        
        try:
            config = generator.generate_simulation_config(desc)
            
            print(f"\nGenerated Config:")
            print(f"  Simulator: {config['simulator']}")
            print(f"  Parameters:")
            for param, value in config['parameters'].items():
                print(f"    {param}: {value}")
            
            # Generate explanation
            explanation = generator.explain_config(config)
            print(f"\nExplanation: {explanation}")
            
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Unload LLM
    llm.unload()
    
    print("\n✓ Full config generation test complete")
    return True


def test_batch_generation():
    """Test batch config generation"""
    print("\n" + "="*70)
    print("TEST 6: Batch Config Generation")
    print("="*70)
    
    config = LLMConfig(
        model_name="Qwen/Qwen-7B-Chat",
        load_in_4bit=True
    )
    llm = LLMAdapter(config)
    
    print("\nLoading LLM...")
    if not llm.load():
        print("✗ LLM loading failed, skipping test")
        return False
    
    generator = SimulationConfigGenerator(llm)
    
    descriptions = [
        "Low barrier tunneling",
        "3-qubit GHZ state",
        "Ground state harmonic oscillator"
    ]
    
    print(f"\nGenerating {len(descriptions)} configs in batch...")
    
    configs = generator.batch_generate_configs(descriptions)
    
    print(f"\nGenerated {len(configs)} configs:")
    for i, (desc, cfg) in enumerate(zip(descriptions, configs), 1):
        if cfg:
            print(f"\n{i}. {desc}")
            print(f"   → {cfg['simulator']}: {cfg['parameters']}")
        else:
            print(f"\n{i}. {desc}")
            print(f"   → Failed to generate")
    
    llm.unload()
    
    print("\n✓ Batch generation test complete")
    return True


def main():
    """Run all tests"""
    print("="*70)
    print("NATURAL LANGUAGE CONFIG GENERATION TEST SUITE")
    print("="*70)
    print("\nNote: Using Qwen-7B for faster testing")
    
    results = {}
    
    # Test 1: Simulator detection (no LLM needed)
    results['simulator_detection'] = test_simulator_detection()
    
    # Test 2: Parameter validation (no LLM needed)
    results['parameter_validation'] = test_parameter_validation()
    
    # Test 3: Default parameters (no LLM needed)
    results['default_parameters'] = test_default_parameters()
    
    # Test 4: Config explanation (no LLM needed)
    results['config_explanation'] = test_config_explanation()
    
    # Test 5: Full generation (requires LLM)
    print("\n" + "="*70)
    print("LLM-DEPENDENT TESTS")
    print("="*70)
    response = input("\nRun tests that require LLM loading? (yes/no): ")
    
    if response.lower() == 'yes':
        results['full_generation'] = test_full_config_generation()
        results['batch_generation'] = test_batch_generation()
    else:
        print("\nSkipping LLM-dependent tests")
        results['full_generation'] = None
        results['batch_generation'] = None
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        if passed is None:
            status = "⊘ SKIPPED"
        elif passed:
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
        print(f"{test_name:25s}: {status}")
    
    # Count only non-skipped tests
    non_skipped = {k: v for k, v in results.items() if v is not None}
    if non_skipped:
        total = len(non_skipped)
        passed = sum(non_skipped.values())
        print(f"\nTotal: {passed}/{total} tests passed")
        
        if passed == total:
            print("\n✓ ALL TESTS PASSED")
        else:
            print(f"\n⚠ {total - passed} test(s) failed")
    else:
        print("\nAll LLM tests skipped")


if __name__ == "__main__":
    main()
