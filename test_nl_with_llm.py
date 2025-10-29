#!/usr/bin/env python3
"""
Test NL Config Generation with Real LLM
Validates Task 16.1 with actual Qwen model
"""

import sys
sys.path.insert(0, '/home/windows/ravan-quantum-ml')

import logging
from gpu_accelerator import GPUAccelerator
from vram_manager import VRAMManager, WorkloadMode
from src.llm.llm_adapter import LLMAdapter, LLMConfig
from src.llm.nl_interface import SimulationConfigGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Test NL config generation with LLM"""
    print("="*70)
    print("NATURAL LANGUAGE CONFIG GENERATION - LLM TEST")
    print("="*70)
    
    # Initialize VRAM management
    logger.info("Initializing GPU and VRAM manager...")
    gpu = GPUAccelerator()
    vram_mgr = VRAMManager(gpu)
    
    # Check initial VRAM
    status = vram_mgr.get_vram_status()
    logger.info(f"Initial VRAM: {status['free_vram_gb']:.2f} GB free")
    
    # Create LLM adapter (use Qwen-7B for testing, or Qwen-30B if enough VRAM)
    free_vram = status['free_vram_gb']
    
    if free_vram >= 22:
        model_name = "Qwen/Qwen-30B-Chat"
        logger.info("Using Qwen-30B (sufficient VRAM)")
    elif free_vram >= 10:
        model_name = "Qwen/Qwen-14B-Chat"
        logger.info("Using Qwen-14B (moderate VRAM)")
    else:
        model_name = "Qwen/Qwen-7B-Chat"
        logger.info("Using Qwen-7B (limited VRAM)")
    
    config = LLMConfig(
        model_name=model_name,
        load_in_4bit=True,
        max_new_tokens=300,
        temperature=0.3  # Lower temperature for more consistent outputs
    )
    
    llm = LLMAdapter(config)
    
    # Load LLM through VRAM manager
    logger.info("Loading LLM...")
    if not vram_mgr.load_llm(llm):
        logger.error("Failed to load LLM")
        return False
    
    logger.info("✓ LLM loaded successfully")
    
    # Create config generator
    generator = SimulationConfigGenerator(llm)
    
    # Test cases matching the requirements
    test_cases = [
        {
            'description': "Simulate quantum tunneling through a high barrier with narrow wavepacket",
            'expected_simulator': 'schrodinger',
            'expected_params': ['V0', 'barrier_width', 'k0', 'sigma', 'x0']
        },
        {
            'description': "Create a Bell state with 2 qubits and measure 5000 times",
            'expected_simulator': 'quantum_circuit',
            'expected_params': ['n_qubits', 'shots', 'gate_sequence']
        },
        {
            'description': "Simulate harmonic oscillator in first excited state with 15 basis functions",
            'expected_simulator': 'harmonic_oscillator',
            'expected_params': ['oscillator_length', 'basis_size', 'initial_n']
        }
    ]
    
    print("\n" + "="*70)
    print("TESTING CONFIG GENERATION")
    print("="*70)
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}/3")
        print(f"{'='*70}")
        print(f"\nDescription: {test_case['description']}")
        
        try:
            # Generate config
            logger.info(f"Generating config for test {i}...")
            config = generator.generate_simulation_config(test_case['description'])
            
            # Validate results
            print(f"\n✓ Config generated successfully")
            print(f"\nSimulator: {config['simulator']}")
            print(f"Expected: {test_case['expected_simulator']}")
            
            simulator_match = config['simulator'] == test_case['expected_simulator']
            if simulator_match:
                print("✓ Simulator type correct")
            else:
                print("✗ Simulator type mismatch")
            
            print(f"\nParameters:")
            for param, value in config['parameters'].items():
                print(f"  {param}: {value}")
            
            # Check all expected parameters present
            params_present = all(p in config['parameters'] for p in test_case['expected_params'])
            if params_present:
                print(f"\n✓ All expected parameters present")
            else:
                print(f"\n✗ Missing parameters")
            
            # Generate explanation
            explanation = generator.explain_config(config)
            print(f"\nExplanation: {explanation}")
            
            # Validate parameter ranges
            validated = generator._validate_parameters(
                config['parameters'],
                config['simulator']
            )
            
            ranges_valid = validated == config['parameters']
            if ranges_valid:
                print("✓ All parameters within valid ranges")
            else:
                print("⚠ Some parameters were clamped to valid ranges")
            
            # Record result
            test_passed = simulator_match and params_present
            results.append(test_passed)
            
            if test_passed:
                print(f"\n✓ TEST {i} PASSED")
            else:
                print(f"\n✗ TEST {i} FAILED")
            
        except Exception as e:
            logger.error(f"Error in test {i}: {e}", exc_info=True)
            print(f"\n✗ TEST {i} FAILED with exception: {e}")
            results.append(False)
    
    # Unload LLM
    print("\n" + "="*70)
    logger.info("Unloading LLM...")
    vram_mgr.unload_llm()
    logger.info("✓ LLM unloaded")
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    for i, result in enumerate(results, 1):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  Test {i}: {status}")
    
    if passed == total:
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED")
        print("="*70)
        print("\nNatural Language Config Generation is working correctly!")
        print("The system can convert NL descriptions to simulation parameters.")
        return True
    else:
        print("\n" + "="*70)
        print(f"⚠ {total - passed} TEST(S) FAILED")
        print("="*70)
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
