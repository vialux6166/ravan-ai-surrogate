#!/usr/bin/env python3
"""
Test VRAM Manager + LLM Integration
Task 15.3: Test integrated VRAM management with LLM
"""

import sys
sys.path.insert(0, '/home/windows/ravan-quantum-ml')

import logging
import time
from gpu_accelerator import GPUAccelerator
from vram_manager import VRAMManager, WorkloadMode
from src.llm.llm_adapter import LLMAdapter, LLMConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def test_vram_status():
    """Test VRAM status reporting"""
    print("\n" + "="*70)
    print("TEST 1: VRAM Status Reporting")
    print("="*70)
    
    gpu = GPUAccelerator()
    vram_mgr = VRAMManager(gpu)
    
    status = vram_mgr.get_vram_status()
    
    print("\nVRAM Status:")
    for key, value in status.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    return True


def test_mode_transitions():
    """Test mode transitions without LLM"""
    print("\n" + "="*70)
    print("TEST 2: Mode Transitions (No LLM)")
    print("="*70)
    
    gpu = GPUAccelerator()
    vram_mgr = VRAMManager(gpu)
    
    # Test SIMULATION mode
    print("\nRequesting SIMULATION mode...")
    vram_mgr.request_mode(WorkloadMode.SIMULATION)
    assert vram_mgr.get_current_mode() == WorkloadMode.SIMULATION
    print("✓ SIMULATION mode active")
    
    # Test TRAINING mode
    print("\nRequesting TRAINING mode...")
    vram_mgr.request_mode(WorkloadMode.TRAINING)
    assert vram_mgr.get_current_mode() == WorkloadMode.TRAINING
    print("✓ TRAINING mode active")
    
    # Test INFERENCE mode
    print("\nRequesting INFERENCE mode...")
    vram_mgr.request_mode(WorkloadMode.INFERENCE)
    assert vram_mgr.get_current_mode() == WorkloadMode.INFERENCE
    print("✓ INFERENCE mode active")
    
    print("\n✓ All mode transitions successful")
    return True


def test_llm_loading():
    """Test LLM loading through VRAM manager"""
    print("\n" + "="*70)
    print("TEST 3: LLM Loading via VRAM Manager")
    print("="*70)
    
    gpu = GPUAccelerator()
    vram_mgr = VRAMManager(gpu)
    
    # Create LLM adapter (use smaller model for testing)
    config = LLMConfig(
        model_name="Qwen/Qwen-7B-Chat",  # Smaller for testing
        load_in_4bit=True
    )
    llm = LLMAdapter(config)
    
    print("\nLoading LLM through VRAM manager...")
    success = vram_mgr.load_llm(llm)
    
    if success:
        print("✓ LLM loaded successfully")
        
        # Check mode
        assert vram_mgr.is_llm_mode()
        print("✓ LLM mode active")
        
        # Check VRAM status
        status = vram_mgr.get_vram_status()
        print(f"\nVRAM after LLM load:")
        print(f"  Allocated: {status['allocated_vram_gb']:.2f} GB")
        print(f"  Free: {status['free_vram_gb']:.2f} GB")
        
        # Unload
        print("\nUnloading LLM...")
        vram_mgr.unload_llm()
        print("✓ LLM unloaded")
        
        # Check VRAM freed
        status = vram_mgr.get_vram_status()
        print(f"\nVRAM after unload:")
        print(f"  Allocated: {status['allocated_vram_gb']:.2f} GB")
        print(f"  Free: {status['free_vram_gb']:.2f} GB")
        
        return True
    else:
        print("✗ LLM loading failed")
        return False


def test_mutual_exclusion():
    """Test mutual exclusion between LLM and other modes"""
    print("\n" + "="*70)
    print("TEST 4: Mutual Exclusion")
    print("="*70)
    
    gpu = GPUAccelerator()
    vram_mgr = VRAMManager(gpu)
    
    # Load LLM
    config = LLMConfig(
        model_name="Qwen/Qwen-7B-Chat",
        load_in_4bit=True
    )
    llm = LLMAdapter(config)
    
    print("\nLoading LLM...")
    vram_mgr.load_llm(llm)
    print("✓ LLM loaded")
    
    # Try to switch to SIMULATION (should fail)
    print("\nAttempting to switch to SIMULATION mode (should fail)...")
    try:
        vram_mgr.request_mode(WorkloadMode.SIMULATION)
        print("✗ Should have raised RuntimeError")
        return False
    except RuntimeError as e:
        print(f"✓ Correctly blocked: {e}")
    
    # Unload LLM
    print("\nUnloading LLM...")
    vram_mgr.unload_llm()
    print("✓ LLM unloaded")
    
    # Now SIMULATION should work
    print("\nAttempting SIMULATION mode after unload...")
    vram_mgr.request_mode(WorkloadMode.SIMULATION)
    print("✓ SIMULATION mode active")
    
    print("\n✓ Mutual exclusion working correctly")
    return True


def test_llm_query_integration():
    """Test querying LLM through VRAM manager"""
    print("\n" + "="*70)
    print("TEST 5: LLM Query Integration")
    print("="*70)
    
    gpu = GPUAccelerator()
    vram_mgr = VRAMManager(gpu)
    
    # Load LLM
    config = LLMConfig(
        model_name="Qwen/Qwen-7B-Chat",
        load_in_4bit=True,
        max_new_tokens=50
    )
    llm = LLMAdapter(config)
    
    print("\nLoading LLM...")
    if not vram_mgr.load_llm(llm):
        print("✗ LLM loading failed")
        return False
    
    # Query through VRAM manager
    print("\nQuerying LLM...")
    prompt = "What is quantum entanglement? Answer in one sentence."
    
    try:
        response = vram_mgr.query_llm(prompt, max_new_tokens=50)
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response[:200]}...")
        print("\n✓ Query successful")
        success = True
    except Exception as e:
        print(f"\n✗ Query failed: {e}")
        success = False
    finally:
        vram_mgr.unload_llm()
    
    return success


def test_workflow_simulation_to_llm():
    """Test realistic workflow: simulation → LLM → simulation"""
    print("\n" + "="*70)
    print("TEST 6: Workflow - Simulation → LLM → Simulation")
    print("="*70)
    
    gpu = GPUAccelerator()
    vram_mgr = VRAMManager(gpu)
    
    # Step 1: Run simulation
    print("\nStep 1: SIMULATION mode")
    vram_mgr.request_mode(WorkloadMode.SIMULATION)
    print("✓ Running quantum simulation...")
    time.sleep(0.5)  # Simulate work
    
    status = vram_mgr.get_vram_status()
    print(f"  VRAM used: {status['allocated_vram_gb']:.2f} GB")
    
    # Step 2: Switch to LLM
    print("\nStep 2: Switching to LLM mode")
    config = LLMConfig(
        model_name="Qwen/Qwen-7B-Chat",
        load_in_4bit=True
    )
    llm = LLMAdapter(config)
    
    vram_mgr.load_llm(llm)
    print("✓ LLM loaded")
    
    status = vram_mgr.get_vram_status()
    print(f"  VRAM used: {status['allocated_vram_gb']:.2f} GB")
    
    # Query LLM
    response = vram_mgr.query_llm("Explain quantum tunneling briefly.", max_new_tokens=30)
    print(f"  LLM response: {response[:100]}...")
    
    # Step 3: Switch back to simulation
    print("\nStep 3: Switching back to SIMULATION mode")
    vram_mgr.unload_llm()
    vram_mgr.request_mode(WorkloadMode.SIMULATION)
    print("✓ Back to simulation mode")
    
    status = vram_mgr.get_vram_status()
    print(f"  VRAM used: {status['allocated_vram_gb']:.2f} GB")
    
    print("\n✓ Workflow completed successfully")
    return True


def main():
    """Run all integration tests"""
    print("="*70)
    print("VRAM MANAGER + LLM INTEGRATION TEST SUITE")
    print("="*70)
    print("\nNote: Using Qwen-7B for faster testing")
    
    results = {}
    
    # Test 1: VRAM status
    results['vram_status'] = test_vram_status()
    
    # Test 2: Mode transitions
    results['mode_transitions'] = test_mode_transitions()
    
    # Test 3: LLM loading
    results['llm_loading'] = test_llm_loading()
    
    # Test 4: Mutual exclusion
    results['mutual_exclusion'] = test_mutual_exclusion()
    
    # Test 5: LLM query integration
    results['llm_query'] = test_llm_query_integration()
    
    # Test 6: Workflow
    results['workflow'] = test_workflow_simulation_to_llm()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:25s}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ ALL INTEGRATION TESTS PASSED")
        print("\nThe VRAM Manager is fully integrated with LLM!")
    else:
        print(f"\n⚠ {total - passed} test(s) failed")


if __name__ == "__main__":
    main()
