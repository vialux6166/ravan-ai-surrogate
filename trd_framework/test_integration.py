#!/usr/bin/env python3
"""
Simple test script to verify TRD integration works
"""

import sys
sys.path.insert(0, '.')

print("Testing TRD Integration...")
print("=" * 70)

# Test 1: Import modules
print("\n1. Testing imports...")
try:
    from trd_framework.ravan_trd_integration import RavanTRDIntegration
    print("   ✓ RavanTRDIntegration imported")
except Exception as e:
    print(f"   ✗ Failed to import: {e}")
    sys.exit(1)

try:
    from trd_framework.phase1_dataset_generation import ResonanceDatasetGenerator
    print("   ✓ ResonanceDatasetGenerator imported")
except Exception as e:
    print(f"   ✗ Failed to import: {e}")
    sys.exit(1)

# Test 2: Initialize integration
print("\n2. Testing RavanTRDIntegration initialization...")
try:
    integration = RavanTRDIntegration()
    print("   ✓ Integration initialized")
except Exception as e:
    print(f"   ✗ Failed to initialize: {e}")
    sys.exit(1)

# Test 3: Generate general queries
print("\n3. Testing query generation...")
try:
    general_queries = integration._get_general_quantum_queries()
    print(f"   ✓ Generated {len(general_queries)} general queries")
    
    ml_queries = integration._get_ml_physics_queries()
    print(f"   ✓ Generated {len(ml_queries)} ML queries")
except Exception as e:
    print(f"   ✗ Failed to generate queries: {e}")
    sys.exit(1)

# Test 4: Create query dataset (without simulations)
print("\n4. Testing query dataset creation...")
try:
    # This will skip simulation queries if simulators aren't available
    queries = integration.create_ravan_specific_dataset(
        output_path="trd_framework/test_queries.jsonl",
        n_simulation_queries=0  # Skip simulations for now
    )
    print(f"   ✓ Created query dataset with {len(queries)} queries")
except Exception as e:
    print(f"   ✗ Failed to create dataset: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ All tests passed!")
print("=" * 70)
print("\nTRD Integration is working correctly.")
print("\nNext steps:")
print("1. Ensure Ollama is running: ollama run qwen2.5:32b")
print("2. Run Phase 1: python trd_framework/run_phase1_complete.py")
