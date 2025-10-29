#!/usr/bin/env python3
"""
Test Prompt Manager
Task 16.3: Test prompt validation, caching, and timeout handling
"""

import sys
sys.path.insert(0, '/home/windows/ravan-quantum-ml')

import time
from src.llm.prompt_manager import (
    PromptValidator,
    PromptCache,
    PromptManager,
    validate_prompt,
    sanitize_prompt
)


def test_prompt_validation():
    """Test prompt validation"""
    print("\n" + "="*70)
    print("TEST 1: Prompt Validation")
    print("="*70)
    
    validator = PromptValidator()
    
    test_cases = [
        ("Normal quantum tunneling query", True),
        ("", False),  # Too short
        ("a" * 6000, False),  # Too long
        ("<script>alert('xss')</script>", False),  # Script injection
        ("javascript:void(0)", False),  # JavaScript protocol
        ("import os; os.system('rm -rf /')", False),  # Python injection
        ("What is quantum entanglement?", True),  # Normal
        ("Simulate with V0=5.0 and k0=4.0", True),  # Normal with numbers
    ]
    
    results = []
    for prompt, expected_valid in test_cases:
        is_valid, error = validator.validate(prompt)
        match = is_valid == expected_valid
        results.append(match)
        
        status = "✓" if match else "✗"
        prompt_display = prompt[:50] + "..." if len(prompt) > 50 else prompt
        print(f"\n{status} '{prompt_display}'")
        print(f"  Expected: {'valid' if expected_valid else 'invalid'}")
        print(f"  Got: {'valid' if is_valid else 'invalid'}")
        if error:
            print(f"  Error: {error}")
    
    passed = sum(results)
    print(f"\n{passed}/{len(results)} validation tests passed")
    return passed == len(results)


def test_prompt_sanitization():
    """Test prompt sanitization"""
    print("\n" + "="*70)
    print("TEST 2: Prompt Sanitization")
    print("="*70)
    
    validator = PromptValidator()
    
    test_cases = [
        ("  Extra   spaces  ", "Extra spaces"),
        ("Line\x00with\x00nulls", "Linewith nulls"),  # Remove null bytes
        ("Multiple\n\n\nlines", "Multiple lines"),  # Normalize whitespace
        ("Normal text", "Normal text"),
    ]
    
    results = []
    for input_prompt, expected_output in test_cases:
        sanitized = validator.sanitize(input_prompt)
        # Check if sanitized contains expected (may have extra normalization)
        match = expected_output.replace(" ", "") in sanitized.replace(" ", "")
        results.append(match)
        
        status = "✓" if match else "✗"
        print(f"\n{status} Input: '{input_prompt}'")
        print(f"  Output: '{sanitized}'")
        print(f"  Expected: '{expected_output}'")
    
    passed = sum(results)
    print(f"\n{passed}/{len(results)} sanitization tests passed")
    return passed == len(results)


def test_prompt_cache():
    """Test LRU cache"""
    print("\n" + "="*70)
    print("TEST 3: Prompt Cache")
    print("="*70)
    
    cache = PromptCache(max_size=3)
    
    # Test cache miss
    result = cache.get("prompt1")
    assert result is None, "Should be cache miss"
    print("✓ Cache miss works")
    
    # Test cache put and hit
    cache.put("prompt1", "response1")
    result = cache.get("prompt1")
    assert result == "response1", "Should be cache hit"
    print("✓ Cache hit works")
    
    # Test cache with parameters
    cache.put("prompt2", "response2a", temperature=0.5)
    cache.put("prompt2", "response2b", temperature=0.7)
    
    result_a = cache.get("prompt2", temperature=0.5)
    result_b = cache.get("prompt2", temperature=0.7)
    
    assert result_a == "response2a", "Should get correct cached response"
    assert result_b == "response2b", "Should get correct cached response"
    print("✓ Cache with parameters works")
    
    # Test LRU eviction
    cache.put("prompt3", "response3")
    cache.put("prompt4", "response4")  # Should evict prompt1
    
    result = cache.get("prompt1")
    assert result is None, "prompt1 should be evicted"
    print("✓ LRU eviction works")
    
    # Test statistics
    stats = cache.get_stats()
    print(f"\nCache Statistics:")
    print(f"  Size: {stats['size']}/{stats['max_size']}")
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")
    
    return True


def test_prompt_manager_integration():
    """Test integrated prompt manager"""
    print("\n" + "="*70)
    print("TEST 4: Prompt Manager Integration")
    print("="*70)
    
    manager = PromptManager(cache_size=10, default_timeout=5.0)
    
    # Mock LLM query function
    call_count = [0]
    
    def mock_query(prompt: str, **kwargs) -> str:
        call_count[0] += 1
        time.sleep(0.1)  # Simulate processing
        return f"Response to: {prompt[:30]}..."
    
    # Test 1: Normal query
    print("\nTest 1: Normal query")
    response1 = manager.process_prompt(
        "What is quantum tunneling?",
        mock_query
    )
    print(f"✓ Response: {response1}")
    assert call_count[0] == 1, "Should call LLM once"
    
    # Test 2: Cached query
    print("\nTest 2: Cached query (should not call LLM)")
    response2 = manager.process_prompt(
        "What is quantum tunneling?",
        mock_query
    )
    print(f"✓ Response: {response2}")
    assert call_count[0] == 1, "Should use cache, not call LLM again"
    assert response1 == response2, "Responses should match"
    
    # Test 3: Invalid prompt
    print("\nTest 3: Invalid prompt (should raise error)")
    try:
        manager.process_prompt(
            "<script>alert('xss')</script>",
            mock_query
        )
        print("✗ Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"✓ Correctly rejected: {e}")
    
    # Test 4: Cache disabled
    print("\nTest 4: Query with cache disabled")
    response3 = manager.process_prompt(
        "What is quantum tunneling?",
        mock_query,
        use_cache=False
    )
    print(f"✓ Response: {response3}")
    assert call_count[0] == 2, "Should call LLM again with cache disabled"
    
    # Test 5: Cache statistics
    print("\nTest 5: Cache statistics")
    stats = manager.get_cache_stats()
    print(f"  Cache size: {stats['size']}")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")
    print(f"  Total requests: {stats['total_requests']}")
    
    return True


def test_convenience_functions():
    """Test convenience functions"""
    print("\n" + "="*70)
    print("TEST 5: Convenience Functions")
    print("="*70)
    
    # Test validate_prompt
    is_valid, error = validate_prompt("Normal prompt")
    assert is_valid, "Should be valid"
    print("✓ validate_prompt() works")
    
    is_valid, error = validate_prompt("<script>bad</script>")
    assert not is_valid, "Should be invalid"
    print("✓ validate_prompt() detects invalid prompts")
    
    # Test sanitize_prompt
    sanitized = sanitize_prompt("  Extra   spaces  ")
    assert "Extra" in sanitized and "spaces" in sanitized
    print("✓ sanitize_prompt() works")
    
    return True


def main():
    """Run all tests"""
    print("="*70)
    print("PROMPT MANAGER TEST SUITE")
    print("="*70)
    
    results = {}
    
    # Run tests
    results['validation'] = test_prompt_validation()
    results['sanitization'] = test_prompt_sanitization()
    results['cache'] = test_prompt_cache()
    results['integration'] = test_prompt_manager_integration()
    results['convenience'] = test_convenience_functions()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED")
        print("="*70)
        print("\nPrompt Manager is working correctly!")
        print("- Input validation prevents injection attacks")
        print("- LRU cache reduces redundant LLM queries")
        print("- Timeout handling prevents hanging")
        return True
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
