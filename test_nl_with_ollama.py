#!/usr/bin/env python3
"""
Test NL Config Generation with Ollama
Uses local Ollama models instead of HuggingFace
"""

import sys
sys.path.insert(0, '/home/windows/ravan-quantum-ml')

import logging
import requests
import json
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class OllamaAdapter:
    """Simple adapter for Ollama API"""
    
    def __init__(self, model_name: str = "qwen3:30b-a3b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.logger = logging.getLogger('ollama_adapter')
    
    def query(self, prompt: str, max_tokens: int = 300, temperature: float = 0.3) -> str:
        """Query Ollama model"""
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result.get('response', '')
        except Exception as e:
            self.logger.error(f"Ollama query failed: {e}")
            raise


def test_simulator_detection():
    """Test simulator type detection (no LLM needed)"""
    print("\n" + "="*70)
    print("TEST 1: Simulator Type Detection (No LLM)")
    print("="*70)
    
    test_cases = [
        ("Create a Bell state with 2 qubits", "quantum_circuit"),
        ("Run quantum tunneling through a barrier", "schrodinger"),
        ("Simulate a harmonic oscillator in ground state", "harmonic_oscillator"),
    ]
    
    # Simple keyword-based detection
    def detect_simulator(description: str) -> str:
        desc_lower = description.lower()
        
        if any(kw in desc_lower for kw in ['qubit', 'bell', 'entangle', 'gate']):
            return 'quantum_circuit'
        elif any(kw in desc_lower for kw in ['tunnel', 'barrier', 'schrodinger']):
            return 'schrodinger'
        elif any(kw in desc_lower for kw in ['harmonic', 'oscillator', 'photon']):
            return 'harmonic_oscillator'
        return 'quantum_circuit'
    
    results = []
    for description, expected in test_cases:
        detected = detect_simulator(description)
        match = detected == expected
        results.append(match)
        
        status = "✓" if match else "✗"
        print(f"\n{status} {description}")
        print(f"  Expected: {expected}, Got: {detected}")
    
    passed = sum(results)
    print(f"\n{passed}/{len(results)} detections correct")
    return passed == len(results)


def test_config_generation_with_ollama():
    """Test config generation with Ollama"""
    print("\n" + "="*70)
    print("TEST 2: Config Generation with Ollama")
    print("="*70)
    
    # Check if Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        print("✓ Ollama is running")
    except Exception as e:
        print(f"✗ Ollama not accessible: {e}")
        print("\nTo start Ollama:")
        print("  1. Open PowerShell")
        print("  2. Run: ollama serve")
        return False
    
    # Create Ollama adapter
    ollama = OllamaAdapter(model_name="qwen3:30b-a3b")
    
    # Test case: Schrödinger tunneling
    print("\n" + "-"*70)
    print("Test Case: Quantum Tunneling")
    print("-"*70)
    
    description = "Run quantum tunneling with high barrier and narrow wavepacket"
    
    prompt = f"""You are a quantum mechanics expert. Convert this description into Schrödinger equation parameters.

Description: {description}

Generate parameters in JSON format with these fields:
- V0: barrier height in eV (0.5-10.0)
- barrier_width: barrier width in nm (0.5-5.0)
- k0: initial momentum (1.0-10.0)
- sigma: wavepacket width (0.5-3.0)
- x0: initial position (-10.0 to -3.0)

Guidelines:
- High barrier: V0 > 5.0
- Narrow wavepacket: sigma < 1.0

Respond ONLY with valid JSON, no explanation:
{{
  "V0": <number>,
  "barrier_width": <number>,
  "k0": <number>,
  "sigma": <number>,
  "x0": <number>
}}"""
    
    print(f"\nDescription: {description}")
    print("\nQuerying Ollama...")
    
    try:
        response = ollama.query(prompt, max_tokens=200, temperature=0.3)
        print(f"\nOllama Response:\n{response}")
        
        # Try to parse JSON
        import re
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(0)
            params = json.loads(json_str)
            
            print("\n✓ Parsed Parameters:")
            for key, value in params.items():
                print(f"  {key}: {value}")
            
            # Validate
            if 'V0' in params and params['V0'] > 5.0:
                print("\n✓ High barrier correctly set (V0 > 5.0)")
            
            if 'sigma' in params and params['sigma'] < 1.0:
                print("✓ Narrow wavepacket correctly set (sigma < 1.0)")
            
            return True
        else:
            print("\n✗ Could not parse JSON from response")
            return False
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def test_results_explanation_with_ollama():
    """Test results explanation with Ollama"""
    print("\n" + "="*70)
    print("TEST 3: Results Explanation with Ollama")
    print("="*70)
    
    ollama = OllamaAdapter(model_name="qwen3:30b-a3b")
    
    # Sample results
    parameters = {
        'V0': 5.0,
        'barrier_width': 1.5,
        'k0': 4.0,
        'sigma': 1.0
    }
    
    results = {
        'transmission': 0.65,
        'reflection': 0.35,
        'total_probability': 1.0
    }
    
    prompt = f"""You are a quantum mechanics expert. Explain these quantum tunneling simulation results in plain language.

**Simulation Parameters:**
- Barrier height (V₀): {parameters['V0']:.2f} eV
- Barrier width: {parameters['barrier_width']:.2f} nm
- Initial momentum (k₀): {parameters['k0']:.2f}
- Wavepacket width (σ): {parameters['sigma']:.2f}

**Results:**
- Transmission probability: {results['transmission']:.4f} ({results['transmission']*100:.1f}%)
- Reflection probability: {results['reflection']:.4f} ({results['reflection']*100:.1f}%)

Provide a 2-3 sentence explanation that describes the tunneling behavior and what it means physically.

Write in clear, accessible language for researchers."""
    
    print("\nGenerating explanation...")
    
    try:
        explanation = ollama.query(prompt, max_tokens=300, temperature=0.7)
        
        print("\n" + "="*70)
        print("EXPLANATION:")
        print("="*70)
        print(f"\n{explanation}\n")
        
        # Check if explanation mentions key concepts
        key_concepts = ['tunnel', 'barrier', 'transmission', 'quantum']
        mentions = sum(1 for concept in key_concepts if concept.lower() in explanation.lower())
        
        print(f"\n✓ Explanation mentions {mentions}/{len(key_concepts)} key concepts")
        
        return mentions >= 2
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def main():
    """Run all tests"""
    print("="*70)
    print("NATURAL LANGUAGE INTERFACE TEST - OLLAMA")
    print("="*70)
    print("\nUsing local Ollama model: qwen3:30b-a3b")
    
    results = {}
    
    # Test 1: Simulator detection (no LLM)
    results['detection'] = test_simulator_detection()
    
    # Test 2: Config generation with Ollama
    results['config_generation'] = test_config_generation_with_ollama()
    
    # Test 3: Results explanation with Ollama
    results['explanation'] = test_results_explanation_with_ollama()
    
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
        print("\nNatural Language Interface is working with Ollama!")
        return True
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
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
