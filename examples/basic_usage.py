"""
Ravan Basic Usage Examples
Demonstrates core functionality with and without LLM
"""

import sys
sys.path.append('..')

from ravan import Ravan
import numpy as np


def example_1_basic_simulation():
    """Example 1: Basic simulation without LLM"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Simulation (No LLM)")
    print("="*70)
    
    ravan = Ravan()
    
    # Run Schrödinger simulation
    results = ravan.simulate({
        'simulator': 'schrodinger',
        'V0': 5.0,
        'k0': 4.0,
        'barrier_width': 1.5,
        'sigma': 1.0,
        'x0': -5.0
    })
    
    print(f"\nResults:")
    print(f"  Transmission: {results.get('transmission', 'N/A'):.4f}")
    print(f"  Reflection: {results.get('reflection', 'N/A'):.4f}")
    print(f"  Total Probability: {results.get('total_probability', 'N/A'):.6f}")


def example_2_natural_language_simulation():
    """Example 2: Natural language simulation with LLM"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Natural Language Simulation (With LLM)")
    print("="*70)
    
    try:
        with Ravan(use_llm=True) as ravan:
            # Describe simulation in plain English
            results = ravan.simulate(
                "quantum tunneling with high barrier and narrow wavepacket",
                explain=True
            )
            
            print(f"\nResults:")
            print(f"  Transmission: {results.get('transmission', 'N/A'):.4f}")
            print(f"  Reflection: {results.get('reflection', 'N/A'):.4f}")
            
            if 'explanation' in results:
                print(f"\nExplanation:")
                print(f"  {results['explanation']}")
    except Exception as e:
        print(f"⚠ LLM not available: {e}")
        print("  Install LLM dependencies: pip install -r requirements-llm.txt")


def example_3_multiple_simulators():
    """Example 3: Run all three simulators"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Multiple Simulators")
    print("="*70)
    
    ravan = Ravan()
    
    # Quantum circuit
    print("\n--- Quantum Circuit ---")
    results = ravan.simulate({
        'simulator': 'quantum_circuit',
        'n_qubits': 2,
        'shots': 1000,
        'gate_sequence': 'bell'
    })
    print(f"  Entropy: {results.get('entropy', 'N/A'):.4f} bits")
    print(f"  Chi-squared: {results.get('chi_squared', 'N/A'):.4f}")
    
    # Schrödinger
    print("\n--- Schrödinger Solver ---")
    results = ravan.simulate({
        'simulator': 'schrodinger',
        'V0': 3.0,
        'k0': 4.0,
        'barrier_width': 1.5
    })
    print(f"  Transmission: {results.get('transmission', 'N/A'):.4f}")
    print(f"  Reflection: {results.get('reflection', 'N/A'):.4f}")
    
    # Harmonic oscillator
    print("\n--- Harmonic Oscillator ---")
    results = ravan.simulate({
        'simulator': 'harmonic',
        'oscillator_length': 1.0,
        'basis_size': 10,
        'initial_n': 2
    })
    print(f"  Ground state energy: {results.get('energy_0', 'N/A'):.4f}")
    print(f"  Energy spacing: {results.get('energy_spacing', 'N/A'):.4f}")


def example_4_ask_questions():
    """Example 4: Ask LLM questions about quantum mechanics"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Ask Questions (With LLM)")
    print("="*70)
    
    try:
        with Ravan(use_llm=True) as ravan:
            questions = [
                "What is quantum tunneling?",
                "Explain the Heisenberg uncertainty principle in one sentence.",
                "What is quantum entanglement?"
            ]
            
            for question in questions:
                print(f"\nQ: {question}")
                answer = ravan.query(question, max_tokens=150)
                print(f"A: {answer}")
    except Exception as e:
        print(f"⚠ LLM not available: {e}")


def example_5_generate_code():
    """Example 5: Generate quantum computing code"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Generate Code (With LLM)")
    print("="*70)
    
    try:
        with Ravan(use_llm=True) as ravan:
            code = ravan.generate_code(
                "Create a Bell state circuit with measurement",
                code_type='quantum_circuit'
            )
            
            print("\nGenerated Code:")
            print("-" * 70)
            print(code)
            print("-" * 70)
    except Exception as e:
        print(f"⚠ LLM not available: {e}")


def example_6_explain_results():
    """Example 6: Get explanations for simulation results"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Explain Results (With LLM)")
    print("="*70)
    
    try:
        with Ravan(use_llm=True) as ravan:
            # Run simulation
            results = ravan.simulate({
                'simulator': 'schrodinger',
                'V0': 5.0,
                'k0': 4.0,
                'barrier_width': 1.5
            })
            
            print(f"\nSimulation Results:")
            print(f"  Transmission: {results.get('transmission', 'N/A'):.4f}")
            print(f"  Reflection: {results.get('reflection', 'N/A'):.4f}")
            
            # Get explanation
            explanation = ravan.explain(results)
            
            print(f"\nPhysics Explanation:")
            print(f"  {explanation}")
    except Exception as e:
        print(f"⚠ LLM not available: {e}")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("RAVAN QUANTUM-ML SYSTEM - USAGE EXAMPLES")
    print("="*70)
    
    examples = [
        ("Basic Simulation", example_1_basic_simulation),
        ("Natural Language Simulation", example_2_natural_language_simulation),
        ("Multiple Simulators", example_3_multiple_simulators),
        ("Ask Questions", example_4_ask_questions),
        ("Generate Code", example_5_generate_code),
        ("Explain Results", example_6_explain_results),
    ]
    
    for name, func in examples:
        try:
            func()
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            break
        except Exception as e:
            print(f"\n✗ Example '{name}' failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("EXAMPLES COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
