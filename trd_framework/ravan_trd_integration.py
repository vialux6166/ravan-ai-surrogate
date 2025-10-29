#!/usr/bin/env python3
"""
Ravan-TRD Integration Layer

Integrates Topological Resonance Distillation with the Ravan Quantum-ML System.
This creates physics-grounded queries from actual simulation results.
"""

import sys
sys.path.insert(0, '.')

import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import time

# Import Ravan components
try:
    from schrodinger_solver import SchrodingerSolver
    from quantum_circuit_simulator import QuantumCircuitSimulator
    from harmonic_oscillator import HarmonicOscillatorModule
except ImportError:
    print("Warning: Could not import Ravan simulators. Make sure you're in the Ravan directory.")


class RavanTRDIntegration:
    """
    Integration layer between Ravan and TRD framework
    
    This class:
    1. Uses Ravan's simulation results to create physics-grounded queries
    2. Generates reasoning datasets specific to quantum physics
    3. Creates a specialized student model for Ravan's domain
    """
    
    def __init__(self):
        """Initialize Ravan-TRD integration"""
        print("\n" + "="*70)
        print("RAVAN-TRD INTEGRATION LAYER")
        print("="*70)
        
        try:
            self.simulators = {
                'schrodinger': SchrodingerSolver(),
                'quantum_circuit': QuantumCircuitSimulator(),
                'harmonic_oscillator': HarmonicOscillatorModule()
            }
            print("✓ Ravan simulators loaded successfully")
        except Exception as e:
            print(f"Warning: Could not initialize simulators: {e}")
            self.simulators = {}
    
    def generate_physics_grounded_queries(self, n_samples: int = 60) -> List[Dict]:
        """
        Generate queries based on actual Ravan simulation results
        
        This creates a dataset where:
        1. We run real simulations
        2. Create questions about the physics
        3. Include the ground truth from simulations
        
        Args:
            n_samples: Number of simulation-based queries to generate
            
        Returns:
            List of physics-grounded queries with simulation data
        """
        print("\n" + "="*70)
        print("GENERATING PHYSICS-GROUNDED QUERIES")
        print("="*70)
        
        queries = []
        
        # Generate Schrödinger-based queries
        print("\n1. Schrödinger Solver Queries...")
        for i in range(n_samples // 3):
            try:
                # Random parameters
                params = {
                    'V0': float(np.random.uniform(1.0, 10.0)),
                    'barrier_width': float(np.random.uniform(0.5, 3.0)),
                    'k0': float(np.random.uniform(2.0, 8.0)),
                    'sigma': float(np.random.uniform(0.5, 2.0)),
                    'x0': float(np.random.uniform(-8.0, -3.0))
                }
                
                # Run simulation
                result = self.simulators['schrodinger'].run(params)
                
                # Create query
                query = self._create_schrodinger_query(params, result)
                queries.append(query)
                
                if (i + 1) % 5 == 0:
                    print(f"  Generated {i + 1} Schrödinger queries")
            except Exception as e:
                print(f"  Warning: Failed to generate query {i+1}: {e}")
        
        # Generate Quantum Circuit queries
        print("\n2. Quantum Circuit Queries...")
        for i in range(n_samples // 3):
            try:
                params = {
                    'n_qubits': int(np.random.choice([2, 3])),
                    'shots': 1000,
                    'gate_sequence': str(np.random.choice(['bell', 'ghz']))
                }
                
                result = self.simulators['quantum_circuit'].run(params)
                query = self._create_circuit_query(params, result)
                queries.append(query)
                
                if (i + 1) % 5 == 0:
                    print(f"  Generated {i + 1} Circuit queries")
            except Exception as e:
                print(f"  Warning: Failed to generate query {i+1}: {e}")
        
        # Generate Harmonic Oscillator queries
        print("\n3. Harmonic Oscillator Queries...")
        for i in range(n_samples // 3):
            try:
                params = {
                    'oscillator_length': float(np.random.uniform(0.5, 2.0)),
                    'basis_size': int(np.random.choice([10, 15, 20])),
                    'initial_n': int(np.random.choice([0, 1, 2, 3, 4, 5]))
                }
                
                result = self.simulators['harmonic_oscillator'].run(params)
                query = self._create_harmonic_query(params, result)
                queries.append(query)
                
                if (i + 1) % 5 == 0:
                    print(f"  Generated {i + 1} Harmonic queries")
            except Exception as e:
                print(f"  Warning: Failed to generate query {i+1}: {e}")
        
        print(f"\n✓ Generated {len(queries)} physics-grounded queries")
        return queries
    
    def _create_schrodinger_query(self, params: Dict, result) -> Dict:
        """Create Schrödinger-specific reasoning query"""
        V0 = params['V0']
        width = params['barrier_width']
        k0 = params['k0']
        T = float(result.observables['transmission'])
        R = float(result.observables['reflection'])
        
        # Create physics question
        query_text = f"""A quantum wavepacket with initial momentum k₀ = {k0:.2f} encounters a rectangular potential barrier of height V₀ = {V0:.2f} eV and width = {width:.2f} nm.

Question: Explain the quantum tunneling process and predict the transmission and reflection coefficients. Consider:
1. How does the barrier height compare to the particle's kinetic energy?
2. What quantum mechanical principles govern the tunneling probability?
3. How would changing the barrier width affect the transmission?

Provide a step-by-step analysis including the relevant physics equations."""
        
        return {
            'query': query_text,
            'simulation_type': 'schrodinger',
            'parameters': params,
            'ground_truth': {
                'transmission': T,
                'reflection': R,
                'conservation_check': abs(T + R - 1.0) < 1e-10
            },
            'physics_concepts': [
                'quantum_tunneling',
                'wavefunction',
                'probability_conservation',
                'barrier_penetration'
            ]
        }
    
    def _create_circuit_query(self, params: Dict, result) -> Dict:
        """Create quantum circuit reasoning query"""
        n_qubits = params['n_qubits']
        gate_seq = params['gate_sequence']
        entropy = float(result.observables.get('entropy', 0))
        
        query_text = f"""A {n_qubits}-qubit quantum circuit implements a '{gate_seq}' gate sequence with {params['shots']} measurement shots.

Question: Analyze the quantum circuit and explain:
1. What type of quantum state is created by this gate sequence?
2. How does entanglement manifest in the measurement statistics?
3. What is the Von Neumann entropy and what does it tell us about the state?
4. How would increasing the number of qubits affect the entanglement?

Provide a detailed quantum information theory analysis."""
        
        return {
            'query': query_text,
            'simulation_type': 'quantum_circuit',
            'parameters': params,
            'ground_truth': {
                'entropy': entropy,
                'n_qubits': n_qubits,
                'gate_sequence': gate_seq
            },
            'physics_concepts': [
                'quantum_entanglement',
                'von_neumann_entropy',
                'quantum_gates',
                'measurement_statistics'
            ]
        }
    
    def _create_harmonic_query(self, params: Dict, result) -> Dict:
        """Create harmonic oscillator reasoning query"""
        length = params['oscillator_length']
        n_initial = params['initial_n']
        photon_number = float(result.observables.get('photon_number', 0))
        
        query_text = f"""A quantum harmonic oscillator with characteristic length {length:.2f} is prepared in the n = {n_initial} energy eigenstate.

Question: Analyze the quantum harmonic oscillator and explain:
1. What is the energy of this state in terms of ħω?
2. How does the wavefunction look in position and momentum space?
3. What is the expectation value of the photon number operator?
4. How would the state evolve under time evolution?

Provide a comprehensive quantum mechanical analysis."""
        
        return {
            'query': query_text,
            'simulation_type': 'harmonic_oscillator',
            'parameters': params,
            'ground_truth': {
                'photon_number': photon_number,
                'energy_level': n_initial,
                'oscillator_length': length
            },
            'physics_concepts': [
                'quantum_harmonic_oscillator',
                'energy_eigenstates',
                'photon_number',
                'time_evolution'
            ]
        }
    
    def _get_general_quantum_queries(self) -> List[Dict]:
        """Get general quantum mechanics queries"""
        queries = [
            {
                'query': "Explain the physical meaning of wavefunction collapse and how it relates to the measurement problem in quantum mechanics.",
                'simulation_type': 'general',
                'physics_concepts': ['wavefunction_collapse', 'measurement_problem']
            },
            {
                'query': "Derive the time-dependent Schrödinger equation from first principles and explain each term.",
                'simulation_type': 'general',
                'physics_concepts': ['schrodinger_equation', 'time_evolution']
            },
            {
                'query': "What is the difference between pure and mixed quantum states, and how is this reflected in the density matrix formalism?",
                'simulation_type': 'general',
                'physics_concepts': ['pure_states', 'mixed_states', 'density_matrix']
            },
            {
                'query': "Explain the Heisenberg uncertainty principle and derive it from the commutation relations.",
                'simulation_type': 'general',
                'physics_concepts': ['uncertainty_principle', 'commutation_relations']
            },
            {
                'query': "What is quantum decoherence and how does it explain the emergence of classical behavior?",
                'simulation_type': 'general',
                'physics_concepts': ['decoherence', 'quantum_to_classical']
            }
        ]
        return queries
    
    def _get_ml_physics_queries(self) -> List[Dict]:
        """Get ML for physics queries"""
        queries = [
            {
                'query': "How can neural networks be trained to respect physical conservation laws, and why is this important for quantum simulations?",
                'simulation_type': 'ml_physics',
                'physics_concepts': ['physics_informed_ml', 'conservation_laws']
            },
            {
                'query': "Explain the challenges of using machine learning to predict quantum observables and how uncertainty quantification helps.",
                'simulation_type': 'ml_physics',
                'physics_concepts': ['uncertainty_quantification', 'ml_validation']
            },
            {
                'query': "What are the advantages and limitations of using ML to accelerate quantum Monte Carlo simulations?",
                'simulation_type': 'ml_physics',
                'physics_concepts': ['quantum_monte_carlo', 'ml_acceleration']
            },
            {
                'query': "How does adaptive sampling improve the efficiency of training ML models for physics simulations?",
                'simulation_type': 'ml_physics',
                'physics_concepts': ['adaptive_sampling', 'active_learning']
            }
        ]
        return queries
    
    def create_ravan_specific_dataset(
        self,
        output_path: str = "trd_framework/ravan_physics_queries.jsonl",
        n_simulation_queries: int = 60
    ) -> List[Dict]:
        """
        Create a complete query dataset specifically for Ravan's quantum physics domain
        
        This combines:
        1. Physics-grounded queries from simulations
        2. General quantum mechanics questions
        3. ML for physics questions
        
        Args:
            output_path: Where to save the dataset
            n_simulation_queries: Number of simulation-based queries
            
        Returns:
            List of all queries
        """
        print("\n" + "="*70)
        print("CREATING RAVAN-SPECIFIC QUERY DATASET")
        print("="*70)
        
        all_queries = []
        
        # Generate physics-grounded queries
        if self.simulators:
            physics_queries = self.generate_physics_grounded_queries(n_samples=n_simulation_queries)
            all_queries.extend(physics_queries)
        else:
            print("\nWarning: Simulators not available, skipping physics-grounded queries")
        
        # Add general quantum mechanics queries
        general_queries = self._get_general_quantum_queries()
        all_queries.extend(general_queries)
        
        # Add ML for physics queries
        ml_queries = self._get_ml_physics_queries()
        all_queries.extend(ml_queries)
        
        print(f"\nTotal queries for TRD dataset: {len(all_queries)}")
        print(f"  Physics-grounded: {len(all_queries) - len(general_queries) - len(ml_queries)}")
        print(f"  General quantum: {len(general_queries)}")
        print(f"  ML for physics: {len(ml_queries)}")
        
        # Save dataset
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for query in all_queries:
                f.write(json.dumps(query, ensure_ascii=False) + '\n')
        
        print(f"\n✓ Ravan-specific query dataset saved to: {output_path}")
        return all_queries


def main():
    """Main integration function"""
    print("\n" + "="*70)
    print("RAVAN-TRD INTEGRATION")
    print("="*70)
    print("\nIntegrating Topological Resonance Distillation with Ravan Quantum-ML System")
    
    # Initialize integration
    integration = RavanTRDIntegration()
    
    # Create Ravan-specific query dataset
    queries = integration.create_ravan_specific_dataset()
    
    print(f"\n✓ Integration complete!")
    print(f"  Query dataset size: {len(queries)} examples")
    print(f"  Next step: Use these queries with teacher model to generate reasoning dataset")
    
    return queries


if __name__ == '__main__':
    queries = main()
