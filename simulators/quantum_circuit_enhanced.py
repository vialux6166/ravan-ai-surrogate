"""
Shim module for legacy import path `simulators.quantum_circuit_enhanced`.
Re-exports QuantumCircuitSimulator from root-level `quantum_circuit_simulator.py`.
"""

from quantum_circuit_simulator import QuantumCircuitSimulator as EnhancedQuantumCircuitSimulator  # noqa: F401


