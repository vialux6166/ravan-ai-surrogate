"""
Compatibility shims for legacy import paths used in tests.
This package re-exports classes from root-level modules.
"""

# Export Simulation base types
from simulation_base import SimulationModule, SimulationResult, PhysicsValidator  # noqa: F401

# Export simulators
from schrodinger_solver import SchrodingerSolver  # noqa: F401
from quantum_circuit_simulator import QuantumCircuitSimulator  # noqa: F401
from harmonic_oscillator import HarmonicOscillator  # noqa: F401


