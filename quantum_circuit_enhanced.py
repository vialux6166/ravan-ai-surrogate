"""
Enhanced Quantum Circuit Simulator for Ravan Quantum-ML System
Implements rich quantum circuit simulation with variable qubits, rotation gates, and diverse observables
"""

import sys
sys.path.insert(0, '/home/windows/ravan-quantum-ml/src')

from typing import Dict, Any, Tuple, List
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, partial_trace, entropy as qiskit_entropy
from qiskit_aer import AerSimulator
from scipy.stats import entropy, chisquare
from simulators.base import SimulationModule, SimulationResult, PhysicsValidator


class EnhancedQuantumCircuitSimulator(SimulationModule):
    """
    Enhanced quantum circuit simulator with rich features
    
    Features:
    - Variable qubit counts (2-6 qubits)
    - Parameterized rotation gates (RX, RY, RZ)
    - Multiple gate types (H, X, Y, Z, CNOT, CZ)
    - Circuit depth variation
    - Rich observables: entanglement entropy, fidelity, purity, expectation values
    """
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize enhanced quantum circuit simulator
        
        Args:
            use_gpu: Whether to use GPU backend (if available)
        """
        super().__init__("quantum_circuit_enhanced")
        self.use_gpu = use_gpu
        
        # Initialize Aer simulator
        if use_gpu:
            try:
                self.simulator = AerSimulator(method='statevector', device='GPU')
                self.logger.info("Using GPU-accelerated Qiskit Aer backend")
            except Exception as e:
                self.logger.warning(f"GPU backend not available: {e}, using CPU")
                self.simulator = AerSimulator(method='statevector')
        else:
            self.simulator = AerSimulator(method='statevector')
            self.logger.info("Using CPU Qiskit Aer backend")
    
    def get_parameter_space(self) -> Dict[str, Tuple[float, float]]:
        """
        Return valid parameter ranges with rich variation
        
        Returns:
            Dictionary of parameter ranges
        """
        return {
            # System size
            'n_qubits': (2, 6),  # Variable qubit count
            
            # Rotation angles (in radians)
            'theta_x': (0, 2 * np.pi),  # RX rotation angle
            'theta_y': (0, 2 * np.pi),  # RY rotation angle
            'theta_z': (0, 2 * np.pi),  # RZ rotation angle
            
            # Circuit structure
            'circuit_depth': (1, 5),  # Number of gate layers
            'entangling_pattern': (0, 2),  # 0=linear, 1=all-to-all, 2=ring
            
            # Measurement
            'shots': (1024, 8192),  # Number of measurements
        }
    
    def get_observable_names(self) -> List[str]:
        """
        Get list of observable names
        
        Returns:
            List of observable names
        """
        return [
            'entanglement_entropy',  # Von Neumann entropy of reduced density matrix
            'purity',                # Purity of the quantum state
            'fidelity_to_ghz',      # Fidelity to GHZ state
            'expect_x',             # <X> expectation value
            'expect_y',             # <Y> expectation value
            'expect_z',             # <Z> expectation value
            'circuit_depth_actual', # Actual circuit depth
            'two_qubit_gate_count', # Number of entangling gates
        ]
    
    def create_parameterized_circuit(
        self,
        n_qubits: int,
        theta_x: float,
        theta_y: float,
        theta_z: float,
        circuit_depth: int,
        entangling_pattern: int
    ) -> QuantumCircuit:
        """
        Create parameterized quantum circuit
        
        Args:
            n_qubits: Number of qubits
            theta_x, theta_y, theta_z: Rotation angles
            circuit_depth: Number of gate layers
            entangling_pattern: Entanglement pattern (0=linear, 1=all-to-all, 2=ring)
            
        Returns:
            Quantum circuit
        """
        qr = QuantumRegister(n_qubits, 'q')
        circuit = QuantumCircuit(qr)
        
        # Apply layers
        for layer in range(int(circuit_depth)):
            # Single-qubit rotations
            for i in range(n_qubits):
                # Vary angles per qubit and layer
                angle_x = theta_x * (1 + 0.1 * i) * (1 + 0.1 * layer)
                angle_y = theta_y * (1 + 0.1 * i) * (1 + 0.1 * layer)
                angle_z = theta_z * (1 + 0.1 * i) * (1 + 0.1 * layer)
                
                circuit.rx(angle_x, qr[i])
                circuit.ry(angle_y, qr[i])
                circuit.rz(angle_z, qr[i])
            
            # Entangling gates based on pattern
            if entangling_pattern == 0:  # Linear
                for i in range(n_qubits - 1):
                    circuit.cx(qr[i], qr[i + 1])
            elif entangling_pattern == 1:  # All-to-all
                for i in range(n_qubits):
                    for j in range(i + 1, n_qubits):
                        circuit.cx(qr[i], qr[j])
            elif entangling_pattern == 2:  # Ring
                for i in range(n_qubits):
                    circuit.cx(qr[i], qr[(i + 1) % n_qubits])
        
        return circuit
    
    def calculate_entanglement_entropy(self, statevector: Statevector, n_qubits: int) -> float:
        """
        Calculate entanglement entropy of first qubit
        
        Args:
            statevector: Quantum statevector
            n_qubits: Total number of qubits
            
        Returns:
            Von Neumann entropy in bits
        """
        try:
            # Trace out all qubits except the first
            qubits_to_trace = list(range(1, n_qubits))
            reduced_dm = partial_trace(statevector, qubits_to_trace)
            
            # Calculate Von Neumann entropy
            ent = qiskit_entropy(reduced_dm, base=2)
            return float(ent)
        except Exception as e:
            self.logger.warning(f"Could not calculate entanglement entropy: {e}")
            return 0.0
    
    def calculate_purity(self, statevector: Statevector) -> float:
        """
        Calculate purity of quantum state
        
        Args:
            statevector: Quantum statevector
            
        Returns:
            Purity (Tr(ρ²))
        """
        try:
            # For pure states, purity = 1
            # For mixed states, purity < 1
            density_matrix = statevector.to_operator()
            purity = np.trace(density_matrix @ density_matrix).real
            return float(purity)
        except Exception as e:
            self.logger.warning(f"Could not calculate purity: {e}")
            return 1.0  # Pure state default
    
    def calculate_fidelity_to_ghz(self, statevector: Statevector, n_qubits: int) -> float:
        """
        Calculate fidelity to GHZ state
        
        Args:
            statevector: Quantum statevector
            n_qubits: Number of qubits
            
        Returns:
            Fidelity to GHZ state
        """
        try:
            # Create GHZ state: (|00...0⟩ + |11...1⟩) / √2
            ghz_circuit = QuantumCircuit(n_qubits)
            ghz_circuit.h(0)
            for i in range(n_qubits - 1):
                ghz_circuit.cx(i, i + 1)
            
            ghz_state = Statevector.from_instruction(ghz_circuit)
            
            # Calculate fidelity
            fidelity = statevector.inner(ghz_state)
            return float(np.abs(fidelity) ** 2)
        except Exception as e:
            self.logger.warning(f"Could not calculate fidelity: {e}")
            return 0.0
    
    def calculate_expectation_values(self, statevector: Statevector) -> Dict[str, float]:
        """
        Calculate expectation values of Pauli operators
        
        Args:
            statevector: Quantum statevector
            
        Returns:
            Dictionary of expectation values
        """
        try:
            from qiskit.quantum_info import Pauli
            
            # Calculate <X>, <Y>, <Z> for first qubit
            n_qubits = statevector.num_qubits
            
            # Create Pauli operators
            pauli_x = Pauli('X' + 'I' * (n_qubits - 1))
            pauli_y = Pauli('Y' + 'I' * (n_qubits - 1))
            pauli_z = Pauli('Z' + 'I' * (n_qubits - 1))
            
            # Calculate expectation values
            expect_x = statevector.expectation_value(pauli_x).real
            expect_y = statevector.expectation_value(pauli_y).real
            expect_z = statevector.expectation_value(pauli_z).real
            
            return {
                'expect_x': float(expect_x),
                'expect_y': float(expect_y),
                'expect_z': float(expect_z)
            }
        except Exception as e:
            self.logger.warning(f"Could not calculate expectation values: {e}")
            return {'expect_x': 0.0, 'expect_y': 0.0, 'expect_z': 0.0}
    
    def run(self, params: Dict[str, Any]) -> SimulationResult:
        """
        Execute quantum circuit simulation
        
        Args:
            params: Simulation parameters
                - n_qubits: Number of qubits
                - theta_x, theta_y, theta_z: Rotation angles
                - circuit_depth: Circuit depth
                - entangling_pattern: Entanglement pattern
                - shots: Number of measurements
                
        Returns:
            SimulationResult with observables
        """
        # Extract parameters
        n_qubits = int(params.get('n_qubits', 2))
        theta_x = float(params.get('theta_x', 0.0))
        theta_y = float(params.get('theta_y', 0.0))
        theta_z = float(params.get('theta_z', 0.0))
        circuit_depth = int(params.get('circuit_depth', 1))
        entangling_pattern = int(params.get('entangling_pattern', 0))
        shots = int(params.get('shots', 8192))
        
        # Validate parameters
        if n_qubits < 2 or n_qubits > 10:
            raise ValueError(f"n_qubits must be between 2 and 10, got {n_qubits}")
        
        # Create circuit
        circuit = self.create_parameterized_circuit(
            n_qubits, theta_x, theta_y, theta_z, circuit_depth, entangling_pattern
        )
        
        # Get statevector
        statevector = Statevector.from_instruction(circuit)
        
        # Calculate observables
        observables = {}
        
        # Entanglement entropy
        observables['entanglement_entropy'] = self.calculate_entanglement_entropy(
            statevector, n_qubits
        )
        
        # Purity
        observables['purity'] = self.calculate_purity(statevector)
        
        # Fidelity to GHZ
        observables['fidelity_to_ghz'] = self.calculate_fidelity_to_ghz(
            statevector, n_qubits
        )
        
        # Expectation values
        expect_vals = self.calculate_expectation_values(statevector)
        observables.update(expect_vals)
        
        # Circuit properties
        observables['circuit_depth_actual'] = float(circuit.depth())
        observables['two_qubit_gate_count'] = float(circuit.num_nonlocal_gates())
        
        # Create result
        result = SimulationResult(
            parameters=params,
            observables=observables,
            metadata={
                'simulator': self.name,
                'n_qubits': n_qubits,
                'circuit_depth': circuit_depth,
                'shots': shots
            }
        )
        
        # Validate
        validation_errors = self.validate_output(result)
        result.validation_passed = len(validation_errors) == 0
        result.validation_errors = validation_errors
        
        return result
    
    def validate_output(self, result: SimulationResult) -> List[str]:
        """
        Validate simulation output
        
        Args:
            result: Simulation result
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        obs = result.observables
        
        # Check entanglement entropy bounds
        n_qubits = result.parameters.get('n_qubits', 2)
        max_entropy = 1.0  # For single qubit reduced density matrix
        if obs.get('entanglement_entropy', 0) > max_entropy + 0.01:
            errors.append(f"Entanglement entropy {obs['entanglement_entropy']:.3f} exceeds maximum {max_entropy}")
        
        # Check purity bounds
        if not (0.0 <= obs.get('purity', 1.0) <= 1.0):
            errors.append(f"Purity {obs['purity']:.3f} outside [0, 1]")
        
        # Check fidelity bounds
        if not (0.0 <= obs.get('fidelity_to_ghz', 0.0) <= 1.0):
            errors.append(f"Fidelity {obs['fidelity_to_ghz']:.3f} outside [0, 1]")
        
        # Check expectation values bounds
        for pauli in ['x', 'y', 'z']:
            key = f'expect_{pauli}'
            val = obs.get(key, 0.0)
            if not (-1.0 <= val <= 1.0):
                errors.append(f"Expectation value {key}={val:.3f} outside [-1, 1]")
        
        return errors
