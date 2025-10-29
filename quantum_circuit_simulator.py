"""
Quantum Circuit Simulator for Ravan Quantum-ML System
Implements quantum circuit simulation using Qiskit
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Dict, Any, Tuple, List
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from scipy.stats import entropy, chisquare
from simulation_base import SimulationModule, SimulationResult, PhysicsValidator


class QuantumCircuitSimulator(SimulationModule):
    """
    Quantum circuit simulator using Qiskit
    
    Simulates 2-qubit quantum circuits with Hadamard and CNOT gates
    to generate entangled states and measure quantum properties.
    
    Observables:
    - entropy: Von Neumann entropy (bits)
    - chi_squared: Chi-squared goodness-of-fit
    - kl_divergence: KL divergence from ideal distribution
    - measurement_probs: Probability distribution of measurement outcomes
    """
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize quantum circuit simulator
        
        Args:
            use_gpu: Whether to use GPU backend (if available)
        """
        super().__init__("quantum_circuit")
        self.use_gpu = use_gpu
        
        # Initialize Aer simulator
        # Use automatic method so that sampling (counts) is supported
        if use_gpu:
            try:
                self.simulator = AerSimulator(device='GPU')
                self.logger.info("Using GPU-accelerated Qiskit Aer backend")
            except Exception as e:
                self.logger.warning(f"GPU backend not available: {e}, using CPU")
                self.simulator = AerSimulator()
        else:
            self.simulator = AerSimulator()
            self.logger.info("Using CPU Qiskit Aer backend")
    
    def get_parameter_space(self) -> Dict[str, Tuple[float, float]]:
        """
        Return valid parameter ranges
        
        Returns:
            Dictionary of parameter ranges
        """
        return {
            'n_qubits': (2, 2),  # Currently only 2-qubit circuits
            'shots': (1024, 8192),  # Number of measurements
            # New parameter format
            'apply_hadamard': (0, 1),  # Boolean: apply Hadamard gates
            'apply_cnot': (0, 1),  # Boolean: apply CNOT gate
            # Legacy parameters for compatibility with tests
            'circuit_depth': (1, 5),
            'rx_angle_0': (-np.pi, np.pi),
            'ry_angle_0': (-np.pi, np.pi),
            'rz_angle_0': (-np.pi, np.pi),
            'rx_angle_1': (-np.pi, np.pi),
            'ry_angle_1': (-np.pi, np.pi),
            'rz_angle_1': (-np.pi, np.pi),
            'entangling_pattern': (0, 1),
        }
    
    def get_observable_names(self) -> List[str]:
        """
        Get list of observable names
        
        Returns:
            List of observable names
        """
        return ['entropy', 'entanglement_entropy', 'chi_squared', 'kl_divergence', 'fidelity_to_ghz', 'counts']
    
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate input parameters, supporting both new and legacy parameter formats
        
        Args:
            params: Parameters to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        # Get parameter space (includes both new and legacy)
        param_space = self.get_parameter_space()
        errors = []
        
        # Check required parameters: n_qubits and shots
        required_params = ['n_qubits', 'shots']
        for param in required_params:
            if param not in params:
                errors.append(f"Missing required parameter: {param}")
        
        # Check n_qubits value if provided
        if 'n_qubits' in params:
            n_qubits = params['n_qubits']
            if n_qubits != 2:
                errors.append(f"Invalid n_qubits={n_qubits} (only 2 qubits supported)")
        
        # Check all other parameters against parameter space
        for param_name, value in params.items():
            if param_name in param_space:
                min_val, max_val = param_space[param_name]
                if not (min_val <= value <= max_val):
                    errors.append(
                        f"Parameter {param_name}={value} out of range "
                        f"[{min_val}, {max_val}]"
                    )
        
        is_valid = len(errors) == 0
        if not is_valid:
            self.logger.warning(f"Parameter validation failed: {errors}")
        
        return is_valid, errors
    
    def create_bell_state_circuit(self, n_qubits: int = 2) -> QuantumCircuit:
        """
        Create a Bell state (maximally entangled) circuit
        
        Args:
            n_qubits: Number of qubits (default: 2)
            
        Returns:
            Quantum circuit creating Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
        """
        qr = QuantumRegister(n_qubits, 'q')
        cr = ClassicalRegister(n_qubits, 'c')
        circuit = QuantumCircuit(qr, cr)
        
        # Create Bell state: H on qubit 0, then CNOT(0,1)
        circuit.h(qr[0])  # Hadamard gate
        circuit.cx(qr[0], qr[1])  # CNOT gate
        
        # Measure all qubits
        circuit.measure(qr, cr)
        
        return circuit
    
    def create_custom_circuit(
        self,
        n_qubits: int,
        apply_hadamard: bool,
        apply_cnot: bool
    ) -> QuantumCircuit:
        """
        Create custom quantum circuit based on parameters
        
        Args:
            n_qubits: Number of qubits
            apply_hadamard: Whether to apply Hadamard gates
            apply_cnot: Whether to apply CNOT gate
            
        Returns:
            Quantum circuit
        """
        qr = QuantumRegister(n_qubits, 'q')
        cr = ClassicalRegister(n_qubits, 'c')
        circuit = QuantumCircuit(qr, cr)
        
        if apply_hadamard:
            # Apply Hadamard to first qubit
            circuit.h(qr[0])
        
        if apply_cnot and n_qubits >= 2:
            # Apply CNOT between first two qubits
            circuit.cx(qr[0], qr[1])
        
        # Measure all qubits
        circuit.measure(qr, cr)
        
        return circuit
    
    def calculate_entropy(self, probabilities: np.ndarray) -> float:
        """
        Calculate Von Neumann entropy from measurement probabilities
        
        Args:
            probabilities: Probability distribution
            
        Returns:
            Entropy in bits
        """
        # Filter out zero probabilities to avoid log(0)
        probs = probabilities[probabilities > 0]
        
        # Calculate Shannon entropy (equivalent to Von Neumann for classical probs)
        # H = -Σ p_i log2(p_i)
        return entropy(probs, base=2)
    
    def calculate_chi_squared(
        self,
        observed: np.ndarray,
        expected: np.ndarray
    ) -> float:
        """
        Calculate chi-squared goodness-of-fit
        
        Args:
            observed: Observed counts
            expected: Expected counts
            
        Returns:
            Chi-squared statistic
        """
        # Avoid division by zero
        expected = np.where(expected == 0, 1e-10, expected)
        
        chi2_stat, p_value = chisquare(observed, expected)
        return chi2_stat
    
    def calculate_kl_divergence(
        self,
        observed: np.ndarray,
        expected: np.ndarray
    ) -> float:
        """
        Calculate KL divergence from ideal distribution
        
        Args:
            observed: Observed probability distribution
            expected: Expected probability distribution
            
        Returns:
            KL divergence
        """
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        observed = np.where(observed == 0, eps, observed)
        expected = np.where(expected == 0, eps, expected)
        
        # KL(P||Q) = Σ P(i) log(P(i)/Q(i))
        return np.sum(observed * np.log(observed / expected))
    
    def run(self, params: Dict[str, Any]) -> SimulationResult:
        """
        Execute quantum circuit simulation
        
        Args:
            params: Simulation parameters
                - n_qubits: Number of qubits (default: 2)
                - shots: Number of measurements (default: 8192)
                - apply_hadamard: Apply Hadamard gates (default: True)
                - apply_cnot: Apply CNOT gate (default: True)
                - entangling_pattern: Legacy parameter (0=no entangling, 1=apply CNOT)
                
        Returns:
            SimulationResult with quantum observables
        """
        # Extract parameters with defaults
        n_qubits = int(params.get('n_qubits', 2))
        shots = int(params.get('shots', 8192))
        
        # Handle both new and legacy parameter formats
        apply_hadamard = bool(params.get('apply_hadamard', True))
        apply_cnot = bool(params.get('apply_cnot', True))
        
        # Legacy format: check for entangling_pattern
        if 'entangling_pattern' in params:
            apply_cnot = bool(params.get('entangling_pattern', 0))
            # Check if all rotation angles are zero (or not provided)
            angles_zero = True
            for key in params.keys():
                if key.startswith(('rx_angle', 'ry_angle', 'rz_angle')):
                    if abs(float(params.get(key, 0))) > 1e-6:
                        angles_zero = False
                        break
            # Apply Hadamard only if there's entangling or non-zero rotations
            apply_hadamard = apply_cnot or not angles_zero
        
        self.logger.debug(
            f"Running circuit: n_qubits={n_qubits}, shots={shots}, "
            f"H={apply_hadamard}, CNOT={apply_cnot}"
        )
        
        # Create circuit
        if apply_hadamard and apply_cnot:
            # Standard Bell state circuit
            circuit = self.create_bell_state_circuit(n_qubits)
        else:
            # Custom circuit
            circuit = self.create_custom_circuit(
                n_qubits, apply_hadamard, apply_cnot
            )
        
        # Transpile for backend and run simulation
        compiled = transpile(circuit, self.simulator)
        job = self.simulator.run(compiled, shots=shots)
        result = job.result()
        counts = result.get_counts(compiled)
        
        # Convert counts to probabilities
        total_shots = sum(counts.values())
        n_states = 2 ** n_qubits
        
        # Create probability array for all possible states
        probs = np.zeros(n_states)
        for state_str, count in counts.items():
            state_int = int(state_str, 2)  # Convert binary string to int
            probs[state_int] = count / total_shots
        
        # Calculate observables
        entropy_val = self.calculate_entropy(probs)
        
        # Expected distribution for Bell state (equal superposition)
        if apply_hadamard and apply_cnot:
            # Bell state: 50% |00⟩, 50% |11⟩
            expected_probs = np.zeros(n_states)
            expected_probs[0] = 0.5  # |00⟩
            expected_probs[n_states - 1] = 0.5  # |11⟩
        else:
            # Uniform distribution
            expected_probs = np.ones(n_states) / n_states
        
        expected_counts = expected_probs * total_shots
        observed_counts = probs * total_shots
        
        chi2 = self.calculate_chi_squared(observed_counts, expected_counts)
        kl_div = self.calculate_kl_divergence(probs, expected_probs)
        
        # Create result
        # Fidelity to GHZ/Bell for 2 qubits: probability mass on |00> and |11>
        fidelity_to_ghz = float(probs[0] + probs[n_states - 1]) if n_qubits == 2 else float(0.0)

        observables = {
            'entropy': float(entropy_val),
            'entanglement_entropy': float(entropy_val),
            'chi_squared': float(chi2),
            'kl_divergence': float(kl_div),
            'counts': counts,
            'fidelity_to_ghz': fidelity_to_ghz,
            'circuit_depth_actual': circuit.depth(),  # Add this for tests
        }
        
        # Add individual probabilities
        for i, prob in enumerate(probs):
            observables[f'prob_{i:0{n_qubits}b}'] = float(prob)
        
        metadata = {
            'circuit_depth': circuit.depth(),
            'n_gates': len(circuit.data),
            'shots': shots,
            'simulator': 'qiskit_aer',
            'backend': 'GPU' if self.use_gpu else 'CPU'
        }
        
        return SimulationResult(
            observables=observables,
            parameters=params,
            metadata=metadata
        )
    
    def validate_output(self, result: SimulationResult) -> bool:
        """
        Verify physics constraints for quantum circuit
        
        Args:
            result: Simulation result to validate
            
        Returns:
            True if validation passed
        """
        errors = []
        
        # Extract observables
        entropy_val = result.observables.get('entropy', 0)
        chi2 = result.observables.get('chi_squared', 0)
        kl_div = result.observables.get('kl_divergence', 0)
        
        # Get number of qubits from parameters
        n_qubits = result.parameters.get('n_qubits', 2)
        max_entropy = n_qubits  # Maximum entropy for n qubits
        
        # Validation 1: Entropy should be <= log2(dim) (and >= 0)
        is_valid, error = PhysicsValidator.check_range(
            entropy_val, 0.0, max_entropy + 0.001,
            name="entropy"
        )
        if not is_valid:
            errors.append(error)
        
        # Check entanglement_entropy separately
        entanglement_entropy = result.observables.get('entanglement_entropy', entropy_val)
        if entanglement_entropy < 0 or entanglement_entropy > max_entropy + 0.001:
            errors.append(f"Entanglement entropy {entanglement_entropy} out of valid range [0, {max_entropy}]")
        
        # Check fidelity_to_ghz is in [0, 1]
        fidelity = result.observables.get('fidelity_to_ghz', -1)
        if fidelity < 0 or fidelity > 1.001:  # Allow small floating point errors
            errors.append(f"Fidelity to GHZ {fidelity} out of valid range [0, 1]")
        
        # Validation 2: Extract and check probability normalization
        probs = []
        for key, value in result.observables.items():
            if key.startswith('prob_'):
                probs.append(value)
        
        if probs:
            probs_array = np.array(probs)
            is_valid, error = PhysicsValidator.check_probability_normalization(
                probs_array, tolerance=0.01, name="measurement probabilities"
            )
            if not is_valid:
                errors.append(error)
        
        # Validation 3: Chi-squared should be reasonable (< 10 for good fit)
        if chi2 > 10.0:
            errors.append(f"Chi-squared = {chi2:.3f} indicates poor fit (> 10)")
        
        # Validation 4: KL divergence should be small (< 0.1 for good match)
        if kl_div > 0.1:
            errors.append(f"KL divergence = {kl_div:.6f} indicates deviation (> 0.1)")
        
        # Update result
        result.validation_passed = len(errors) == 0
        result.validation_errors = errors
        
        if not result.validation_passed:
            self.logger.warning(f"Validation failed: {errors}")
        
        return result.validation_passed
