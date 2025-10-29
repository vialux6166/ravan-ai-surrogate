"""
Harmonic Oscillator Module for Ravan Quantum-ML System
Implements quantum harmonic oscillator simulation
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Dict, Any, Tuple, List
import numpy as np
from scipy.linalg import eigh
from simulation_base import SimulationModule, SimulationResult, PhysicsValidator


class HarmonicOscillator(SimulationModule):
    """
    Quantum Harmonic Oscillator Simulator
    
    Simulates quantum harmonic oscillator using ladder operators and
    coherent state evolution.
    
    Hamiltonian: H = p^2/2m + (1/2)m*omega^2*x^2
    Energy eigenvalues: E_n = (n + 1/2)*hbar*omega
    
    Observables:
    - energy_levels: Eigenvalues E_n
    - photon_number: Expected photon number <n>(t)
    - coherence: Coherence measure
    - position_variance: <x^2>
    """
    
    def __init__(self):
        """Initialize harmonic oscillator simulator"""
        super().__init__("harmonic_oscillator")
        self.logger.info("Harmonic oscillator simulator initialized")
    
    def get_parameter_space(self) -> Dict[str, Tuple[float, float]]:
        """
        Return valid parameter ranges
        
        Returns:
            Dictionary of parameter ranges
        """
        return {
            'oscillator_length': (0.5, 2.0),  # Characteristic length a (nm)
            'basis_size': (20, 50),           # Number of basis states N
            'initial_n': (0, 10),             # Initial photon number
            'n_time_points': (10, 100),       # Number of time points
            'max_time': (1.0, 10.0),          # Maximum time (periods)
        }

    def get_observable_names(self) -> List[str]:
        """
        Get list of observable names
        
        Returns:
            List of observable names
        """
        return ['energy_levels', 'photon_number', 'coherence', 'position_variance']

    def create_ladder_operators(self, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create ladder operators (creation and annihilation)
        
        a|n> = sqrt(n)|n-1>
        a_dagger|n> = sqrt(n+1)|n+1>
        
        Args:
            N: Basis size (number of states)
            
        Returns:
            Tuple of (a, a_dagger) operators
        """
        # Annihilation operator
        a = np.zeros((N, N), dtype=complex)
        for n in range(1, N):
            a[n-1, n] = np.sqrt(n)
        
        # Creation operator (Hermitian conjugate of a)
        a_dagger = a.conj().T
        
        return a, a_dagger
    
    def create_hamiltonian(
        self,
        N: int,
        omega: float = 1.0,
        hbar: float = 1.0
    ) -> np.ndarray:
        """
        Create harmonic oscillator Hamiltonian
        
        H = hbar*omega*(a_dagger*a + 1/2)
        
        Args:
            N: Basis size
            omega: Angular frequency (default: 1.0)
            hbar: Reduced Planck constant (default: 1.0)
            
        Returns:
            Hamiltonian matrix
        """
        a, a_dagger = self.create_ladder_operators(N)
        
        # Number operator: n = a_dagger * a
        n_op = a_dagger @ a
        
        # Hamiltonian: H = hbar*omega*(n + 1/2)
        H = hbar * omega * (n_op + 0.5 * np.eye(N))
        
        return H
    
    def create_coherent_state(
        self,
        alpha: complex,
        N: int
    ) -> np.ndarray:
        """
        Create coherent state |alpha>
        
        |alpha> = exp(-|alpha|^2/2) * sum_n (alpha^n / sqrt(n!)) |n>
        
        Args:
            alpha: Complex amplitude
            N: Basis size
            
        Returns:
            Coherent state vector
        """
        # Calculate coefficients
        coeffs = np.zeros(N, dtype=complex)
        
        # Normalization factor
        norm = np.exp(-0.5 * np.abs(alpha)**2)
        
        # Calculate each coefficient
        factorial = 1.0
        alpha_power = 1.0
        
        for n in range(N):
            if n > 0:
                factorial *= n
                alpha_power *= alpha
            
            coeffs[n] = norm * alpha_power / np.sqrt(factorial)
        
        return coeffs
    
    def create_fock_state(self, n: int, N: int) -> np.ndarray:
        """
        Create Fock state (number state) |n>
        
        Args:
            n: Photon number
            N: Basis size
            
        Returns:
            Fock state vector
        """
        state = np.zeros(N, dtype=complex)
        if n < N:
            state[n] = 1.0
        return state
    
    def time_evolve(
        self,
        psi0: np.ndarray,
        H: np.ndarray,
        times: np.ndarray,
        hbar: float = 1.0
    ) -> np.ndarray:
        """
        Time evolve state using Hamiltonian
        
        psi(t) = exp(-i*H*t/hbar) * psi(0)
        
        Args:
            psi0: Initial state
            H: Hamiltonian
            times: Array of time points
            hbar: Reduced Planck constant
            
        Returns:
            Array of states at each time point (n_times, N)
        """
        # Diagonalize Hamiltonian
        energies, eigenstates = eigh(H)
        
        # Project initial state onto eigenbasis
        coeffs = eigenstates.conj().T @ psi0
        
        # Time evolution
        psi_t = np.zeros((len(times), len(psi0)), dtype=complex)
        
        for i, t in enumerate(times):
            # Apply time evolution to each eigenstate
            phases = np.exp(-1j * energies * t / hbar)
            psi_t[i] = eigenstates @ (phases * coeffs)
        
        return psi_t
    
    def calculate_expectation(
        self,
        psi: np.ndarray,
        operator: np.ndarray
    ) -> float:
        """
        Calculate expectation value <psi|O|psi>
        
        Args:
            psi: State vector
            operator: Operator matrix
            
        Returns:
            Expectation value
        """
        return float(np.real(psi.conj() @ operator @ psi))
    
    def calculate_photon_number(self, psi: np.ndarray) -> float:
        """
        Calculate expected photon number <n>
        
        Args:
            psi: State vector
            
        Returns:
            Expected photon number
        """
        N = len(psi)
        n_values = np.arange(N)
        prob = np.abs(psi)**2
        return float(np.sum(n_values * prob))
    
    def calculate_coherence(self, psi: np.ndarray) -> float:
        """
        Calculate coherence measure (purity)
        
        Coherence = Tr(rho^2) where rho = |psi><psi|
        
        Args:
            psi: State vector
            
        Returns:
            Coherence measure (1 for pure state, <1 for mixed)
        """
        # For pure state, coherence = 1
        # Calculate as sum of |c_n|^4
        prob = np.abs(psi)**2
        coherence = float(np.sum(prob**2))
        return coherence
    
    def run(self, params: Dict[str, Any]) -> SimulationResult:
        """
        Execute harmonic oscillator simulation
        
        Args:
            params: Simulation parameters
                - oscillator_length: Characteristic length a (nm)
                - basis_size: Number of basis states N
                - initial_n: Initial photon number
                - n_time_points: Number of time points (default: 50)
                - max_time: Maximum time in periods (default: 5.0)
                
        Returns:
            SimulationResult with energy levels and dynamics
        """
        # Extract parameters with defaults
        a = float(params.get('oscillator_length', 1.0))
        N = int(params.get('basis_size', 30))
        initial_n = int(params.get('initial_n', 0))
        n_time_points = int(params.get('n_time_points', 50))
        max_time = float(params.get('max_time', 5.0))
        
        self.logger.debug(
            f"Running harmonic oscillator: a={a}, N={N}, n={initial_n}"
        )
        
        # Set units (hbar = omega = 1)
        hbar = 1.0
        omega = 1.0
        
        # Create Hamiltonian
        H = self.create_hamiltonian(N, omega, hbar)
        
        # Calculate energy eigenvalues
        energies, eigenstates = eigh(H)
        
        # Create initial state (Fock state)
        psi0 = self.create_fock_state(initial_n, N)
        
        # Time evolution
        times = np.linspace(0, max_time * 2 * np.pi / omega, n_time_points)
        psi_t = self.time_evolve(psi0, H, times, hbar)
        
        # Calculate observables over time
        photon_numbers = np.array([
            self.calculate_photon_number(psi) for psi in psi_t
        ])
        
        coherences = np.array([
            self.calculate_coherence(psi) for psi in psi_t
        ])
        
        # Calculate position operator and variance
        a_op, a_dagger = self.create_ladder_operators(N)
        x_op = (a_op + a_dagger) / np.sqrt(2)  # Position operator
        x2_op = x_op @ x_op  # x^2 operator
        
        position_variances = np.array([
            self.calculate_expectation(psi, x2_op) for psi in psi_t
        ])
        
        # Calculate energy level spacing
        energy_spacings = np.diff(energies[:min(10, N)])
        
        # Create observables
        observables = {
            'photon_number_mean': float(np.mean(photon_numbers)),
            'photon_number_std': float(np.std(photon_numbers)),
            'photon_number_initial': float(photon_numbers[0]),
            'photon_number_final': float(photon_numbers[-1]),
            # Legacy single-value keys expected by some tests
            'photon_number': float(np.mean(photon_numbers)),
            'coherence_mean': float(np.mean(coherences)),
            'position_variance_mean': float(np.mean(position_variances)),
            'energy_spacing_mean': float(np.mean(energy_spacings)),
            'energy_spacing_std': float(np.std(energy_spacings)),
            'energy_spacing': float(np.mean(energy_spacings)),
            'energy_0': float(energies[0])
        }
        
        # Add first few energy levels
        for i in range(min(5, N)):
            observables[f'energy_level_{i}'] = float(energies[i])
        
        # Add energy spacings
        for i in range(min(4, len(energy_spacings))):
            observables[f'energy_spacing_{i}'] = float(energy_spacings[i])
        
        metadata = {
            'basis_size': N,
            'n_time_points': n_time_points,
            'max_time': max_time,
            'omega': omega,
            'hbar': hbar,
            'time_resolution': float(times[1] - times[0]) if len(times) > 1 else 0.0
        }
        
        return SimulationResult(
            observables=observables,
            parameters=params,
            metadata=metadata
        )
    
    def validate_output(self, result: SimulationResult) -> bool:
        """
        Verify physics constraints for harmonic oscillator
        
        Args:
            result: Simulation result to validate
            
        Returns:
            True if validation passed
        """
        errors = []
        
        # Extract observables
        energy_spacing_mean = result.observables.get('energy_spacing_mean', 0)
        energy_spacing_std = result.observables.get('energy_spacing_std', 0)
        photon_initial = result.observables.get('photon_number_initial', 0)
        photon_final = result.observables.get('photon_number_final', 0)
        
        # Validation 1: Energy spacing should be uniform (E_{n+1} - E_n = 1.0)
        is_valid, error = PhysicsValidator.check_conservation(
            energy_spacing_mean, 1.0, tolerance=0.001,
            name="energy spacing"
        )
        if not is_valid:
            errors.append(error)
        
        # Validation 2: Energy spacing should have low variance
        if energy_spacing_std > 0.01:
            errors.append(
                f"Energy spacing variance {energy_spacing_std:.6f} too high (> 0.01)"
            )
        
        # Validation 3: Photon number should be conserved (within 0.01%)
        photon_change = abs(photon_final - photon_initial) / max(photon_initial, 1.0)
        if photon_change > 0.0001:
            errors.append(
                f"Photon number not conserved: {photon_change*100:.4f}% change (> 0.01%)"
            )
        
        # Validation 4: Coherence should be close to 1 (pure state)
        coherence = result.observables.get('coherence_mean', 0)
        if coherence < 0.99:
            errors.append(
                f"Coherence {coherence:.4f} too low (< 0.99), state not pure"
            )
        
        # Update result
        result.validation_passed = len(errors) == 0
        result.validation_errors = errors
        
        if not result.validation_passed:
            self.logger.warning(f"Validation failed: {errors}")
        
        return result.validation_passed

# Legacy alias for backward compatibility
HarmonicOscillatorModule = HarmonicOscillator
