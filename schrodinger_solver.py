"""
Schrödinger Equation Solver for Ravan Quantum-ML System
Implements 1D time-dependent Schrödinger equation solver for quantum tunneling
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Dict, Any, Tuple, List
import numpy as np
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = np

from simulation_base import SimulationModule, SimulationResult, PhysicsValidator


class SchrodingerSolver(SimulationModule):
    """
    1D Time-Dependent Schrödinger Equation Solver
    
    Solves the Schrödinger equation for a wavepacket scattering off a
    potential barrier using the split-operator method.
    
    Equation: i*hbar d(psi)/dt = H*psi, where H = -hbar^2/2m nabla^2 + V(x)
    
    Observables:
    - transmission: Transmission coefficient T
    - reflection: Reflection coefficient R
    - total_probability: Total probability (should be 1.0)
    - energy: Average energy
    """
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize Schrödinger solver
        
        Args:
            use_gpu: Whether to use GPU acceleration with CuPy (default: False)
                    Note: CPU (NumPy) is faster for small FFT problems due to lower overhead
        """
        super().__init__("schrodinger")
        # Force CPU for optimal performance on small grids
        self.use_gpu = False
        self.xp = np
        self.logger.info("Using NumPy (CPU) for Schrödinger solver - optimal for small FFTs")
    
    def get_parameter_space(self) -> Dict[str, Tuple[float, float]]:
        """
        Return valid parameter ranges
        
        Returns:
            Dictionary of parameter ranges
        """
        return {
            'V0': (1.0, 10.0),          # Barrier height (eV)
            'barrier_width': (0.5, 2.0), # Barrier width (nm)
            'k0': (5.0, 15.0),          # Initial momentum (1/nm)
            'sigma': (0.5, 1.5),        # Wavepacket width (nm)
            'x0': (-10.0, -5.0),        # Initial position (nm)
            'L': (40.0, 60.0),          # System size (nm)
            'Nx': (512, 2048),          # Grid points
            'dt': (0.001, 0.01),        # Time step
            'n_steps': (500, 2000),     # Number of time steps
        }
    
    def get_observable_names(self) -> List[str]:
        """
        Get list of observable names
        
        Returns:
            List of observable names
        """
        return ['transmission', 'reflection', 'total_probability', 'energy']
    
    def create_potential_barrier(
        self,
        x,  # Can be np.ndarray or cp.ndarray
        V0: float,
        barrier_width: float,
        barrier_center: float = 0.0
    ):
        """
        Create rectangular potential barrier V(x)
        
        Args:
            x: Position array (NumPy or CuPy)
            V0: Barrier height
            barrier_width: Width of barrier
            barrier_center: Center position of barrier
            
        Returns:
            Potential array V(x) (same type as input)
        """
        xp = self.xp
        
        # Rectangular barrier
        V = xp.zeros_like(x)
        barrier_left = barrier_center - barrier_width / 2
        barrier_right = barrier_center + barrier_width / 2
        
        mask = (x >= barrier_left) & (x <= barrier_right)
        V[mask] = V0
        
        return V
    
    def create_gaussian_wavepacket(
        self,
        x,  # Can be np.ndarray or cp.ndarray
        x0: float,
        k0: float,
        sigma: float
    ):
        """
        Create Gaussian wavepacket ψ(x,0)
        
        psi(x,0) = (2*pi*sigma^2)^(-1/4) exp(i*k0*x) exp(-(x-x0)^2/4*sigma^2)
        
        Args:
            x: Position array (NumPy or CuPy)
            x0: Initial center position
            k0: Initial momentum (wavenumber)
            sigma: Wavepacket width
            
        Returns:
            Initial wavefunction ψ(x,0) (same type as input)
        """
        xp = self.xp
        
        # Normalization factor
        norm = (2 * xp.pi * sigma**2) ** (-0.25)
        
        # Gaussian envelope with momentum
        psi = norm * xp.exp(1j * k0 * x) * xp.exp(-(x - x0)**2 / (4 * sigma**2))
        
        return psi
    
    def split_operator_step(
        self,
        psi,  # Can be np.ndarray or cp.ndarray
        V,    # Can be np.ndarray or cp.ndarray
        k,    # Can be np.ndarray or cp.ndarray
        dt: float,
        hbar: float = 1.0,
        m: float = 1.0
    ):
        """
        Perform one time step using split-operator method
        
        The split-operator method splits the evolution operator:
        exp(-i*H*t/hbar) = exp(-i*V*t/2*hbar) exp(-i*T*t/hbar) exp(-i*V*t/2*hbar)
        
        Args:
            psi: Wavefunction in position space
            V: Potential in position space
            k: Momentum array
            dt: Time step
            hbar: Reduced Planck constant (default: 1.0)
            m: Particle mass (default: 1.0)
            
        Returns:
            Updated wavefunction
        """
        xp = self.xp
        
        # Half step in position space (potential)
        psi = psi * xp.exp(-1j * V * dt / (2 * hbar))
        
        # Full step in momentum space (kinetic)
        psi_k = xp.fft.fft(psi)
        psi_k = psi_k * xp.exp(-1j * hbar * k**2 * dt / (2 * m))
        psi = xp.fft.ifft(psi_k)
        
        # Half step in position space (potential)
        psi = psi * xp.exp(-1j * V * dt / (2 * hbar))
        
        return psi
    
    def calculate_transmission_reflection(
        self,
        psi,  # Can be np.ndarray or cp.ndarray
        x,    # Can be np.ndarray or cp.ndarray
        barrier_center: float = 0.0
    ) -> Tuple[float, float]:
        """
        Calculate transmission and reflection coefficients
        
        Args:
            psi: Final wavefunction
            x: Position array
            barrier_center: Center of barrier
            
        Returns:
            Tuple of (transmission, reflection)
        """
        xp = self.xp
        
        # Probability density
        prob_density = xp.abs(psi)**2
        
        # Integrate probability on left (reflection) and right (transmission)
        left_mask = x < barrier_center
        right_mask = x > barrier_center
        
        R = float(xp.sum(prob_density[left_mask]))
        T = float(xp.sum(prob_density[right_mask]))
        
        # Normalize to ensure R + T = 1
        total = R + T
        if total > 0:
            R /= total
            T /= total
        
        return T, R
    
    def run(self, params: Dict[str, Any]) -> SimulationResult:
        """
        Execute Schrödinger equation simulation
        
        Args:
            params: Simulation parameters
                - V0: Barrier height (eV)
                - barrier_width: Barrier width (nm)
                - k0: Initial momentum (1/nm)
                - sigma: Wavepacket width (nm)
                - x0: Initial position (nm)
                - L: System size (nm, default: 50.0)
                - Nx: Grid points (default: 1024)
                - dt: Time step (default: 0.005)
                - n_steps: Number of time steps (default: 1000)
                
        Returns:
            SimulationResult with transmission and reflection coefficients
        """
        xp = self.xp
        
        # Extract parameters with defaults
        V0 = float(params.get('V0', 5.0))
        barrier_width = float(params.get('barrier_width', 1.0))
        k0 = float(params.get('k0', 10.0))
        sigma = float(params.get('sigma', 1.0))
        x0 = float(params.get('x0', -8.0))
        L = float(params.get('L', 50.0))
        # Use same grid size for fair comparison
        Nx = int(params.get('Nx', 1024))
        dt = float(params.get('dt', 0.005))
        n_steps = int(params.get('n_steps', 1000))
        hbar = 1.0
        m = 1.0
        
        self.logger.debug(
            f"Running Schrödinger: V0={V0}, width={barrier_width}, "
            f"k0={k0}, sigma={sigma}"
        )
        
        # Create spatial grid (CPU only for optimal performance)
        x = np.linspace(-L/2, L/2, Nx)
        dx = x[1] - x[0]
        
        # Create momentum grid for FFT
        k = 2 * np.pi * np.fft.fftfreq(Nx, dx)
        
        # Create potential barrier
        V = self.create_potential_barrier(x, V0, barrier_width)
        
        # Create initial wavepacket
        psi = self.create_gaussian_wavepacket(x, x0, k0, sigma)
        
        # PRE-COMPUTE exponential operators (avoid recomputing in loop)
        V_exp_half = xp.exp(-1j * V * dt / (2 * hbar))
        k_exp = xp.exp(-1j * hbar * k**2 * dt / (2 * m))
        
        # Calculate initial energy (only once, before loop)
        psi_k = xp.fft.fft(psi)
        kinetic = float(xp.sum(xp.abs(psi_k)**2 * k**2) / (2 * Nx))
        potential = float(xp.sum(xp.abs(psi)**2 * V) * dx)
        initial_energy = kinetic + potential
        
        # Time evolution using split-operator method (OPTIMIZED)
        for step in range(n_steps):
            # Half step in position space (potential)
            psi *= V_exp_half
            
            # Full step in momentum space (kinetic)
            psi_k = xp.fft.fft(psi)
            psi_k *= k_exp
            psi = xp.fft.ifft(psi_k)
            
            # Half step in position space (potential)
            psi *= V_exp_half
        
        # Calculate final observables (minimize GPU->CPU transfers)
        T, R = self.calculate_transmission_reflection(psi, x)
        
        # Calculate total probability
        prob_density = xp.abs(psi)**2
        total_prob = float(xp.sum(prob_density) * dx)
        
        # Calculate final energy
        psi_k = xp.fft.fft(psi)
        kinetic_final = float(xp.sum(xp.abs(psi_k)**2 * k**2) / (2 * Nx))
        potential_final = float(xp.sum(xp.abs(psi)**2 * V) * dx)
        final_energy = kinetic_final + potential_final
        
        # Create result
        observables = {
            'transmission': T,
            'reflection': R,
            'total_probability': total_prob,
            'energy': final_energy,
            'initial_energy': initial_energy,
            'energy_change': abs(final_energy - initial_energy) / initial_energy
        }
        
        metadata = {
            'Nx': Nx,
            'n_steps': n_steps,
            'dt': dt,
            'dx': float(dx),
            'total_time': dt * n_steps,
            'backend': 'CuPy (GPU)' if self.use_gpu else 'NumPy (CPU)'
        }
        
        return SimulationResult(
            observables=observables,
            parameters=params,
            metadata=metadata
        )
    
    def validate_output(self, result: SimulationResult) -> bool:
        """
        Verify physics constraints for Schrödinger solver
        
        Args:
            result: Simulation result to validate
            
        Returns:
            True if validation passed
        """
        errors = []
        
        # Extract observables
        T = result.observables.get('transmission', 0)
        R = result.observables.get('reflection', 0)
        total_prob = result.observables.get('total_probability', 0)
        
        # Validation 1: T + R should equal 1.0 (within tolerance)
        is_valid, error = PhysicsValidator.check_conservation(
            T + R, 1.0, tolerance=0.001, name="T + R"
        )
        if not is_valid:
            errors.append(error)
        
        # Validation 2: Total probability should be conserved
        is_valid, error = PhysicsValidator.check_conservation(
            total_prob, 1.0, tolerance=0.01, name="total probability"
        )
        if not is_valid:
            errors.append(error)
        
        # Validation 3: T and R should be in [0, 1]
        is_valid, error = PhysicsValidator.check_range(
            T, 0.0, 1.0, name="transmission"
        )
        if not is_valid:
            errors.append(error)
        
        is_valid, error = PhysicsValidator.check_range(
            R, 0.0, 1.0, name="reflection"
        )
        if not is_valid:
            errors.append(error)
        
        # Validation 4: Energy should be approximately conserved (within 1%)
        energy_change = result.observables.get('energy_change', 0)
        if energy_change > 0.01:
            errors.append(
                f"Energy not conserved: {energy_change*100:.2f}% change (> 1%)"
            )
        
        # Update result
        result.validation_passed = len(errors) == 0
        result.validation_errors = errors
        
        if not result.validation_passed:
            self.logger.warning(f"Validation failed: {errors}")
        
        return result.validation_passed
