"""
Base Simulation Module for Ravan Quantum-ML System
Defines abstract interface for all quantum simulators
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass
import logging
import numpy as np


@dataclass
class SimulationResult:
    """
    Result from a simulation run
    
    Attributes:
        observables: Dictionary of observable names to values
        parameters: Input parameters used for simulation
        metadata: Additional metadata (timestamps, versions, etc.)
        validation_passed: Whether physics validation passed
        validation_errors: List of validation error messages
    """
    observables: Dict[str, float]
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]
    validation_passed: bool = True
    validation_errors: List[str] = None
    
    def __post_init__(self):
        if self.validation_errors is None:
            self.validation_errors = []

    # Dict-like access for compatibility with tests
    def __getitem__(self, key: str):
        return self.observables[key]

    def get(self, key: str, default=None):
        return self.observables.get(key, default)

    def keys(self):
        return self.observables.keys()

    def items(self):
        return self.observables.items()

    def __contains__(self, key: str) -> bool:
        return key in self.observables


class SimulationModule(ABC):
    """
    Abstract base class for quantum simulation modules
    
    All simulators (Quantum Circuit, Schrödinger, Harmonic Oscillator)
    must inherit from this class and implement the required methods.
    """
    
    def __init__(self, name: str):
        """
        Initialize simulation module
        
        Args:
            name: Name of the simulator
        """
        self.name = name
        self.logger = logging.getLogger(f'ravan.simulators.{name}')
        self.logger.info(f"{name} simulator initialized")
    
    @abstractmethod
    def run(self, params: Dict[str, Any]) -> SimulationResult:
        """
        Execute simulation with given parameters
        
        Args:
            params: Dictionary of simulation parameters
            
        Returns:
            SimulationResult with observables and metadata
            
        Raises:
            ValueError: If parameters are invalid
        """
        pass
    
    @abstractmethod
    def get_parameter_space(self) -> Dict[str, Tuple[float, float]]:
        """
        Return valid parameter ranges for sweeps
        
        Returns:
            Dictionary mapping parameter names to (min, max) tuples
            
        Example:
            {
                'V0': (1.0, 10.0),
                'barrier_width': (0.5, 2.0),
                'k0': (5.0, 15.0)
            }
        """
        pass
    
    @abstractmethod
    def validate_output(self, result: SimulationResult) -> bool:
        """
        Verify physics constraints (conservation laws, etc.)
        
        Args:
            result: Simulation result to validate
            
        Returns:
            True if validation passed, False otherwise
            
        Note:
            This method should update result.validation_passed and
            result.validation_errors if validation fails.
        """
        pass
    
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate input parameters against parameter space
        
        Args:
            params: Parameters to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        param_space = self.get_parameter_space()
        errors = []
        
        # Check for missing required parameters
        for param_name in param_space.keys():
            if param_name not in params:
                errors.append(f"Missing required parameter: {param_name}")
        
        # Check parameter ranges
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
    
    def run_with_validation(self, params: Dict[str, Any]) -> SimulationResult:
        """
        Run simulation with parameter and output validation
        
        Args:
            params: Simulation parameters
            
        Returns:
            SimulationResult with validation status
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate parameters
        is_valid, errors = self.validate_parameters(params)
        if not is_valid:
            raise ValueError(f"Invalid parameters: {', '.join(errors)}")
        
        # Run simulation
        self.logger.debug(f"Running simulation with params: {params}")
        result = self.run(params)
        
        # Validate output
        self.validate_output(result)
        
        if not result.validation_passed:
            self.logger.warning(
                f"Physics validation failed: {result.validation_errors}"
            )
        
        return result
    
    def get_observable_names(self) -> List[str]:
        """
        Get list of observable names produced by this simulator
        
        Returns:
            List of observable names
            
        Note:
            Subclasses should override this to provide specific observables
        """
        return []
    
    def get_parameter_names(self) -> List[str]:
        """
        Get list of parameter names required by this simulator
        
        Returns:
            List of parameter names
        """
        return list(self.get_parameter_space().keys())


class PhysicsValidator:
    """
    Utility class for common physics validation checks
    """
    
    @staticmethod
    def check_probability_normalization(
        probabilities: np.ndarray,
        tolerance: float = 0.001,
        name: str = "probabilities"
    ) -> Tuple[bool, str]:
        """
        Check if probabilities sum to 1.0
        
        Args:
            probabilities: Array of probabilities
            tolerance: Allowed deviation from 1.0
            name: Name for error message
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        total = np.sum(probabilities)
        is_valid = abs(total - 1.0) < tolerance
        
        if not is_valid:
            error = f"{name} sum to {total:.6f}, expected 1.0 ± {tolerance}"
            return False, error
        
        return True, ""
    
    @staticmethod
    def check_conservation(
        value: float,
        expected: float,
        tolerance: float = 0.001,
        name: str = "quantity"
    ) -> Tuple[bool, str]:
        """
        Check if a conserved quantity matches expected value
        
        Args:
            value: Actual value
            expected: Expected value
            tolerance: Allowed deviation
            name: Name for error message
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        is_valid = abs(value - expected) < tolerance
        
        if not is_valid:
            error = f"{name} = {value:.6f}, expected {expected:.6f} ± {tolerance}"
            return False, error
        
        return True, ""
    
    @staticmethod
    def check_range(
        value: float,
        min_val: float,
        max_val: float,
        name: str = "value"
    ) -> Tuple[bool, str]:
        """
        Check if value is within expected range
        
        Args:
            value: Value to check
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            name: Name for error message
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        is_valid = min_val <= value <= max_val
        
        if not is_valid:
            error = f"{name} = {value:.6f}, expected range [{min_val}, {max_val}]"
            return False, error
        
        return True, ""
