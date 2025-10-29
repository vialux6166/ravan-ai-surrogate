"""
Natural Language Interface for Ravan Quantum-ML System
Task 16.1: Implement simulation config generation

Converts natural language descriptions into simulation parameters
"""

import sys
import os
import json
import re
from typing import Dict, Any, Optional, List
import logging

from src.llm.llm_adapter import LLMAdapter


class SimulationConfigGenerator:
    """
    Generates simulation configurations from natural language descriptions
    
    Supports:
    - Quantum circuit simulations
    - Schrödinger equation solver
    - Harmonic oscillator
    """
    
    def __init__(self, llm_adapter: LLMAdapter):
        """
        Initialize config generator
        
        Args:
            llm_adapter: LLMAdapter instance for NL processing
        """
        self.llm = llm_adapter
        self.logger = logging.getLogger('ravan.nl_interface')
        
        # Parameter ranges for validation
        self.parameter_ranges = {
            'quantum_circuit': {
                'n_qubits': (2, 10),
                'shots': (100, 100000),
                'gate_sequence': ['bell', 'ghz', 'w_state', 'custom']
            },
            'schrodinger': {
                'V0': (0.5, 10.0),
                'barrier_width': (0.5, 5.0),
                'k0': (1.0, 10.0),
                'sigma': (0.5, 3.0),
                'x0': (-10.0, -3.0)
            },
            'harmonic_oscillator': {
                'oscillator_length': (0.3, 3.0),
                'basis_size': (5, 30),
                'initial_n': (0, 10)
            }
        }
    
    def generate_simulation_config(
        self,
        description: str,
        simulator_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate simulation configuration from natural language
        
        Args:
            description: Natural language description of simulation
            simulator_type: Optional simulator type hint
                          ('quantum_circuit', 'schrodinger', 'harmonic_oscillator')
            
        Returns:
            Dictionary with simulation parameters
            
        Example:
            >>> gen.generate_simulation_config(
            ...     "Run quantum tunneling with high barrier and narrow wavepacket"
            ... )
            {
                'simulator': 'schrodinger',
                'parameters': {
                    'V0': 8.0,
                    'barrier_width': 1.5,
                    'k0': 5.0,
                    'sigma': 0.8,
                    'x0': -5.0
                }
            }
        """
        self.logger.info(f"Generating config from: {description}")
        
        # Detect simulator type if not provided
        if simulator_type is None:
            simulator_type = self._detect_simulator_type(description)
        
        self.logger.info(f"Detected simulator: {simulator_type}")
        
        # Generate parameters using LLM
        parameters = self._generate_parameters(description, simulator_type)
        
        # Validate parameters
        validated_params = self._validate_parameters(parameters, simulator_type)
        
        config = {
            'simulator': simulator_type,
            'parameters': validated_params,
            'description': description
        }
        
        self.logger.info(f"Generated config: {config}")
        return config
    
    def _detect_simulator_type(self, description: str) -> str:
        """
        Detect which simulator to use based on description
        
        Args:
            description: Natural language description
            
        Returns:
            Simulator type string
        """
        description_lower = description.lower()
        
        # Keywords for each simulator
        quantum_circuit_keywords = [
            'qubit', 'entangle', 'bell', 'ghz', 'superposition',
            'measurement', 'quantum gate', 'cnot', 'hadamard'
        ]
        
        schrodinger_keywords = [
            'tunnel', 'barrier', 'wavepacket', 'transmission',
            'reflection', 'potential', 'schrödinger', 'schrodinger'
        ]
        
        harmonic_keywords = [
            'harmonic', 'oscillator', 'photon', 'coherent',
            'energy level', 'ladder operator', 'phonon'
        ]
        
        # Count keyword matches
        qc_score = sum(1 for kw in quantum_circuit_keywords if kw in description_lower)
        sch_score = sum(1 for kw in schrodinger_keywords if kw in description_lower)
        ho_score = sum(1 for kw in harmonic_keywords if kw in description_lower)
        
        # Return simulator with highest score
        scores = {
            'quantum_circuit': qc_score,
            'schrodinger': sch_score,
            'harmonic_oscillator': ho_score
        }
        
        detected = max(scores, key=scores.get)
        
        # Default to quantum_circuit if no clear match
        if scores[detected] == 0:
            self.logger.warning("No clear simulator detected, defaulting to quantum_circuit")
            return 'quantum_circuit'
        
        return detected
    
    def _generate_parameters(
        self,
        description: str,
        simulator_type: str
    ) -> Dict[str, Any]:
        """
        Generate parameters using LLM
        
        Args:
            description: Natural language description
            simulator_type: Type of simulator
            
        Returns:
            Dictionary of parameters
        """
        # Create prompt for LLM
        prompt = self._create_parameter_prompt(description, simulator_type)
        
        # Query LLM
        response = self.llm.query(prompt, max_new_tokens=300, temperature=0.3)
        
        # Parse response
        parameters = self._parse_llm_response(response, simulator_type)
        
        return parameters
    
    def _create_parameter_prompt(
        self,
        description: str,
        simulator_type: str
    ) -> str:
        """
        Create prompt for LLM to generate parameters
        
        Args:
            description: User description
            simulator_type: Simulator type
            
        Returns:
            Formatted prompt
        """
        if simulator_type == 'quantum_circuit':
            prompt = f"""You are a quantum computing expert. Convert this description into quantum circuit parameters.

Description: {description}

Generate parameters in JSON format with these fields:
- n_qubits: number of qubits (2-10)
- shots: number of measurements (100-100000)
- gate_sequence: type of circuit ('bell', 'ghz', 'w_state')

Respond ONLY with valid JSON, no explanation:
{{
  "n_qubits": <number>,
  "shots": <number>,
  "gate_sequence": "<type>"
}}"""
        
        elif simulator_type == 'schrodinger':
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
- Low barrier: V0 < 3.0
- Narrow wavepacket: sigma < 1.0
- Wide wavepacket: sigma > 2.0
- High energy: k0 > 6.0

Respond ONLY with valid JSON, no explanation:
{{
  "V0": <number>,
  "barrier_width": <number>,
  "k0": <number>,
  "sigma": <number>,
  "x0": <number>
}}"""
        
        elif simulator_type == 'harmonic_oscillator':
            prompt = f"""You are a quantum mechanics expert. Convert this description into harmonic oscillator parameters.

Description: {description}

Generate parameters in JSON format with these fields:
- oscillator_length: characteristic length (0.3-3.0)
- basis_size: number of basis states (5-30)
- initial_n: initial quantum number (0-10)

Guidelines:
- Ground state: initial_n = 0
- Excited state: initial_n > 0
- High accuracy: basis_size > 15
- Fast computation: basis_size < 12

Respond ONLY with valid JSON, no explanation:
{{
  "oscillator_length": <number>,
  "basis_size": <number>,
  "initial_n": <number>
}}"""
        
        else:
            raise ValueError(f"Unknown simulator type: {simulator_type}")
        
        return prompt
    
    def _parse_llm_response(
        self,
        response: str,
        simulator_type: str
    ) -> Dict[str, Any]:
        """
        Parse LLM response to extract parameters
        
        Args:
            response: LLM response text
            simulator_type: Simulator type
            
        Returns:
            Dictionary of parameters
        """
        # Try to find JSON in response
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(0)
            try:
                parameters = json.loads(json_str)
                return parameters
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON parse error: {e}")
        
        # Fallback: use default parameters
        self.logger.warning("Could not parse LLM response, using defaults")
        return self._get_default_parameters(simulator_type)
    
    def _get_default_parameters(self, simulator_type: str) -> Dict[str, Any]:
        """
        Get default parameters for simulator
        
        Args:
            simulator_type: Simulator type
            
        Returns:
            Default parameters
        """
        defaults = {
            'quantum_circuit': {
                'n_qubits': 2,
                'shots': 1000,
                'gate_sequence': 'bell'
            },
            'schrodinger': {
                'V0': 3.0,
                'barrier_width': 1.5,
                'k0': 4.0,
                'sigma': 1.0,
                'x0': -5.0
            },
            'harmonic_oscillator': {
                'oscillator_length': 1.0,
                'basis_size': 10,
                'initial_n': 2
            }
        }
        
        return defaults.get(simulator_type, {})
    
    def _validate_parameters(
        self,
        parameters: Dict[str, Any],
        simulator_type: str
    ) -> Dict[str, Any]:
        """
        Validate and clamp parameters to valid ranges
        
        Args:
            parameters: Generated parameters
            simulator_type: Simulator type
            
        Returns:
            Validated parameters
        """
        if simulator_type not in self.parameter_ranges:
            return parameters
        
        ranges = self.parameter_ranges[simulator_type]
        validated = {}
        
        for param, value in parameters.items():
            if param not in ranges:
                validated[param] = value
                continue
            
            range_spec = ranges[param]
            
            # Handle list of valid values
            if isinstance(range_spec, list):
                if value not in range_spec:
                    self.logger.warning(
                        f"Invalid {param}={value}, using default {range_spec[0]}"
                    )
                    validated[param] = range_spec[0]
                else:
                    validated[param] = value
            
            # Handle numeric ranges
            elif isinstance(range_spec, tuple):
                min_val, max_val = range_spec
                
                # Clamp to range
                if value < min_val:
                    self.logger.warning(
                        f"{param}={value} below minimum {min_val}, clamping"
                    )
                    validated[param] = min_val
                elif value > max_val:
                    self.logger.warning(
                        f"{param}={value} above maximum {max_val}, clamping"
                    )
                    validated[param] = max_val
                else:
                    validated[param] = value
            else:
                validated[param] = value
        
        return validated
    
    def batch_generate_configs(
        self,
        descriptions: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple configs from descriptions
        
        Args:
            descriptions: List of natural language descriptions
            
        Returns:
            List of configuration dictionaries
        """
        configs = []
        
        for desc in descriptions:
            try:
                config = self.generate_simulation_config(desc)
                configs.append(config)
            except Exception as e:
                self.logger.error(f"Error generating config for '{desc}': {e}")
                configs.append(None)
        
        return configs
    
    def explain_config(self, config: Dict[str, Any]) -> str:
        """
        Generate human-readable explanation of config
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Plain language explanation
        """
        simulator = config['simulator']
        params = config['parameters']
        
        if simulator == 'quantum_circuit':
            return (
                f"Quantum circuit with {params['n_qubits']} qubits, "
                f"{params['shots']} measurements, "
                f"using {params['gate_sequence']} gate sequence"
            )
        
        elif simulator == 'schrodinger':
            return (
                f"Quantum tunneling simulation with barrier height {params['V0']:.1f} eV, "
                f"width {params['barrier_width']:.1f} nm, "
                f"wavepacket momentum {params['k0']:.1f}, "
                f"width {params['sigma']:.1f}"
            )
        
        elif simulator == 'harmonic_oscillator':
            return (
                f"Harmonic oscillator with length {params['oscillator_length']:.1f}, "
                f"{params['basis_size']} basis states, "
                f"initial state n={params['initial_n']}"
            )
        
        return "Unknown configuration"


# =============================================================================
# Convenience Functions
# =============================================================================

def create_config_generator(llm_adapter: LLMAdapter) -> SimulationConfigGenerator:
    """
    Create config generator with LLM adapter
    
    Args:
        llm_adapter: LLMAdapter instance
        
    Returns:
        SimulationConfigGenerator instance
    """
    return SimulationConfigGenerator(llm_adapter)


def quick_config(description: str, llm_adapter: LLMAdapter) -> Dict[str, Any]:
    """
    Quick config generation
    
    Args:
        description: Natural language description
        llm_adapter: LLMAdapter instance
        
    Returns:
        Configuration dictionary
    """
    generator = SimulationConfigGenerator(llm_adapter)
    return generator.generate_simulation_config(description)
