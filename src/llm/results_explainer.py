"""
Results Explanation Module for Ravan Quantum-ML System
Task 16.2: Implement results explanation

Generates plain language explanations of simulation results with physics context
"""

import sys
import os
from typing import Dict, Any, List, Optional
import logging
import numpy as np

from src.llm.llm_adapter import LLMAdapter


class ResultsExplainer:
    """
    Generates human-readable explanations of simulation results
    
    Provides physics context and interprets observables in plain language
    """
    
    def __init__(self, llm_adapter: LLMAdapter):
        """
        Initialize results explainer
        
        Args:
            llm_adapter: LLMAdapter instance for NL generation
        """
        self.llm = llm_adapter
        self.logger = logging.getLogger('ravan.results_explainer')
    
    def explain_results(
        self,
        simulator_type: str,
        parameters: Dict[str, Any],
        results: Dict[str, Any],
        include_physics: bool = True
    ) -> str:
        """
        Generate plain language explanation of simulation results
        
        Args:
            simulator_type: Type of simulator ('quantum_circuit', 'schrodinger', 'harmonic_oscillator')
            parameters: Input parameters used
            results: Simulation output observables
            include_physics: Whether to include physics context
            
        Returns:
            Plain language explanation string
            
        Example:
            >>> explainer.explain_results(
            ...     'schrodinger',
            ...     {'V0': 5.0, 'k0': 4.0},
            ...     {'transmission': 0.65, 'reflection': 0.35}
            ... )
            "The quantum tunneling simulation shows 65% transmission through
             the barrier. This indicates significant tunneling probability..."
        """
        self.logger.info(f"Generating explanation for {simulator_type} results")
        
        # Create prompt for LLM
        prompt = self._create_explanation_prompt(
            simulator_type,
            parameters,
            results,
            include_physics
        )
        
        # Query LLM
        explanation = self.llm.query(prompt, max_new_tokens=400, temperature=0.7)
        
        # Clean up response
        explanation = self._clean_explanation(explanation)
        
        return explanation
    
    def _create_explanation_prompt(
        self,
        simulator_type: str,
        parameters: Dict[str, Any],
        results: Dict[str, Any],
        include_physics: bool
    ) -> str:
        """
        Create prompt for LLM to generate explanation
        
        Args:
            simulator_type: Simulator type
            parameters: Input parameters
            results: Output results
            include_physics: Include physics context
            
        Returns:
            Formatted prompt
        """
        if simulator_type == 'quantum_circuit':
            return self._create_quantum_circuit_prompt(parameters, results, include_physics)
        elif simulator_type == 'schrodinger':
            return self._create_schrodinger_prompt(parameters, results, include_physics)
        elif simulator_type == 'harmonic_oscillator':
            return self._create_harmonic_oscillator_prompt(parameters, results, include_physics)
        else:
            raise ValueError(f"Unknown simulator type: {simulator_type}")
    
    def _create_quantum_circuit_prompt(
        self,
        parameters: Dict[str, Any],
        results: Dict[str, Any],
        include_physics: bool
    ) -> str:
        """Create prompt for quantum circuit results"""
        
        physics_context = ""
        if include_physics:
            physics_context = """
Include physics context:
- Explain what entropy means for quantum states
- Discuss entanglement and superposition
- Relate measurements to quantum mechanics principles
"""
        
        prompt = f"""You are a quantum computing expert. Explain these quantum circuit simulation results in plain language.

**Simulation Parameters:**
- Number of qubits: {parameters.get('n_qubits', 'N/A')}
- Measurements: {parameters.get('shots', 'N/A')}
- Gate sequence: {parameters.get('gate_sequence', 'N/A')}

**Results:**
- Entropy: {results.get('entropy', 'N/A'):.4f} bits
- Chi-squared: {results.get('chi_squared', 'N/A'):.4f}
- KL divergence: {results.get('kl_divergence', 'N/A'):.6f}

{physics_context}

Provide a 2-3 sentence explanation that:
1. Describes what the results mean
2. Interprets the entropy value
3. Explains the significance for quantum computing

Write in clear, accessible language for researchers."""
        
        return prompt
    
    def _create_schrodinger_prompt(
        self,
        parameters: Dict[str, Any],
        results: Dict[str, Any],
        include_physics: bool
    ) -> str:
        """Create prompt for Schrödinger solver results"""
        
        physics_context = ""
        if include_physics:
            physics_context = """
Include physics context:
- Explain quantum tunneling phenomenon
- Discuss how barrier height affects transmission
- Relate to wave-particle duality
- Mention applications (STM, alpha decay, etc.)
"""
        
        prompt = f"""You are a quantum mechanics expert. Explain these quantum tunneling simulation results in plain language.

**Simulation Parameters:**
- Barrier height (V₀): {parameters.get('V0', 'N/A'):.2f} eV
- Barrier width: {parameters.get('barrier_width', 'N/A'):.2f} nm
- Initial momentum (k₀): {parameters.get('k0', 'N/A'):.2f}
- Wavepacket width (σ): {parameters.get('sigma', 'N/A'):.2f}

**Results:**
- Transmission probability: {results.get('transmission', 'N/A'):.4f} ({results.get('transmission', 0)*100:.1f}%)
- Reflection probability: {results.get('reflection', 'N/A'):.4f} ({results.get('reflection', 0)*100:.1f}%)
- Total probability: {results.get('total_probability', 'N/A'):.6f}

{physics_context}

Provide a 2-3 sentence explanation that:
1. Describes the tunneling behavior
2. Interprets the transmission/reflection ratio
3. Explains what this means physically

Write in clear, accessible language for researchers."""
        
        return prompt
    
    def _create_harmonic_oscillator_prompt(
        self,
        parameters: Dict[str, Any],
        results: Dict[str, Any],
        include_physics: bool
    ) -> str:
        """Create prompt for harmonic oscillator results"""
        
        physics_context = ""
        if include_physics:
            physics_context = """
Include physics context:
- Explain energy quantization
- Discuss coherent states and photon statistics
- Relate to quantum optics applications
- Mention zero-point energy
"""
        
        prompt = f"""You are a quantum optics expert. Explain these harmonic oscillator simulation results in plain language.

**Simulation Parameters:**
- Oscillator length: {parameters.get('oscillator_length', 'N/A'):.2f}
- Basis size: {parameters.get('basis_size', 'N/A')}
- Initial quantum number: {parameters.get('initial_n', 'N/A')}

**Results:**
- Ground state energy (E₀): {results.get('energy_0', 'N/A'):.4f}
- Energy spacing (ΔE): {results.get('energy_spacing', 'N/A'):.4f}
- Photon number ⟨n⟩: {results.get('photon_number', 'N/A'):.4f}

{physics_context}

Provide a 2-3 sentence explanation that:
1. Describes the energy level structure
2. Interprets the photon number
3. Explains the physical significance

Write in clear, accessible language for researchers."""
        
        return prompt
    
    def _clean_explanation(self, explanation: str) -> str:
        """
        Clean up LLM response
        
        Args:
            explanation: Raw LLM output
            
        Returns:
            Cleaned explanation
        """
        # Remove common LLM artifacts
        explanation = explanation.strip()
        
        # Remove "Here is..." or "The explanation is..." prefixes
        prefixes_to_remove = [
            "Here is the explanation:",
            "Here's the explanation:",
            "The explanation is:",
            "Explanation:",
        ]
        
        for prefix in prefixes_to_remove:
            if explanation.startswith(prefix):
                explanation = explanation[len(prefix):].strip()
        
        return explanation
    
    def explain_comparison(
        self,
        simulator_type: str,
        results_list: List[Dict[str, Any]],
        labels: Optional[List[str]] = None
    ) -> str:
        """
        Compare multiple simulation results
        
        Args:
            simulator_type: Type of simulator
            results_list: List of result dictionaries
            labels: Optional labels for each result
            
        Returns:
            Comparative explanation
        """
        if labels is None:
            labels = [f"Run {i+1}" for i in range(len(results_list))]
        
        # Create comparison prompt
        prompt = self._create_comparison_prompt(simulator_type, results_list, labels)
        
        # Query LLM
        explanation = self.llm.query(prompt, max_new_tokens=500, temperature=0.7)
        
        return self._clean_explanation(explanation)
    
    def _create_comparison_prompt(
        self,
        simulator_type: str,
        results_list: List[Dict[str, Any]],
        labels: List[str]
    ) -> str:
        """Create prompt for comparing results"""
        
        results_text = ""
        for label, results in zip(labels, results_list):
            results_text += f"\n**{label}:**\n"
            for key, value in results.items():
                if isinstance(value, float):
                    results_text += f"  - {key}: {value:.4f}\n"
                else:
                    results_text += f"  - {key}: {value}\n"
        
        prompt = f"""You are a quantum physics expert. Compare these {simulator_type} simulation results and explain the differences.

{results_text}

Provide a 3-4 sentence comparison that:
1. Identifies key differences between the runs
2. Explains what causes these differences
3. Discusses the physical implications
4. Suggests which configuration is more interesting/useful

Write in clear, accessible language."""
        
        return prompt
    
    def explain_trend(
        self,
        simulator_type: str,
        parameter_name: str,
        parameter_values: List[float],
        observable_name: str,
        observable_values: List[float]
    ) -> str:
        """
        Explain trend in observable vs parameter
        
        Args:
            simulator_type: Type of simulator
            parameter_name: Name of varied parameter
            parameter_values: Parameter values
            observable_name: Name of observable
            observable_values: Observable values
            
        Returns:
            Trend explanation
        """
        # Create trend prompt
        prompt = f"""You are a quantum physics expert. Explain this trend in {simulator_type} simulation results.

**Parameter varied:** {parameter_name}
**Values:** {parameter_values}

**Observable measured:** {observable_name}
**Values:** {observable_values}

Provide a 2-3 sentence explanation that:
1. Describes the trend (increasing, decreasing, non-monotonic)
2. Explains the physical reason for this behavior
3. Relates it to quantum mechanics principles

Write in clear, accessible language."""
        
        # Query LLM
        explanation = self.llm.query(prompt, max_new_tokens=400, temperature=0.7)
        
        return self._clean_explanation(explanation)
    
    def suggest_next_experiment(
        self,
        simulator_type: str,
        parameters: Dict[str, Any],
        results: Dict[str, Any]
    ) -> str:
        """
        Suggest next experiment based on results
        
        Args:
            simulator_type: Type of simulator
            parameters: Current parameters
            results: Current results
            
        Returns:
            Suggestion for next experiment
        """
        prompt = f"""You are a quantum physics researcher. Based on these {simulator_type} simulation results, suggest what to explore next.

**Current Parameters:**
{self._format_dict(parameters)}

**Current Results:**
{self._format_dict(results)}

Provide a 2-3 sentence suggestion that:
1. Identifies an interesting aspect of the results
2. Suggests a specific parameter to vary
3. Explains what new insight this would provide

Write in clear, actionable language."""
        
        # Query LLM
        suggestion = self.llm.query(prompt, max_new_tokens=300, temperature=0.8)
        
        return self._clean_explanation(suggestion)
    
    def _format_dict(self, d: Dict[str, Any]) -> str:
        """Format dictionary for prompt"""
        lines = []
        for key, value in d.items():
            if isinstance(value, float):
                lines.append(f"  - {key}: {value:.4f}")
            else:
                lines.append(f"  - {key}: {value}")
        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_results_explainer(llm_adapter: LLMAdapter) -> ResultsExplainer:
    """
    Create results explainer with LLM adapter
    
    Args:
        llm_adapter: LLMAdapter instance
        
    Returns:
        ResultsExplainer instance
    """
    return ResultsExplainer(llm_adapter)


def quick_explain(
    simulator_type: str,
    parameters: Dict[str, Any],
    results: Dict[str, Any],
    llm_adapter: LLMAdapter
) -> str:
    """
    Quick explanation generation
    
    Args:
        simulator_type: Simulator type
        parameters: Input parameters
        results: Output results
        llm_adapter: LLMAdapter instance
        
    Returns:
        Explanation string
    """
    explainer = ResultsExplainer(llm_adapter)
    return explainer.explain_results(simulator_type, parameters, results)
