"""
Ravan Quantum-ML System - Python API
Unified interface for quantum simulations, ML predictions, and LLM assistance
"""

from typing import Dict, Any, Optional, List, Union
import numpy as np
import logging

logger = logging.getLogger('ravan')


class Ravan:
    """
    Main Ravan API class
    
    Provides unified interface to:
    - Quantum simulators (circuit, Schrödinger, harmonic oscillator)
    - ML models (MLP, XGBoost)
    - LLM assistance (optional, requires GPU)
    
    Example:
        >>> ravan = Ravan(use_llm=True)
        >>> results = ravan.simulate("quantum tunneling with high barrier")
        >>> explanation = ravan.explain(results)
        >>> print(explanation)
    """
    
    def __init__(self, use_llm: bool = False, llm_config: Optional[Dict] = None):
        """
        Initialize Ravan system
        
        Args:
            use_llm: Enable LLM features (requires GPU with 20GB+ VRAM)
            llm_config: Optional LLM configuration dict
        """
        self.use_llm = use_llm
        self.llm = None
        self.llm_components = {}
        
        # Initialize LLM if requested
        if use_llm:
            self._init_llm(llm_config)
    
    def _init_llm(self, config: Optional[Dict] = None):
        """Initialize LLM and related components"""
        try:
            import sys
            import os
            
            # Add src/llm to path
            src_llm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'llm')
            if os.path.exists(src_llm_path):
                sys.path.insert(0, src_llm_path)
            
            from llm_adapter import LLMAdapter, LLMConfig
            from nl_interface import SimulationConfigGenerator
            from results_explainer import ResultsExplainer
            from code_generator import CodeGenerator
            
            logger.info("Initializing LLM (Qwen2.5-32B with 4-bit quantization)...")
            
            # Create config
            if config:
                llm_config = LLMConfig(**config)
            else:
                llm_config = LLMConfig()
            
            # Initialize adapter
            self.llm = LLMAdapter(llm_config)
            
            # Check VRAM
            vram_ok, free_vram = self.llm.check_vram_available(required_gb=20.0)
            if not vram_ok:
                logger.warning(f"Insufficient VRAM ({free_vram:.2f} GB). Using CPU (slower).")
            
            # Load model
            self.llm.load()
            logger.info("✓ LLM loaded successfully")
            
            # Initialize components
            self.llm_components = {
                'config_generator': SimulationConfigGenerator(self.llm),
                'explainer': ResultsExplainer(self.llm),
                'code_generator': CodeGenerator(self.llm)
            }
            
        except ImportError as e:
            logger.error(f"LLM dependencies not installed: {e}. Install with: pip install -r requirements-llm.txt")
            self.use_llm = False
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self.use_llm = False
    
    def simulate(
        self,
        config: Union[str, Dict[str, Any]],
        explain: bool = False
    ) -> Dict[str, Any]:
        """
        Run quantum simulation
        
        Args:
            config: Either natural language description (str) or parameter dict
            explain: Generate plain language explanation (requires LLM)
            
        Returns:
            Dictionary with simulation results
            
        Example:
            >>> # Natural language (requires LLM)
            >>> results = ravan.simulate("quantum tunneling with high barrier", explain=True)
            
            >>> # Parameter dict
            >>> results = ravan.simulate({
            ...     'simulator': 'schrodinger',
            ...     'V0': 5.0,
            ...     'k0': 4.0,
            ...     'barrier_width': 1.5
            ... })
        """
        # Parse config
        if isinstance(config, str):
            if not self.use_llm:
                raise ValueError("Natural language config requires LLM. Set use_llm=True")
            
            # Generate config from natural language
            generator = self.llm_components['config_generator']
            sim_config = generator.generate_simulation_config(config)
            simulator_type = sim_config['simulator']
            params = sim_config['parameters']
        else:
            simulator_type = config.get('simulator')
            params = {k: v for k, v in config.items() if k != 'simulator'}
        
        # Run simulation
        results = self._run_simulator(simulator_type, params)
        
        # Add explanation if requested
        if explain:
            if not self.use_llm:
                logger.warning("Explanation requires LLM. Skipping.")
            else:
                explainer = self.llm_components['explainer']
                explanation = explainer.explain_results(
                    simulator_type,
                    params,
                    results,
                    include_physics=True
                )
                results['explanation'] = explanation
        
        return results
    
    def _run_simulator(self, simulator_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run quantum simulator"""
        
        if simulator_type == 'quantum_circuit':
            from quantum_circuit_simulator import QuantumCircuitSimulator
            sim = QuantumCircuitSimulator()
        
        elif simulator_type == 'schrodinger':
            from schrodinger_solver import SchrodingerSolver
            sim = SchrodingerSolver()
        
        elif simulator_type == 'harmonic_oscillator' or simulator_type == 'harmonic':
            from harmonic_oscillator import HarmonicOscillator
            sim = HarmonicOscillator()
        
        else:
            raise ValueError(f"Unknown simulator: {simulator_type}")
        
        return sim.run(params)
    
    def train(
        self,
        dataset_path: str,
        model_type: str = 'mlp',
        output_path: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Train ML model on simulation data
        
        Args:
            dataset_path: Path to HDF5 dataset
            model_type: 'mlp' or 'xgboost'
            output_path: Path to save trained model
            **kwargs: Additional training parameters
            
        Returns:
            Trained model instance
            
        Example:
            >>> model = ravan.train(
            ...     'data/qc_dataset.h5',
            ...     model_type='mlp',
            ...     output_path='models/mlp_model'
            ... )
        """
        from hdf5_storage import HDF5Storage
        from model_base import ModelConfig
        
        # Load dataset
        storage = HDF5Storage(dataset_path)
        X_train, y_train = storage.load_training_data()
        
        logger.info(f"Loaded {len(X_train)} training samples")
        
        # Create model
        config = ModelConfig(
            model_type=model_type,
            input_dim=X_train.shape[1],
            output_dim=y_train.shape[1],
            save_path=output_path,
            hyperparameters=kwargs
        )
        
        if model_type == 'mlp':
            from mlp_regressor import MLPRegressor
            model = MLPRegressor(config)
        elif model_type == 'xgboost':
            from xgboost_regressor import XGBoostRegressor
            model = XGBoostRegressor(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train
        logger.info(f"Training {model_type} model...")
        model.train(X_train, y_train)
        
        # Save if path provided
        if output_path:
            model.save(output_path)
            logger.info(f"✓ Model saved to {output_path}")
        
        return model
    
    def predict(
        self,
        model_path: str,
        inputs: Union[str, np.ndarray, List],
        uncertainty: bool = False
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Make predictions with trained model
        
        Args:
            model_path: Path to trained model
            inputs: Input parameters (array, list, or natural language)
            uncertainty: Return uncertainty estimates (MC Dropout)
            
        Returns:
            Predictions array or dict with predictions and uncertainties
            
        Example:
            >>> # Array input
            >>> predictions = ravan.predict('models/mlp_model', [[2, 1000, 1, 1]])
            
            >>> # Natural language input (requires LLM)
            >>> predictions = ravan.predict(
            ...     'models/mlp_model',
            ...     "two qubits with bell state"
            ... )
        """
        from model_base import ModelConfig
        
        # Parse inputs
        if isinstance(inputs, str):
            if not self.use_llm:
                raise ValueError("Natural language input requires LLM")
            
            # Generate parameters from natural language
            generator = self.llm_components['config_generator']
            config = generator.generate_simulation_config(inputs)
            X = self._params_to_array(config['parameters'])
        else:
            X = np.array(inputs)
        
        # Load model
        if 'mlp' in model_path:
            from mlp_regressor import MLPRegressor
            model = MLPRegressor(ModelConfig(model_type='mlp'))
        elif 'xgboost' in model_path:
            from xgboost_regressor import XGBoostRegressor
            model = XGBoostRegressor(ModelConfig(model_type='xgboost'))
        else:
            raise ValueError("Cannot determine model type from path")
        
        model.load(model_path)
        
        # Predict
        predictions = model.predict(X)
        
        # Add uncertainty if requested
        if uncertainty:
            if hasattr(model, 'predict_with_uncertainty'):
                mean, std = model.predict_with_uncertainty(X)
                return {'predictions': mean, 'uncertainty': std}
            else:
                logger.warning("Model does not support uncertainty quantification")
                return {'predictions': predictions, 'uncertainty': None}
        
        return predictions
    
    def explain(
        self,
        results: Dict[str, Any],
        simulator_type: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate plain language explanation of results
        
        Args:
            results: Simulation results dict
            simulator_type: Type of simulator (auto-detected if not provided)
            params: Input parameters (optional)
            
        Returns:
            Plain language explanation
            
        Example:
            >>> results = ravan.simulate({'simulator': 'schrodinger', 'V0': 5.0})
            >>> explanation = ravan.explain(results)
            >>> print(explanation)
        """
        if not self.use_llm:
            raise ValueError("Explanation requires LLM. Set use_llm=True")
        
        # Auto-detect simulator type if not provided
        if simulator_type is None:
            simulator_type = results.get('simulator_type', 'unknown')
        
        # Extract params if not provided
        if params is None:
            params = results.get('parameters', {})
        
        explainer = self.llm_components['explainer']
        return explainer.explain_results(simulator_type, params, results)
    
    def generate_code(
        self,
        description: str,
        code_type: str = 'quantum_circuit',
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate Python/Qiskit code from description
        
        Args:
            description: Natural language description of code
            code_type: 'quantum_circuit', 'simulation', or 'analysis'
            save_path: Optional path to save generated code
            
        Returns:
            Generated Python code
            
        Example:
            >>> code = ravan.generate_code(
            ...     "Create a Bell state circuit with measurement",
            ...     code_type='quantum_circuit'
            ... )
            >>> print(code)
        """
        if not self.use_llm:
            raise ValueError("Code generation requires LLM. Set use_llm=True")
        
        generator = self.llm_components['code_generator']
        code = generator.generate_code(
            description,
            code_type=code_type,
            include_comments=True,
            include_example=True
        )
        
        # Save if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(code)
            logger.info(f"✓ Code saved to {save_path}")
        
        return code
    
    def query(self, question: str, max_tokens: int = 300) -> str:
        """
        Ask LLM a question about quantum mechanics
        
        Args:
            question: Question to ask
            max_tokens: Maximum response length
            
        Returns:
            LLM response
            
        Example:
            >>> answer = ravan.query("What is quantum tunneling?")
            >>> print(answer)
        """
        if not self.use_llm:
            raise ValueError("Query requires LLM. Set use_llm=True")
        
        return self.llm.query(question, max_new_tokens=max_tokens)
    
    def _params_to_array(self, params: Dict[str, Any]) -> np.ndarray:
        """Convert parameter dict to numpy array"""
        # This is simplified - adjust based on actual parameter order
        values = list(params.values())
        return np.array([values])
    
    def get_llm_info(self) -> Dict[str, Any]:
        """Get information about loaded LLM"""
        if not self.use_llm or self.llm is None:
            return {'enabled': False}
        
        return self.llm.get_model_info()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup LLM"""
        if self.llm:
            self.llm.unload()
    
    def __del__(self):
        """Destructor - ensure LLM cleanup"""
        if self.llm:
            self.llm.unload()


# Convenience functions

def simulate(config: Union[str, Dict], use_llm: bool = False, explain: bool = False) -> Dict:
    """
    Quick simulation without creating Ravan instance
    
    Example:
        >>> results = simulate("quantum tunneling", use_llm=True, explain=True)
    """
    with Ravan(use_llm=use_llm) as ravan:
        return ravan.simulate(config, explain=explain)


def train(dataset_path: str, model_type: str = 'mlp', **kwargs) -> Any:
    """
    Quick training without creating Ravan instance
    
    Example:
        >>> model = train('data/dataset.h5', model_type='mlp')
    """
    ravan = Ravan()
    return ravan.train(dataset_path, model_type, **kwargs)


def predict(model_path: str, inputs: Union[str, np.ndarray], use_llm: bool = False) -> np.ndarray:
    """
    Quick prediction without creating Ravan instance
    
    Example:
        >>> predictions = predict('models/mlp_model', [[2, 1000, 1, 1]])
    """
    with Ravan(use_llm=use_llm) as ravan:
        return ravan.predict(model_path, inputs)


def ask(question: str) -> str:
    """
    Quick LLM query without creating Ravan instance
    
    Example:
        >>> answer = ask("What is quantum entanglement?")
    """
    with Ravan(use_llm=True) as ravan:
        return ravan.query(question)
