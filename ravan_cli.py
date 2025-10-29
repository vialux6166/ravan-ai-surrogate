#!/usr/bin/env python3
"""
Ravan Quantum-ML System - Main CLI Application
Integrates quantum simulators, ML models, and optional LLM assistance
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ravan')


def setup_llm(use_llm: bool = False) -> Optional[Any]:
    """
    Setup LLM adapter if requested
    
    Args:
        use_llm: Whether to enable LLM features
        
    Returns:
        LLMAdapter instance or None
    """
    if not use_llm:
        return None
    
    try:
        from src.llm.llm_adapter import LLMAdapter, LLMConfig
        
        logger.info("Initializing LLM (Qwen2.5-32B with 4-bit quantization)...")
        config = LLMConfig()
        adapter = LLMAdapter(config)
        
        # Check VRAM
        vram_ok, free_vram = adapter.check_vram_available(required_gb=20.0)
        if not vram_ok:
            logger.warning(f"Insufficient VRAM ({free_vram:.2f} GB). LLM will use CPU (slower).")
        
        # Load model
        adapter.load()
        logger.info("✓ LLM loaded successfully")
        
        return adapter
    except ImportError:
        logger.error("LLM dependencies not installed. Install with: pip install -r requirements-llm.txt")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        return None


def simulate_command(args):
    """Run quantum simulation"""
    logger.info(f"Running {args.simulator} simulation...")
    
    # Setup LLM if requested
    llm = setup_llm(args.use_llm)
    
    # Parse parameters
    if args.nl_config and llm:
        # Use natural language to generate config
        from src.llm.nl_interface import SimulationConfigGenerator
        
        logger.info(f"Generating config from: '{args.nl_config}'")
        generator = SimulationConfigGenerator(llm)
        config = generator.generate_simulation_config(args.nl_config)
        
        simulator_type = config['simulator']
        params = config['parameters']
        
        logger.info(f"✓ Detected simulator: {simulator_type}")
        logger.info(f"✓ Generated parameters: {params}")
    else:
        # Use command-line parameters
        simulator_type = args.simulator
        params = parse_simulator_params(args)
    
    # Run simulation
    results = run_simulation(simulator_type, params)
    
    # Display results
    print("\n" + "="*70)
    print("SIMULATION RESULTS")
    print("="*70)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    
    # Explain results with LLM if available
    if llm and args.explain:
        from src.llm.results_explainer import ResultsExplainer
        
        logger.info("\nGenerating explanation...")
        explainer = ResultsExplainer(llm)
        explanation = explainer.explain_results(
            simulator_type,
            params,
            results,
            include_physics=True
        )
        
        print("\n" + "="*70)
        print("EXPLANATION")
        print("="*70)
        print(explanation)
    
    # Cleanup
    if llm:
        llm.unload()
    
    return results


def train_command(args):
    """Train ML models"""
    logger.info(f"Training {args.model} model on {args.dataset}...")
    
    # Setup LLM if requested
    llm = setup_llm(args.use_llm)
    
    # Load dataset
    from hdf5_storage import HDF5Storage
    
    storage = HDF5Storage(args.dataset)
    X_train, y_train = storage.load_training_data()
    
    logger.info(f"Loaded {len(X_train)} training samples")
    
    # Train model
    if args.model == 'mlp':
        from mlp_regressor import MLPRegressor
        from model_base import ModelConfig
        
        config = ModelConfig(
            model_type='mlp',
            input_dim=X_train.shape[1],
            output_dim=y_train.shape[1],
            save_path=args.output
        )
        model = MLPRegressor(config)
    elif args.model == 'xgboost':
        from xgboost_regressor import XGBoostRegressor
        from model_base import ModelConfig
        
        config = ModelConfig(
            model_type='xgboost',
            input_dim=X_train.shape[1],
            output_dim=y_train.shape[1],
            save_path=args.output
        )
        model = XGBoostRegressor(config)
    else:
        logger.error(f"Unknown model type: {args.model}")
        return
    
    # Train
    logger.info("Training model...")
    model.train(X_train, y_train)
    
    # Save
    model.save(args.output)
    logger.info(f"✓ Model saved to {args.output}")
    
    # Cleanup
    if llm:
        llm.unload()


def predict_command(args):
    """Make predictions with trained model"""
    logger.info(f"Loading model from {args.model_path}...")
    
    # Setup LLM if requested
    llm = setup_llm(args.use_llm)
    
    # Load model
    import numpy as np
    from model_base import ModelConfig
    
    if 'mlp' in args.model_path:
        from mlp_regressor import MLPRegressor
        model = MLPRegressor(ModelConfig(model_type='mlp'))
    elif 'xgboost' in args.model_path:
        from xgboost_regressor import XGBoostRegressor
        model = XGBoostRegressor(ModelConfig(model_type='xgboost'))
    else:
        logger.error("Cannot determine model type from path")
        return
    
    model.load(args.model_path)
    logger.info("✓ Model loaded")
    
    # Parse input parameters
    if args.nl_input and llm:
        # Use natural language to generate parameters
        from src.llm.nl_interface import SimulationConfigGenerator
        
        logger.info(f"Generating parameters from: '{args.nl_input}'")
        generator = SimulationConfigGenerator(llm)
        config = generator.generate_simulation_config(args.nl_input)
        
        # Convert to input array
        params = config['parameters']
        X = params_to_array(params)
    else:
        # Parse from command line
        X = np.array([args.params])
    
    # Predict
    logger.info("Making prediction...")
    predictions = model.predict(X)
    
    # Display results
    print("\n" + "="*70)
    print("PREDICTIONS")
    print("="*70)
    print(f"Input: {X[0]}")
    print(f"Output: {predictions[0]}")
    
    # Cleanup
    if llm:
        llm.unload()


def chat_command(args):
    """Interactive chat with LLM about quantum mechanics"""
    logger.info("Starting interactive chat mode...")
    
    # Setup LLM
    llm = setup_llm(use_llm=True)
    if llm is None:
        logger.error("Failed to initialize LLM. Chat mode requires LLM.")
        return
    
    print("\n" + "="*70)
    print("RAVAN QUANTUM ASSISTANT")
    print("="*70)
    print("Ask questions about quantum mechanics, simulations, or request code.")
    print("Commands: /help, /config, /code, /explain, /quit")
    print("="*70 + "\n")
    
    from src.llm.results_explainer import ResultsExplainer
    from src.llm.code_generator import CodeGenerator
    from src.llm.nl_interface import SimulationConfigGenerator
    
    explainer = ResultsExplainer(llm)
    code_gen = CodeGenerator(llm)
    config_gen = SimulationConfigGenerator(llm)
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input == '/quit':
                print("Goodbye!")
                break
            
            elif user_input == '/help':
                print("\nAvailable commands:")
                print("  /help    - Show this help")
                print("  /config  - Generate simulation config from description")
                print("  /code    - Generate Python/Qiskit code")
                print("  /explain - Explain simulation results")
                print("  /quit    - Exit chat")
                print("\nOr just ask any question about quantum mechanics!\n")
            
            elif user_input == '/config':
                desc = input("Describe simulation: ").strip()
                config = config_gen.generate_simulation_config(desc)
                print(f"\nGenerated config:")
                print(f"  Simulator: {config['simulator']}")
                print(f"  Parameters: {config['parameters']}\n")
            
            elif user_input == '/code':
                task = input("Describe code task: ").strip()
                code = code_gen.generate_code(task, code_type='quantum_circuit')
                print(f"\nGenerated code:\n{code}\n")
            
            elif user_input == '/explain':
                print("Feature coming soon: paste simulation results for explanation")
            
            else:
                # General query
                response = llm.query(user_input, max_new_tokens=300)
                print(f"\nAssistant: {response}\n")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
    
    # Cleanup
    llm.unload()


def generate_code_command(args):
    """Generate Python/Qiskit code from description"""
    logger.info("Generating code...")
    
    # Setup LLM
    llm = setup_llm(use_llm=True)
    if llm is None:
        logger.error("Code generation requires LLM.")
        return
    
    from src.llm.code_generator import CodeGenerator
    
    generator = CodeGenerator(llm)
    
    # Generate code
    code = generator.generate_code(
        args.description,
        code_type=args.code_type,
        include_comments=True,
        include_example=True
    )
    
    # Display
    print("\n" + "="*70)
    print("GENERATED CODE")
    print("="*70)
    print(code)
    print("="*70)
    
    # Save if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write(code)
        logger.info(f"✓ Code saved to {args.output}")
    
    # Cleanup
    llm.unload()


# Helper functions

def parse_simulator_params(args) -> Dict[str, Any]:
    """Parse simulator parameters from command line"""
    params = {}
    
    if args.simulator == 'quantum_circuit':
        params = {
            'n_qubits': args.n_qubits or 2,
            'shots': args.shots or 1000,
            'gate_sequence': args.gate_sequence or 'bell'
        }
    elif args.simulator == 'schrodinger':
        params = {
            'V0': args.V0 or 3.0,
            'barrier_width': args.barrier_width or 1.5,
            'k0': args.k0 or 4.0,
            'sigma': args.sigma or 1.0,
            'x0': args.x0 or -5.0
        }
    elif args.simulator == 'harmonic':
        params = {
            'oscillator_length': args.oscillator_length or 1.0,
            'basis_size': args.basis_size or 10,
            'initial_n': args.initial_n or 2
        }
    
    return params


def run_simulation(simulator_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Run quantum simulation"""
    
    if simulator_type == 'quantum_circuit':
        from quantum_circuit_simulator import QuantumCircuitSimulator
        
        sim = QuantumCircuitSimulator()
        results = sim.run(params)
    
    elif simulator_type == 'schrodinger':
        from schrodinger_solver import SchrodingerSolver
        
        sim = SchrodingerSolver()
        results = sim.run(params)
    
    elif simulator_type == 'harmonic':
        from harmonic_oscillator import HarmonicOscillator
        
        sim = HarmonicOscillator()
        results = sim.run(params)
    
    else:
        raise ValueError(f"Unknown simulator: {simulator_type}")
    
    return results


def params_to_array(params: Dict[str, Any]) -> Any:
    """Convert parameter dict to numpy array"""
    import numpy as np
    
    # This is a simplified version - adjust based on actual parameter order
    values = list(params.values())
    return np.array([values])


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Ravan Quantum-ML System - Quantum simulations with AI assistance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run simulation with natural language
  ravan simulate --nl-config "quantum tunneling with high barrier" --explain --use-llm
  
  # Run simulation with parameters
  ravan simulate --simulator schrodinger --V0 5.0 --k0 4.0 --explain
  
  # Train ML model
  ravan train --dataset data/qc_dataset.h5 --model mlp --output models/mlp_model
  
  # Make predictions
  ravan predict --model-path models/mlp_model --params 2 1000 1 1
  
  # Generate code
  ravan generate-code --description "Create Bell state circuit" --output bell.py
  
  # Interactive chat
  ravan chat
        """
    )
    
    # Global options
    parser.add_argument('--use-llm', action='store_true',
                       help='Enable LLM features (requires GPU)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Simulate command
    sim_parser = subparsers.add_parser('simulate', help='Run quantum simulation')
    sim_parser.add_argument('--simulator', choices=['quantum_circuit', 'schrodinger', 'harmonic'],
                           help='Simulator type')
    sim_parser.add_argument('--nl-config', type=str,
                           help='Natural language description of simulation (requires --use-llm)')
    sim_parser.add_argument('--explain', action='store_true',
                           help='Explain results in plain language (requires --use-llm)')
    
    # Quantum circuit params
    sim_parser.add_argument('--n-qubits', type=int, help='Number of qubits')
    sim_parser.add_argument('--shots', type=int, help='Number of measurements')
    sim_parser.add_argument('--gate-sequence', type=str, help='Gate sequence type')
    
    # Schrodinger params
    sim_parser.add_argument('--V0', type=float, help='Barrier height (eV)')
    sim_parser.add_argument('--barrier-width', type=float, help='Barrier width (nm)')
    sim_parser.add_argument('--k0', type=float, help='Initial momentum')
    sim_parser.add_argument('--sigma', type=float, help='Wavepacket width')
    sim_parser.add_argument('--x0', type=float, help='Initial position')
    
    # Harmonic params
    sim_parser.add_argument('--oscillator-length', type=float, help='Oscillator length')
    sim_parser.add_argument('--basis-size', type=int, help='Basis size')
    sim_parser.add_argument('--initial-n', type=int, help='Initial quantum number')
    
    sim_parser.set_defaults(func=simulate_command)
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train ML model')
    train_parser.add_argument('--dataset', required=True, help='Path to HDF5 dataset')
    train_parser.add_argument('--model', required=True, choices=['mlp', 'xgboost'],
                             help='Model type')
    train_parser.add_argument('--output', required=True, help='Output path for trained model')
    train_parser.set_defaults(func=train_command)
    
    # Predict command
    pred_parser = subparsers.add_parser('predict', help='Make predictions')
    pred_parser.add_argument('--model-path', required=True, help='Path to trained model')
    pred_parser.add_argument('--params', nargs='+', type=float,
                            help='Input parameters')
    pred_parser.add_argument('--nl-input', type=str,
                            help='Natural language input (requires --use-llm)')
    pred_parser.set_defaults(func=predict_command)
    
    # Generate code command
    code_parser = subparsers.add_parser('generate-code', help='Generate Python/Qiskit code')
    code_parser.add_argument('--description', required=True,
                            help='Description of code to generate')
    code_parser.add_argument('--code-type', default='quantum_circuit',
                            choices=['quantum_circuit', 'simulation', 'analysis'],
                            help='Type of code to generate')
    code_parser.add_argument('--output', help='Output file path')
    code_parser.set_defaults(func=generate_code_command)
    
    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Interactive chat with quantum assistant')
    chat_parser.set_defaults(func=chat_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Execute command
    if hasattr(args, 'func'):
        try:
            args.func(args)
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
