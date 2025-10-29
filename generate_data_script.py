#!/usr/bin/env python3
"""
Data Generation Script for Ravan Quantum-ML System
Generates training datasets from quantum simulations
"""

import sys
sys.path.insert(0, '/home/windows/ravan-quantum-ml/src')

import argparse
from pathlib import Path
import logging

from simulators.quantum_circuit import QuantumCircuitSimulator
from simulators.schrodinger import SchrodingerSolver
from simulators.harmonic import HarmonicOscillatorModule
from pipeline.parameter_sweep import ParameterSweepGenerator
from pipeline.parallel_executor import ParallelExecutor
from pipeline.dataset import Dataset
from pipeline.hdf5_storage import HDF5Storage
from utils.logging import setup_logging


def generate_dataset(
    simulator_type: str,
    n_samples: int,
    output_path: str,
    sampling_method: str = 'lhs',
    n_workers: int = None,
    random_seed: int = 42
):
    """
    Generate dataset from simulations
    
    Args:
        simulator_type: Type of simulator ('quantum_circuit', 'schrodinger', 'harmonic')
        n_samples: Number of samples to generate
        output_path: Output HDF5 file path
        sampling_method: Sampling method ('lhs', 'grid', 'random')
        n_workers: Number of parallel workers
        random_seed: Random seed for reproducibility
    """
    logger = logging.getLogger('ravan')
    
    # Initialize simulator
    logger.info(f"Initializing {simulator_type} simulator...")
    if simulator_type == 'quantum_circuit':
        simulator = QuantumCircuitSimulator(use_gpu=False)
    elif simulator_type == 'schrodinger':
        simulator = SchrodingerSolver(use_gpu=False)
    elif simulator_type == 'harmonic':
        simulator = HarmonicOscillatorModule()
    else:
        raise ValueError(f"Unknown simulator type: {simulator_type}")
    
    # Get parameter space
    parameter_space = simulator.get_parameter_space()
    logger.info(f"Parameter space: {list(parameter_space.keys())}")
    
    # Generate parameter samples
    logger.info(f"Generating {n_samples} parameter samples using {sampling_method}...")
    sweep_gen = ParameterSweepGenerator(random_seed=random_seed)
    
    if sampling_method == 'lhs':
        parameter_samples = sweep_gen.latin_hypercube_sampling(
            parameter_space, n_samples
        )
    elif sampling_method == 'grid':
        n_points = int(n_samples ** (1/len(parameter_space)))
        parameter_samples = sweep_gen.grid_search(
            parameter_space, n_points_per_dim=n_points
        )
    elif sampling_method == 'random':
        parameter_samples = sweep_gen.random_sampling(
            parameter_space, n_samples
        )
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")
    
    # Calculate coverage metrics
    coverage = sweep_gen.get_coverage_metrics(parameter_samples, parameter_space)
    logger.info(f"Parameter space coverage: {coverage['overall_coverage']:.2%}")
    
    # Execute simulations in parallel
    logger.info(f"Executing {len(parameter_samples)} simulations...")
    executor = ParallelExecutor(n_workers=n_workers)
    results = executor.execute_simulations(
        simulator,
        parameter_samples,
        show_progress=True,
        error_handling='skip'
    )
    
    # Filter valid results
    valid_results = executor.filter_valid_results(results)
    logger.info(f"Valid results: {len(valid_results)}/{len(results)}")
    
    # Convert to arrays
    parameter_names = simulator.get_parameter_names()
    observable_names = simulator.get_observable_names()
    
    X, y = executor.results_to_arrays(
        valid_results,
        parameter_names,
        observable_names
    )
    
    # Create dataset
    metadata = {
        'simulator_type': simulator_type,
        'sampling_method': sampling_method,
        'random_seed': random_seed,
        'n_requested': n_samples,
        'n_valid': len(valid_results),
        'coverage': coverage
    }
    
    dataset = Dataset(
        X=X,
        y=y,
        parameter_names=parameter_names,
        observable_names=observable_names,
        metadata=metadata
    )
    
    # Check data quality
    quality = dataset.check_data_quality()
    logger.info(f"Data quality: {quality['n_issues']} issues found")
    
    # Save to HDF5
    logger.info(f"Saving dataset to {output_path}...")
    storage = HDF5Storage()
    storage.save_dataset(dataset, output_path)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Dataset Generation Complete!")
    logger.info("=" * 60)
    logger.info(dataset.summary())
    logger.info("=" * 60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Generate training dataset from quantum simulations'
    )
    
    parser.add_argument(
        '--simulator',
        type=str,
        required=True,
        choices=['quantum_circuit', 'schrodinger', 'harmonic'],
        help='Type of simulator to use'
    )
    
    parser.add_argument(
        '--n-samples',
        type=int,
        default=500,
        help='Number of samples to generate (default: 500)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output HDF5 file path'
    )
    
    parser.add_argument(
        '--sampling',
        type=str,
        default='lhs',
        choices=['lhs', 'grid', 'random'],
        help='Sampling method (default: lhs)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: all CPUs)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    # Generate dataset
    generate_dataset(
        simulator_type=args.simulator,
        n_samples=args.n_samples,
        output_path=args.output,
        sampling_method=args.sampling,
        n_workers=args.workers,
        random_seed=args.seed
    )


if __name__ == '__main__':
    main()
