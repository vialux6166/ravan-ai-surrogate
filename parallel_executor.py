"""
Parallel Simulation Executor for Ravan Quantum-ML System
Executes simulations in parallel using multiprocessing
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import List, Dict, Any, Callable, Optional
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import logging
import time

from simulation_base import SimulationModule, SimulationResult


class ParallelExecutor:
    """
    Execute simulations in parallel
    
    Uses multiprocessing to parallelize simulation execution
    across multiple CPU cores.
    """
    
    def __init__(self, n_workers: Optional[int] = None):
        """
        Initialize parallel executor
        
        Args:
            n_workers: Number of worker processes (None = use all CPUs)
        """
        self.n_workers = n_workers or cpu_count()
        self.logger = logging.getLogger('ravan.parallel')
        self.logger.info(f"Parallel executor initialized with {self.n_workers} workers")
    
    def execute_simulations(
        self,
        simulator: SimulationModule,
        parameter_samples: List[Dict[str, Any]],
        show_progress: bool = True,
        error_handling: str = 'skip'
    ) -> List[SimulationResult]:
        """
        Execute simulations in parallel
        
        Args:
            simulator: Simulator instance
            parameter_samples: List of parameter dictionaries
            show_progress: Whether to show progress bar
            error_handling: How to handle errors ('skip', 'raise', 'log')
            
        Returns:
            List of simulation results
        """
        n_samples = len(parameter_samples)
        self.logger.info(
            f"Executing {n_samples} simulations with {self.n_workers} workers"
        )
        
        start_time = time.time()
        
        # Create worker function
        def worker(params):
            try:
                result = simulator.run_with_validation(params)
                return result, None
            except Exception as e:
                return None, str(e)
        
        # Execute in parallel
        results = []
        errors = []
        
        with Pool(processes=self.n_workers) as pool:
            if show_progress:
                iterator = tqdm(
                    pool.imap(worker, parameter_samples),
                    total=n_samples,
                    desc="Simulating"
                )
            else:
                iterator = pool.imap(worker, parameter_samples)
            
            for i, (result, error) in enumerate(iterator):
                if error is not None:
                    errors.append((i, error))
                    
                    if error_handling == 'raise':
                        raise RuntimeError(f"Simulation {i} failed: {error}")
                    elif error_handling == 'log':
                        self.logger.error(f"Simulation {i} failed: {error}")
                    # 'skip' - just continue
                else:
                    results.append(result)
        
        elapsed_time = time.time() - start_time
        
        self.logger.info(
            f"Completed {len(results)} simulations in {elapsed_time:.2f}s "
            f"({len(results)/elapsed_time:.2f} samples/s)"
        )
        
        if errors:
            self.logger.warning(f"Failed simulations: {len(errors)}/{n_samples}")
        
        return results
    
    def execute_batch(
        self,
        simulator: SimulationModule,
        parameter_samples: List[Dict[str, Any]],
        batch_size: int = 100,
        show_progress: bool = True
    ) -> List[SimulationResult]:
        """
        Execute simulations in batches
        
        Useful for very large datasets to manage memory.
        
        Args:
            simulator: Simulator instance
            parameter_samples: List of parameter dictionaries
            batch_size: Number of samples per batch
            show_progress: Whether to show progress bar
            
        Returns:
            List of simulation results
        """
        n_samples = len(parameter_samples)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        self.logger.info(
            f"Executing {n_samples} simulations in {n_batches} batches "
            f"of size {batch_size}"
        )
        
        all_results = []
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_params = parameter_samples[start_idx:end_idx]
            
            self.logger.info(
                f"Processing batch {batch_idx+1}/{n_batches} "
                f"({len(batch_params)} samples)"
            )
            
            batch_results = self.execute_simulations(
                simulator,
                batch_params,
                show_progress=show_progress,
                error_handling='skip'
            )
            
            all_results.extend(batch_results)
        
        self.logger.info(f"Completed all batches: {len(all_results)} total results")
        return all_results
    
    def results_to_arrays(
        self,
        results: List[SimulationResult],
        parameter_names: List[str],
        observable_names: List[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert simulation results to numpy arrays
        
        Args:
            results: List of simulation results
            parameter_names: List of parameter names
            observable_names: List of observable names
            
        Returns:
            Tuple of (X, y) arrays
        """
        n_samples = len(results)
        n_features = len(parameter_names)
        n_targets = len(observable_names)
        
        X = np.zeros((n_samples, n_features))
        y = np.zeros((n_samples, n_targets))
        
        for i, result in enumerate(results):
            # Extract parameters
            for j, param_name in enumerate(parameter_names):
                X[i, j] = result.parameters.get(param_name, 0.0)
            
            # Extract observables
            for j, obs_name in enumerate(observable_names):
                y[i, j] = result.observables.get(obs_name, 0.0)
        
        return X, y
    
    def filter_valid_results(
        self,
        results: List[SimulationResult]
    ) -> List[SimulationResult]:
        """
        Filter results to keep only validated ones
        
        Args:
            results: List of simulation results
            
        Returns:
            List of valid results
        """
        valid_results = [r for r in results if r.validation_passed]
        
        n_invalid = len(results) - len(valid_results)
        if n_invalid > 0:
            self.logger.warning(
                f"Filtered out {n_invalid} invalid results "
                f"({n_invalid/len(results)*100:.1f}%)"
            )
        
        return valid_results
    
    def identify_edge_cases(
        self,
        results: List[SimulationResult],
        threshold: float = 0.1
    ) -> List[int]:
        """
        Identify edge cases based on validation errors
        
        Args:
            results: List of simulation results
            threshold: Threshold for edge case detection
            
        Returns:
            List of edge case indices
        """
        edge_case_indices = []
        
        for i, result in enumerate(results):
            if not result.validation_passed:
                edge_case_indices.append(i)
            elif len(result.validation_errors) > 0:
                # Has warnings but passed
                edge_case_indices.append(i)
        
        if edge_case_indices:
            self.logger.info(
                f"Identified {len(edge_case_indices)} edge cases "
                f"({len(edge_case_indices)/len(results)*100:.1f}%)"
            )
        
        return edge_case_indices
