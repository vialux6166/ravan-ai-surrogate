"""
Parameter Sweep Generator for Ravan Quantum-ML System
Implements Latin Hypercube Sampling and grid search
"""

import sys
sys.path.insert(0, '/home/windows/ravan-quantum-ml/src')

from typing import Dict, Tuple, List
import numpy as np
from scipy.stats import qmc
import logging


class ParameterSweepGenerator:
    """
    Generate parameter combinations for simulation sweeps
    
    Supports:
    - Latin Hypercube Sampling (LHS) for efficient space coverage
    - Grid search for systematic exploration
    - Random sampling
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize parameter sweep generator
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.logger = logging.getLogger('ravan.parameter_sweep')
        np.random.seed(random_seed)
    
    def latin_hypercube_sampling(
        self,
        parameter_space: Dict[str, Tuple[float, float]],
        n_samples: int,
        centered: bool = True
    ) -> List[Dict[str, float]]:
        """
        Generate samples using Latin Hypercube Sampling
        
        LHS ensures good coverage of parameter space with fewer samples
        than grid search.
        
        Args:
            parameter_space: Dict mapping parameter names to (min, max) tuples
            n_samples: Number of samples to generate
            centered: Whether to center samples in LHS cells
            
        Returns:
            List of parameter dictionaries
        """
        param_names = list(parameter_space.keys())
        n_params = len(param_names)
        
        self.logger.info(
            f"Generating {n_samples} samples using LHS "
            f"for {n_params} parameters"
        )
        
        # Create Latin Hypercube sampler
        sampler = qmc.LatinHypercube(
            d=n_params,
            centered=centered,
            seed=self.random_seed
        )
        
        # Generate samples in [0, 1]^d
        samples_unit = sampler.random(n=n_samples)
        
        # Scale to parameter ranges
        samples = []
        for sample_unit in samples_unit:
            param_dict = {}
            for i, param_name in enumerate(param_names):
                min_val, max_val = parameter_space[param_name]
                # Scale from [0, 1] to [min_val, max_val]
                value = min_val + sample_unit[i] * (max_val - min_val)
                param_dict[param_name] = float(value)
            samples.append(param_dict)
        
        self.logger.info(f"Generated {len(samples)} LHS samples")
        return samples
    
    def grid_search(
        self,
        parameter_space: Dict[str, Tuple[float, float]],
        n_points_per_dim: int = 10
    ) -> List[Dict[str, float]]:
        """
        Generate samples using grid search
        
        Creates a regular grid over the parameter space.
        Warning: Number of samples grows exponentially with dimensions!
        
        Args:
            parameter_space: Dict mapping parameter names to (min, max) tuples
            n_points_per_dim: Number of points per dimension
            
        Returns:
            List of parameter dictionaries
        """
        param_names = list(parameter_space.keys())
        n_params = len(param_names)
        
        total_samples = n_points_per_dim ** n_params
        
        self.logger.info(
            f"Generating grid search with {n_points_per_dim} points/dim "
            f"for {n_params} parameters (total: {total_samples} samples)"
        )
        
        if total_samples > 100000:
            self.logger.warning(
                f"Grid search will generate {total_samples} samples! "
                f"Consider using LHS instead."
            )
        
        # Create grid for each parameter
        grids = []
        for param_name in param_names:
            min_val, max_val = parameter_space[param_name]
            grid = np.linspace(min_val, max_val, n_points_per_dim)
            grids.append(grid)
        
        # Create meshgrid
        mesh = np.meshgrid(*grids, indexing='ij')
        
        # Flatten and combine
        samples = []
        for i in range(total_samples):
            indices = np.unravel_index(i, mesh[0].shape)
            param_dict = {}
            for j, param_name in enumerate(param_names):
                param_dict[param_name] = float(mesh[j][indices])
            samples.append(param_dict)
        
        self.logger.info(f"Generated {len(samples)} grid samples")
        return samples
    
    def random_sampling(
        self,
        parameter_space: Dict[str, Tuple[float, float]],
        n_samples: int
    ) -> List[Dict[str, float]]:
        """
        Generate random samples (uniform distribution)
        
        Args:
            parameter_space: Dict mapping parameter names to (min, max) tuples
            n_samples: Number of samples to generate
            
        Returns:
            List of parameter dictionaries
        """
        param_names = list(parameter_space.keys())
        
        self.logger.info(
            f"Generating {n_samples} random samples "
            f"for {len(param_names)} parameters"
        )
        
        samples = []
        for _ in range(n_samples):
            param_dict = {}
            for param_name in param_names:
                min_val, max_val = parameter_space[param_name]
                value = np.random.uniform(min_val, max_val)
                param_dict[param_name] = float(value)
            samples.append(param_dict)
        
        self.logger.info(f"Generated {len(samples)} random samples")
        return samples
    
    def stratified_sampling(
        self,
        parameter_space: Dict[str, Tuple[float, float]],
        n_samples: int,
        n_strata: int = 5
    ) -> List[Dict[str, float]]:
        """
        Generate stratified samples
        
        Divides parameter space into strata and samples from each.
        
        Args:
            parameter_space: Dict mapping parameter names to (min, max) tuples
            n_samples: Number of samples to generate
            n_strata: Number of strata per dimension
            
        Returns:
            List of parameter dictionaries
        """
        param_names = list(parameter_space.keys())
        n_params = len(param_names)
        
        self.logger.info(
            f"Generating {n_samples} stratified samples "
            f"with {n_strata} strata per dimension"
        )
        
        samples = []
        samples_per_stratum = max(1, n_samples // (n_strata ** n_params))
        
        # Create strata boundaries
        strata_bounds = {}
        for param_name in param_names:
            min_val, max_val = parameter_space[param_name]
            bounds = np.linspace(min_val, max_val, n_strata + 1)
            strata_bounds[param_name] = bounds
        
        # Sample from each stratum
        for _ in range(samples_per_stratum):
            for stratum_idx in range(n_strata ** n_params):
                if len(samples) >= n_samples:
                    break
                
                # Convert linear index to multi-dimensional stratum indices
                indices = np.unravel_index(stratum_idx, [n_strata] * n_params)
                
                param_dict = {}
                for i, param_name in enumerate(param_names):
                    bounds = strata_bounds[param_name]
                    min_val = bounds[indices[i]]
                    max_val = bounds[indices[i] + 1]
                    value = np.random.uniform(min_val, max_val)
                    param_dict[param_name] = float(value)
                
                samples.append(param_dict)
        
        # Trim to exact number of samples
        samples = samples[:n_samples]
        
        self.logger.info(f"Generated {len(samples)} stratified samples")
        return samples
    
    def validate_parameters(
        self,
        samples: List[Dict[str, float]],
        parameter_space: Dict[str, Tuple[float, float]]
    ) -> Tuple[List[Dict[str, float]], List[int]]:
        """
        Validate parameter samples against parameter space
        
        Args:
            samples: List of parameter dictionaries
            parameter_space: Valid parameter ranges
            
        Returns:
            Tuple of (valid_samples, invalid_indices)
        """
        valid_samples = []
        invalid_indices = []
        
        for i, sample in enumerate(samples):
            is_valid = True
            
            for param_name, value in sample.items():
                if param_name in parameter_space:
                    min_val, max_val = parameter_space[param_name]
                    if not (min_val <= value <= max_val):
                        is_valid = False
                        break
            
            if is_valid:
                valid_samples.append(sample)
            else:
                invalid_indices.append(i)
        
        if invalid_indices:
            self.logger.warning(
                f"Found {len(invalid_indices)} invalid samples out of {len(samples)}"
            )
        
        return valid_samples, invalid_indices
    
    def get_coverage_metrics(
        self,
        samples: List[Dict[str, float]],
        parameter_space: Dict[str, Tuple[float, float]]
    ) -> Dict[str, float]:
        """
        Calculate coverage metrics for parameter samples
        
        Args:
            samples: List of parameter dictionaries
            parameter_space: Parameter ranges
            
        Returns:
            Dictionary with coverage metrics
        """
        param_names = list(parameter_space.keys())
        
        # Convert to array
        X = np.array([[s[name] for name in param_names] for s in samples])
        
        # Calculate metrics
        metrics = {}
        
        for i, param_name in enumerate(param_names):
            min_val, max_val = parameter_space[param_name]
            values = X[:, i]
            
            # Coverage: fraction of range covered
            actual_min = values.min()
            actual_max = values.max()
            coverage = (actual_max - actual_min) / (max_val - min_val)
            
            metrics[f'{param_name}_coverage'] = float(coverage)
            metrics[f'{param_name}_mean'] = float(values.mean())
            metrics[f'{param_name}_std'] = float(values.std())
        
        # Overall coverage (geometric mean)
        coverages = [v for k, v in metrics.items() if k.endswith('_coverage')]
        metrics['overall_coverage'] = float(np.prod(coverages) ** (1/len(coverages)))
        
        return metrics
