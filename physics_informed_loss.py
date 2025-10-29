"""
Physics-Informed Loss Functions for Ravan Quantum-ML System
Enforces physics constraints during neural network training
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhysicsInformedLoss(nn.Module):
    """
    Physics-informed loss function for quantum ML models
    
    Combines standard MSE loss with physics constraint penalties to ensure
    predictions satisfy known physical laws (conservation, quantization, etc.)
    """
    
    def __init__(
        self,
        base_loss: nn.Module = None,
        constraint_weight: float = 0.1,
        constraints: Optional[List[str]] = None
    ):
        """
        Initialize physics-informed loss
        
        Args:
            base_loss: Base loss function (default: MSE)
            constraint_weight: Weight for constraint penalty term
            constraints: List of constraint types to apply
                        ['schrodinger', 'harmonic_oscillator', 'probability_norm']
        """
        super(PhysicsInformedLoss, self).__init__()
        
        self.base_loss = base_loss if base_loss is not None else nn.MSELoss()
        self.constraint_weight = constraint_weight
        self.constraints = constraints if constraints is not None else []
        
        logger.info(
            f"PhysicsInformedLoss initialized: "
            f"constraint_weight={constraint_weight}, "
            f"constraints={self.constraints}"
        )
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        observable_names: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Compute physics-informed loss
        
        Args:
            predictions: Model predictions (batch_size, n_observables)
            targets: Ground truth values (batch_size, n_observables)
            observable_names: Names of observables (for identifying which constraints to apply)
            metadata: Additional metadata (e.g., simulator type)
            
        Returns:
            Total loss (base_loss + constraint_penalty)
        """
        # Base MSE loss
        mse_loss = self.base_loss(predictions, targets)
        
        # Initialize constraint penalty
        constraint_penalty = torch.tensor(0.0, device=predictions.device)
        
        # Apply constraints based on observable names or metadata
        if observable_names is not None:
            # Schrödinger constraint: R + T = 1
            if 'schrodinger' in self.constraints:
                penalty = self._schrodinger_constraint(predictions, observable_names)
                if penalty is not None:
                    constraint_penalty += penalty
            
            # Harmonic oscillator constraint: uniform energy spacing
            if 'harmonic_oscillator' in self.constraints:
                penalty = self._harmonic_oscillator_constraint(predictions, observable_names)
                if penalty is not None:
                    constraint_penalty += penalty
            
            # Probability normalization constraint
            if 'probability_norm' in self.constraints:
                penalty = self._probability_norm_constraint(predictions, observable_names)
                if penalty is not None:
                    constraint_penalty += penalty
        
        # Total loss
        total_loss = mse_loss + self.constraint_weight * constraint_penalty
        
        return total_loss
    
    def _schrodinger_constraint(
        self,
        predictions: torch.Tensor,
        observable_names: List[str]
    ) -> Optional[torch.Tensor]:
        """
        Schrödinger equation constraint: R + T = 1
        
        For quantum tunneling, transmission (T) + reflection (R) must equal 1.
        
        Args:
            predictions: Model predictions
            observable_names: Names of observables
            
        Returns:
            Constraint penalty or None if observables not found
        """
        try:
            # Find transmission and reflection indices
            transmission_idx = None
            reflection_idx = None
            
            for i, name in enumerate(observable_names):
                if 'transmission' in name.lower():
                    transmission_idx = i
                elif 'reflection' in name.lower():
                    reflection_idx = i
            
            if transmission_idx is not None and reflection_idx is not None:
                T = predictions[:, transmission_idx]
                R = predictions[:, reflection_idx]
                
                # Penalty: mean((R + T - 1)²)
                penalty = torch.mean((R + T - 1.0) ** 2)
                
                return penalty
            
        except Exception as e:
            logger.debug(f"Could not apply Schrödinger constraint: {e}")
        
        return None
    
    def _harmonic_oscillator_constraint(
        self,
        predictions: torch.Tensor,
        observable_names: List[str]
    ) -> Optional[torch.Tensor]:
        """
        Harmonic oscillator constraint: uniform energy level spacing
        
        Energy levels should be uniformly spaced: Eₙ₊₁ - Eₙ = constant
        
        Args:
            predictions: Model predictions
            observable_names: Names of observables
            
        Returns:
            Constraint penalty or None if observables not found
        """
        try:
            # Find energy level indices
            energy_indices = []
            
            for i, name in enumerate(observable_names):
                if 'energy' in name.lower() and 'level' in name.lower():
                    energy_indices.append(i)
            
            if len(energy_indices) >= 2:
                # Extract energy levels
                energies = predictions[:, energy_indices]
                
                # Compute spacing between consecutive levels
                spacings = energies[:, 1:] - energies[:, :-1]
                
                # Penalty: variance of spacings (should be zero for uniform spacing)
                penalty = torch.var(spacings)
                
                return penalty
            
        except Exception as e:
            logger.debug(f"Could not apply harmonic oscillator constraint: {e}")
        
        return None
    
    def _probability_norm_constraint(
        self,
        predictions: torch.Tensor,
        observable_names: List[str]
    ) -> Optional[torch.Tensor]:
        """
        Probability normalization constraint: probabilities sum to 1
        
        For measurement probabilities, the sum should equal 1.
        
        Args:
            predictions: Model predictions
            observable_names: Names of observables
            
        Returns:
            Constraint penalty or None if observables not found
        """
        try:
            # Find probability indices
            prob_indices = []
            
            for i, name in enumerate(observable_names):
                if 'prob' in name.lower() or 'probability' in name.lower():
                    prob_indices.append(i)
            
            if len(prob_indices) >= 2:
                # Extract probabilities
                probs = predictions[:, prob_indices]
                
                # Ensure non-negative (apply softmax or clamp)
                probs = torch.clamp(probs, min=0.0)
                
                # Penalty: mean((sum(probs) - 1)²)
                prob_sums = torch.sum(probs, dim=1)
                penalty = torch.mean((prob_sums - 1.0) ** 2)
                
                return penalty
            
        except Exception as e:
            logger.debug(f"Could not apply probability norm constraint: {e}")
        
        return None
    
    def get_constraint_info(self) -> Dict:
        """
        Get information about active constraints
        
        Returns:
            Dictionary with constraint information
        """
        return {
            'constraint_weight': self.constraint_weight,
            'active_constraints': self.constraints,
            'base_loss': str(self.base_loss)
        }


class PhysicsInformedMLPRegressor:
    """
    Wrapper for MLP with physics-informed loss
    
    This is a convenience class that wraps the standard MLPRegressor
    and replaces its loss function with PhysicsInformedLoss.
    """
    
    def __init__(
        self,
        mlp_model,
        constraint_weight: float = 0.1,
        constraints: Optional[List[str]] = None
    ):
        """
        Initialize physics-informed MLP
        
        Args:
            mlp_model: Standard MLPRegressor instance
            constraint_weight: Weight for constraint penalty
            constraints: List of constraint types
        """
        self.mlp_model = mlp_model
        self.physics_loss = PhysicsInformedLoss(
            constraint_weight=constraint_weight,
            constraints=constraints
        )
        
        logger.info("PhysicsInformedMLPRegressor initialized")
    
    def train_with_physics_constraints(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        observable_names: List[str]
    ):
        """
        Train MLP with physics-informed loss
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            observable_names: Names of observables
            
        Returns:
            Training metrics
        """
        # Store observable names for loss computation
        self.observable_names = observable_names
        
        # Replace the loss function in the training loop
        # This would require modifying the MLPRegressor.train() method
        # to accept a custom loss function
        
        logger.info("Training with physics-informed loss...")
        
        # For now, we'll need to modify the MLP training loop
        # This is a placeholder for the integration
        raise NotImplementedError(
            "Physics-informed training requires modification of MLPRegressor.train() "
            "to accept custom loss functions. See design document for integration details."
        )


def create_physics_informed_loss(
    constraint_weight: float = 0.1,
    constraints: Optional[List[str]] = None
) -> PhysicsInformedLoss:
    """
    Factory function to create physics-informed loss
    
    Args:
        constraint_weight: Weight for constraint penalty
        constraints: List of constraint types
        
    Returns:
        PhysicsInformedLoss instance
    """
    return PhysicsInformedLoss(
        constraint_weight=constraint_weight,
        constraints=constraints
    )


# Example usage and testing
if __name__ == '__main__':
    print("=" * 80)
    print("Testing Physics-Informed Loss")
    print("=" * 80)
    
    # Create loss function
    loss_fn = PhysicsInformedLoss(
        constraint_weight=0.1,
        constraints=['schrodinger']
    )
    
    print(f"\nLoss function created: {loss_fn.get_constraint_info()}")
    
    # Test with dummy data
    batch_size = 32
    n_observables = 5
    
    predictions = torch.randn(batch_size, n_observables)
    targets = torch.randn(batch_size, n_observables)
    
    # Simulate Schrödinger observables
    observable_names = ['transmission', 'reflection', 'energy', 'momentum', 'position']
    
    # Compute loss
    loss = loss_fn(predictions, targets, observable_names)
    
    print(f"\nTest loss computed: {loss.item():.6f}")
    print("\n✓ Physics-informed loss is working!")
    print("=" * 80)
