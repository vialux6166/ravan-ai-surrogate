"""
Uncertainty Quantifier for Ravan Quantum-ML System
Provides uncertainty estimation for ML predictions using multiple methods
"""

import sys
sys.path.insert(0, '/home/windows/ravan-quantum-ml/src')

import numpy as np
from typing import Tuple, List, Literal
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UncertaintyQuantifier:
    """
    Uncertainty quantification for ML predictions
    
    Supports multiple methods:
    - 'mc_dropout': Monte Carlo Dropout
    - 'ensemble': Ensemble of models
    """
    
    def __init__(
        self,
        model=None,
        method: Literal['mc_dropout', 'ensemble'] = 'mc_dropout',
        n_samples: int = 50,
        confidence_level: float = 0.95
    ):
        """
        Initialize uncertainty quantifier
        
        Args:
            model: Trained ML model (single model for mc_dropout, list for ensemble)
            method: Uncertainty quantification method
            n_samples: Number of samples for MC dropout
            confidence_level: Confidence level for intervals (e.g., 0.95 for 95%)
        """
        self.model = model
        self.method = method
        self.n_samples = n_samples
        self.confidence_level = confidence_level
        
        # Validate inputs
        if method == 'ensemble' and not isinstance(model, list):
            raise ValueError("For ensemble method, model must be a list of models")
        
        if method == 'mc_dropout' and not hasattr(model, 'predict_with_uncertainty'):
            raise ValueError(
                "For mc_dropout method, model must have predict_with_uncertainty() method"
            )
        
        logger.info(
            f"UncertaintyQuantifier initialized: method={method}, "
            f"n_samples={n_samples}, confidence={confidence_level}"
        )
    
    def predict_with_uncertainty(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with uncertainty estimates
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Tuple of (predictions, uncertainties)
            - predictions: Mean predictions (n_samples, n_outputs)
            - uncertainties: Standard deviations (n_samples, n_outputs)
        """
        if self.method == 'mc_dropout':
            return self._mc_dropout_uncertainty(X)
        elif self.method == 'ensemble':
            return self._ensemble_uncertainty(X)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _mc_dropout_uncertainty(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Monte Carlo Dropout uncertainty estimation
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (mean predictions, standard deviations)
        """
        mean_pred, std_pred = self.model.predict_with_uncertainty(X, self.n_samples)
        return mean_pred, std_pred
    
    def _ensemble_uncertainty(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ensemble uncertainty estimation
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (mean predictions, standard deviations)
        """
        predictions_list = []
        
        for model in self.model:
            pred = model.predict(X)
            predictions_list.append(pred)
        
        predictions_array = np.array(predictions_list)
        mean_pred = predictions_array.mean(axis=0)
        std_pred = predictions_array.std(axis=0)
        
        return mean_pred, std_pred
    
    def get_confidence_intervals(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get predictions with confidence intervals
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
        """
        mean_pred, std_pred = self.predict_with_uncertainty(X)
        
        # Calculate z-score for confidence level
        from scipy import stats
        z_score = stats.norm.ppf((1 + self.confidence_level) / 2)
        
        lower_bound = mean_pred - z_score * std_pred
        upper_bound = mean_pred + z_score * std_pred
        
        return mean_pred, lower_bound, upper_bound
    
    def get_total_uncertainty(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Get total uncertainty (average across all outputs)
        
        Args:
            X: Input features
            
        Returns:
            Total uncertainty per sample (n_samples,)
        """
        _, std_pred = self.predict_with_uncertainty(X)
        
        # Average uncertainty across all outputs
        total_uncertainty = std_pred.mean(axis=1)
        
        return total_uncertainty
    
    def identify_high_uncertainty_samples(
        self,
        X: np.ndarray,
        top_k: int = None,
        threshold: float = None
    ) -> np.ndarray:
        """
        Identify samples with high uncertainty
        
        Args:
            X: Input features
            top_k: Return top k samples with highest uncertainty
            threshold: Return samples with uncertainty above threshold
            
        Returns:
            Indices of high-uncertainty samples
        """
        total_uncertainty = self.get_total_uncertainty(X)
        
        if top_k is not None:
            # Return top k samples
            indices = np.argsort(total_uncertainty)[-top_k:]
            return indices
        elif threshold is not None:
            # Return samples above threshold
            indices = np.where(total_uncertainty > threshold)[0]
            return indices
        else:
            raise ValueError("Must specify either top_k or threshold")
    
    def calibration_metrics(
        self,
        X: np.ndarray,
        y_true: np.ndarray
    ) -> dict:
        """
        Compute calibration metrics for uncertainty estimates
        
        Args:
            X: Input features
            y_true: True target values
            
        Returns:
            Dictionary of calibration metrics
        """
        mean_pred, lower_bound, upper_bound = self.get_confidence_intervals(X)
        
        # Check if true values fall within confidence intervals
        within_interval = np.logical_and(
            y_true >= lower_bound,
            y_true <= upper_bound
        )
        
        # Coverage: fraction of samples within confidence interval
        coverage = within_interval.mean()
        
        # Average interval width
        interval_width = (upper_bound - lower_bound).mean()
        
        # Prediction error
        prediction_error = np.abs(y_true - mean_pred).mean()
        
        metrics = {
            'coverage': coverage,
            'expected_coverage': self.confidence_level,
            'interval_width': interval_width,
            'prediction_error': prediction_error,
            'calibration_gap': abs(coverage - self.confidence_level)
        }
        
        logger.info(
            f"Calibration metrics: coverage={coverage:.3f} "
            f"(expected={self.confidence_level:.3f}), "
            f"gap={metrics['calibration_gap']:.3f}"
        )
        
        return metrics


def create_uncertainty_quantifier(
    model,
    method: str = 'mc_dropout',
    n_samples: int = 50,
    confidence_level: float = 0.95
) -> UncertaintyQuantifier:
    """
    Factory function to create uncertainty quantifier
    
    Args:
        model: Trained model or list of models
        method: 'mc_dropout' or 'ensemble'
        n_samples: Number of MC samples
        confidence_level: Confidence level for intervals
        
    Returns:
        UncertaintyQuantifier instance
    """
    return UncertaintyQuantifier(
        model=model,
        method=method,
        n_samples=n_samples,
        confidence_level=confidence_level
    )
