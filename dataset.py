"""
Dataset Management for Ravan Quantum-ML System
Handles in-memory dataset representation and operations
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging


@dataclass
class Dataset:
    """
    In-memory dataset representation
    
    Attributes:
        X: Feature matrix (n_samples, n_features)
        y: Target matrix (n_samples, n_targets)
        parameter_names: List of parameter names
        observable_names: List of observable names
        metadata: Additional metadata
        scaler_X: Fitted scaler for features (optional)
        scaler_y: Fitted scaler for targets (optional)
    """
    X: np.ndarray
    y: np.ndarray
    parameter_names: Optional[List[str]] = None
    observable_names: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    scaler_X: Optional[StandardScaler] = None
    scaler_y: Optional[StandardScaler] = None
    
    def __post_init__(self):
        """Validate dataset after initialization"""
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples: "
                f"X={self.X.shape[0]}, y={self.y.shape[0]}"
            )
        
        # Auto-generate names if not provided
        if self.parameter_names is None:
            self.parameter_names = [f"param_{i}" for i in range(self.X.shape[1])]
        if self.observable_names is None:
            self.observable_names = [f"obs_{i}" for i in range(self.y.shape[1])]

        if self.X.shape[1] != len(self.parameter_names):
            raise ValueError(
                f"X features ({self.X.shape[1]}) must match "
                f"parameter_names ({len(self.parameter_names)})"
            )
        
        if self.y.shape[1] != len(self.observable_names):
            raise ValueError(
                f"y targets ({self.y.shape[1]}) must match "
                f"observable_names ({len(self.observable_names)})"
            )
        
        self.logger = logging.getLogger('ravan.dataset')
        self.logger.info(
            f"Dataset created: {self.X.shape[0]} samples, "
            f"{self.X.shape[1]} features, {self.y.shape[1]} targets"
        )
    
    @property
    def n_samples(self) -> int:
        """Number of samples"""
        return self.X.shape[0]
    
    @property
    def n_features(self) -> int:
        """Number of features"""
        return self.X.shape[1]
    
    @property
    def n_targets(self) -> int:
        """Number of targets"""
        return self.y.shape[1]
    
    def split(
        self,
        train_ratio: float = 0.8,
        random_state: int = 42,
        shuffle: bool = True,
        test_size: Optional[float] = None
    ) -> Tuple['Dataset', 'Dataset']:
        """
        Split into train and validation sets
        
        Args:
            train_ratio: Fraction of data for training
            random_state: Random seed for reproducibility
            shuffle: Whether to shuffle before splitting
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        if test_size is not None:
            train_size = 1.0 - test_size
        else:
            train_size = train_ratio

        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y,
            train_size=train_size,
            random_state=random_state,
            shuffle=shuffle
        )
        
        # Create train dataset
        train_dataset = Dataset(
            X=X_train,
            y=y_train,
            parameter_names=self.parameter_names.copy(),
            observable_names=self.observable_names.copy(),
            metadata={**self.metadata, 'split': 'train'},
            scaler_X=self.scaler_X,
            scaler_y=self.scaler_y
        )
        
        # Create validation dataset
        val_dataset = Dataset(
            X=X_val,
            y=y_val,
            parameter_names=self.parameter_names.copy(),
            observable_names=self.observable_names.copy(),
            metadata={**self.metadata, 'split': 'validation'},
            scaler_X=self.scaler_X,
            scaler_y=self.scaler_y
        )
        
        self.logger.info(
            f"Split dataset: train={train_dataset.n_samples}, "
            f"val={val_dataset.n_samples}"
        )
        
        return train_dataset, val_dataset
    
    def normalize(
        self,
        fit_scaler: bool = True
    ) -> Tuple['Dataset', StandardScaler, StandardScaler]:
        """
        Normalize features and targets using StandardScaler
        
        Args:
            fit_scaler: Whether to fit new scalers (True) or use existing (False)
            
        Returns:
            Tuple of (normalized_dataset, scaler_X, scaler_y)
        """
        if fit_scaler or self.scaler_X is None:
            # Fit new scalers
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            
            X_normalized = scaler_X.fit_transform(self.X)
            y_normalized = scaler_y.fit_transform(self.y)
            
            self.logger.info("Fitted new scalers for normalization")
        else:
            # Use existing scalers
            scaler_X = self.scaler_X
            scaler_y = self.scaler_y
            
            X_normalized = scaler_X.transform(self.X)
            y_normalized = scaler_y.transform(self.y)
            
            self.logger.info("Used existing scalers for normalization")
        
        # Create normalized dataset
        normalized_dataset = Dataset(
            X=X_normalized,
            y=y_normalized,
            parameter_names=self.parameter_names.copy(),
            observable_names=self.observable_names.copy(),
            metadata={**self.metadata, 'normalized': True},
            scaler_X=scaler_X,
            scaler_y=scaler_y
        )
        
        return normalized_dataset, scaler_X, scaler_y
    
    def denormalize_predictions(
        self,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Denormalize predictions back to original scale
        
        Args:
            y_pred: Normalized predictions
            
        Returns:
            Denormalized predictions
        """
        if self.scaler_y is None:
            self.logger.warning("No scaler_y available, returning predictions as-is")
            return y_pred
        
        return self.scaler_y.inverse_transform(y_pred)
    
    def get_feature_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for each feature
        
        Returns:
            Dictionary mapping feature names to statistics
        """
        stats = {}
        for i, name in enumerate(self.parameter_names):
            stats[name] = {
                'mean': float(np.mean(self.X[:, i])),
                'std': float(np.std(self.X[:, i])),
                'min': float(np.min(self.X[:, i])),
                'max': float(np.max(self.X[:, i])),
                'median': float(np.median(self.X[:, i]))
            }
        return stats
    
    def get_target_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for each target
        
        Returns:
            Dictionary mapping target names to statistics
        """
        stats = {}
        for i, name in enumerate(self.observable_names):
            stats[name] = {
                'mean': float(np.mean(self.y[:, i])),
                'std': float(np.std(self.y[:, i])),
                'min': float(np.min(self.y[:, i])),
                'max': float(np.max(self.y[:, i])),
                'median': float(np.median(self.y[:, i]))
            }
        return stats
    
    def check_data_quality(self) -> Dict[str, Any]:
        """
        Check for data quality issues
        
        Returns:
            Dictionary with quality check results
        """
        issues = []
        
        # Check for NaN values
        nan_X = np.isnan(self.X).sum()
        nan_y = np.isnan(self.y).sum()
        if nan_X > 0:
            issues.append(f"Found {nan_X} NaN values in features")
        if nan_y > 0:
            issues.append(f"Found {nan_y} NaN values in targets")
        
        # Check for Inf values
        inf_X = np.isinf(self.X).sum()
        inf_y = np.isinf(self.y).sum()
        if inf_X > 0:
            issues.append(f"Found {inf_X} Inf values in features")
        if inf_y > 0:
            issues.append(f"Found {inf_y} Inf values in targets")
        
        # Check for constant features
        for i, name in enumerate(self.parameter_names):
            if np.std(self.X[:, i]) < 1e-10:
                issues.append(f"Feature '{name}' is constant")
        
        # Check for constant targets
        for i, name in enumerate(self.observable_names):
            if np.std(self.y[:, i]) < 1e-10:
                issues.append(f"Target '{name}' is constant")
        
        quality_report = {
            'n_issues': int(len(issues)),
            'issues': issues,
            'has_nan': bool((nan_X + nan_y) > 0),
            'has_inf': bool((inf_X + inf_y) > 0),
            'is_clean': bool(len(issues) == 0)
        }
        
        if issues:
            self.logger.warning(f"Data quality issues found: {issues}")
        else:
            self.logger.info("Data quality check passed")
        
        return quality_report
    
    def remove_invalid_samples(self) -> 'Dataset':
        """
        Remove samples with NaN or Inf values
        
        Returns:
            Cleaned dataset
        """
        # Find valid samples (no NaN or Inf)
        valid_X = ~(np.isnan(self.X).any(axis=1) | np.isinf(self.X).any(axis=1))
        valid_y = ~(np.isnan(self.y).any(axis=1) | np.isinf(self.y).any(axis=1))
        valid_mask = valid_X & valid_y
        
        n_removed = self.n_samples - valid_mask.sum()
        
        if n_removed > 0:
            self.logger.warning(f"Removed {n_removed} invalid samples")
            
            return Dataset(
                X=self.X[valid_mask],
                y=self.y[valid_mask],
                parameter_names=self.parameter_names.copy(),
                observable_names=self.observable_names.copy(),
                metadata={**self.metadata, 'cleaned': True},
                scaler_X=self.scaler_X,
                scaler_y=self.scaler_y
            )
        else:
            self.logger.info("No invalid samples found")
            return self
    
    def summary(self) -> str:
        """
        Get dataset summary string
        
        Returns:
            Summary string
        """
        lines = [
            "Dataset Summary:",
            f"  Samples: {self.n_samples}",
            f"  Features: {self.n_features} {self.parameter_names}",
            f"  Targets: {self.n_targets} {self.observable_names}",
            f"  Normalized: {self.scaler_X is not None}",
        ]
        
        if self.metadata:
            lines.append(f"  Metadata: {list(self.metadata.keys())}")
        
        return "\n".join(lines)
