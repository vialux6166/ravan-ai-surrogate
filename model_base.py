"""
Base ML Model Interface for Ravan Quantum-ML System
Defines abstract interface for all machine learning models
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
import logging
import numpy as np
import pickle
import json


@dataclass
class ModelConfig:
    """
    Configuration for ML model
    
    Attributes:
        model_type: Type of model (mlp, xgboost, lstm)
        hyperparameters: Model-specific hyperparameters
        input_dim: Input feature dimension
        output_dim: Output dimension
        save_path: Path to save model
    """
    model_type: str = ""
    hyperparameters: Dict[str, Any] = None
    input_dim: int = 0
    output_dim: int = 0
    save_path: str = ""
    
    def __post_init__(self):
        """Initialize default values if not provided"""
        if self.hyperparameters is None:
            self.hyperparameters = {}


@dataclass
class TrainingMetrics:
    """
    Metrics from model training
    
    Attributes:
        train_loss: Training loss history
        val_loss: Validation loss history
        train_mse: Training MSE
        val_mse: Validation MSE
        train_mae: Training MAE
        val_mae: Validation MAE
        r2_score: R² score on validation set
        best_epoch: Epoch with best validation loss
        training_time: Total training time in seconds
    """
    train_loss: List[float]
    val_loss: List[float]
    train_mse: float
    val_mse: float
    train_mae: float
    val_mae: float
    r2_score: float
    best_epoch: int
    training_time: float


class MLModel(ABC):
    """
    Abstract base class for machine learning models
    
    All ML models (MLP, XGBoost, LSTM) must inherit from this class
    and implement the required methods.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize ML model
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.metrics: Optional[TrainingMetrics] = None
        self.logger = logging.getLogger(f'ravan.models.{config.model_type}')
        self.logger.info(f"{config.model_type} model initialized")
    
    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> TrainingMetrics:
        """
        Train model on data
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training targets (n_samples, n_outputs)
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            TrainingMetrics with loss history and performance metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Predictions (n_samples, n_outputs)
        """
        pass
    
    @abstractmethod
    def save(self, path: Optional[str] = None):
        """
        Serialize model to disk
        
        Args:
            path: Save path (uses config.save_path if None)
        """
        pass
    
    @abstractmethod
    def load(self, path: str):
        """
        Load model from disk
        
        Args:
            path: Path to saved model
        """
        pass
    
    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        n_samples: int = 50
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with uncertainty estimates
        
        Args:
            X: Input features
            n_samples: Number of samples for uncertainty estimation
            
        Returns:
            Tuple of (predictions, uncertainties)
            
        Note:
            Default implementation returns predictions with zero uncertainty.
            Subclasses should override for proper uncertainty quantification.
        """
        predictions = self.predict(X)
        uncertainties = np.zeros_like(predictions)
        
        self.logger.warning(
            f"{self.config.model_type} does not implement uncertainty quantification"
        )
        
        return predictions, uncertainties
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores
        
        Returns:
            Feature importance array or None if not supported
            
        Note:
            Subclasses should override if feature importance is available
        """
        self.logger.warning(
            f"{self.config.model_type} does not support feature importance"
        )
        return None
    
    def validate_input_shape(self, X: np.ndarray):
        """
        Validate input feature shape
        
        Args:
            X: Input features
            
        Raises:
            ValueError: If shape is invalid
        """
        if X.ndim != 2:
            raise ValueError(
                f"Expected 2D input, got {X.ndim}D with shape {X.shape}"
            )
        
        if X.shape[1] != self.config.input_dim:
            raise ValueError(
                f"Expected {self.config.input_dim} features, "
                f"got {X.shape[1]}"
            )
    
    def save_metadata(self, path: str):
        """
        Save model metadata (config, metrics)
        
        Args:
            path: Directory path
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_dict = {
            'model_type': self.config.model_type,
            'hyperparameters': self.config.hyperparameters,
            'input_dim': self.config.input_dim,
            'output_dim': self.config.output_dim,
        }
        
        with open(path / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Save metrics if available
        if self.metrics:
            metrics_dict = {
                'train_mse': self.metrics.train_mse,
                'val_mse': self.metrics.val_mse,
                'train_mae': self.metrics.train_mae,
                'val_mae': self.metrics.val_mae,
                'r2_score': self.metrics.r2_score,
                'best_epoch': self.metrics.best_epoch,
                'training_time': self.metrics.training_time,
            }
            
            with open(path / 'metrics.json', 'w') as f:
                json.dump(metrics_dict, f, indent=2)
        
        self.logger.info(f"Metadata saved to {path}")
    
    def load_metadata(self, path: str) -> Dict[str, Any]:
        """
        Load model metadata
        
        Args:
            path: Directory path
            
        Returns:
            Dictionary with config and metrics
        """
        path = Path(path)
        
        metadata = {}
        
        # Load config
        config_path = path / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                metadata['config'] = json.load(f)
        
        # Load metrics
        metrics_path = path / 'metrics.json'
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metadata['metrics'] = json.load(f)
        
        return metadata


class ModelRegistry:
    """
    Registry for ML model types
    
    Allows plugin-style registration of new model types
    """
    
    _models: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, model_class: type):
        """
        Register a model class
        
        Args:
            name: Model type name
            model_class: Model class (must inherit from MLModel)
        """
        if not issubclass(model_class, MLModel):
            raise ValueError(
                f"Model class must inherit from MLModel, got {model_class}"
            )
        
        cls._models[name] = model_class
        logging.getLogger('ravan.models').info(
            f"Registered model type: {name}"
        )
    
    @classmethod
    def get(cls, name: str) -> type:
        """
        Get a registered model class
        
        Args:
            name: Model type name
            
        Returns:
            Model class
            
        Raises:
            KeyError: If model type not registered
        """
        if name not in cls._models:
            raise KeyError(
                f"Model type '{name}' not registered. "
                f"Available: {list(cls._models.keys())}"
            )
        
        return cls._models[name]
    
    @classmethod
    def list_models(cls) -> List[str]:
        """
        List all registered model types
        
        Returns:
            List of model type names
        """
        return list(cls._models.keys())
    
    @classmethod
    def create(cls, config: ModelConfig) -> MLModel:
        """
        Create a model instance from config
        
        Args:
            config: Model configuration
            
        Returns:
            Model instance
        """
        model_class = cls.get(config.model_type)
        return model_class(config)
