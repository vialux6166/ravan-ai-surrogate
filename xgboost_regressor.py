"""
XGBoost Regressor for Ravan Quantum-ML System
GPU-accelerated gradient boosting for quantum observable prediction
"""

import sys
sys.path.insert(0, '/home/windows/ravan-quantum-ml/src')

import xgboost as xgb
import numpy as np
from pathlib import Path
import time
import pickle

from models.base import MLModel, ModelConfig, TrainingMetrics


class XGBoostRegressor(MLModel):
    """
    XGBoost Gradient Boosting Regressor
    
    GPU-accelerated gradient boosting for predicting quantum observables.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize XGBoost regressor
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        
        # Extract hyperparameters
        self.n_estimators = config.hyperparameters.get('n_estimators', 500)
        self.max_depth = config.hyperparameters.get('max_depth', 7)
        self.learning_rate = config.hyperparameters.get('learning_rate', 0.05)
        self.early_stopping_rounds = config.hyperparameters.get('early_stopping_rounds', 50)
        self.subsample = config.hyperparameters.get('subsample', 0.8)
        self.colsample_bytree = config.hyperparameters.get('colsample_bytree', 0.8)
        
        # For XGBoost 3.x: use tree_method='hist' with device='cuda' for GPU
        # For XGBoost 2.x: use tree_method='gpu_hist'
        # Check GPU availability
        gpu_available = self._check_xgboost_gpu()
        
        if gpu_available:
            self.tree_method = 'hist'  # XGBoost 3.x uses 'hist' for both CPU and GPU
            self.device = 'cuda'
            self.logger.info("✓ XGBoost GPU enabled")
        else:
            self.tree_method = 'hist'
            self.device = 'cpu'
            self.logger.warning("XGBoost GPU not available, using CPU")
        
        # Create model with proper GPU configuration
        model_params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'tree_method': self.tree_method,
            'device': self.device,
            'objective': 'reg:squarederror',
            'random_state': 42,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
        }
        
        self.model = xgb.XGBRegressor(**model_params)
        
        self.logger.info(
            f"XGBoost created: n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth}, device={self.device}"
        )
    
    def _check_xgboost_gpu(self) -> bool:
        """
        Check if XGBoost GPU support is available
        
        Returns:
            True if GPU is available for XGBoost
        """
        try:
            # Check CUDA availability first
            import torch
            if not torch.cuda.is_available():
                return False
            
            # Try to create a simple GPU-based model (XGBoost 3.x style)
            test_model = xgb.XGBRegressor(
                n_estimators=1,
                tree_method='hist',
                device='cuda'
            )
            # Try a tiny fit to verify GPU works
            import numpy as np
            X_test = np.random.rand(10, 5).astype(np.float32)
            y_test = np.random.rand(10).astype(np.float32)
            test_model.fit(X_test, y_test, verbose=False)
            return True
        except Exception as e:
            self.logger.debug(f"XGBoost GPU check failed: {e}")
            return False
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> TrainingMetrics:
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Training metrics
        """
        self.logger.info(
            f"Training XGBoost with {self.n_estimators} estimators "
            f"(device={self.device})..."
        )
        start_time = time.time()
        
        # Convert to float32 for better GPU performance
        if self.device == 'cuda':
            X_train = X_train.astype(np.float32)
            y_train = y_train.astype(np.float32)
            X_val = X_val.astype(np.float32)
            y_val = y_val.astype(np.float32)
        
        # Train model with early stopping
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        
        training_time = time.time() - start_time
        
        # Get training history
        results = self.model.evals_result()
        train_losses = results['validation_0']['rmse']
        val_losses = results['validation_1']['rmse']
        
        # Convert RMSE to MSE
        train_losses_mse = [loss**2 for loss in train_losses]
        val_losses_mse = [loss**2 for loss in val_losses]
        
        # Calculate final metrics
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        train_mse = mean_squared_error(y_train, train_pred)
        val_mse = mean_squared_error(y_val, val_pred)
        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        r2 = r2_score(y_val, val_pred)
        
        # Find best epoch
        best_epoch = int(np.argmin(val_losses_mse))
        
        self.is_trained = True
        
        metrics = TrainingMetrics(
            train_loss=train_losses_mse,
            val_loss=val_losses_mse,
            train_mse=train_mse,
            val_mse=val_mse,
            train_mae=train_mae,
            val_mae=val_mae,
            r2_score=r2,
            best_epoch=best_epoch,
            training_time=training_time
        )
        
        self.metrics = metrics
        
        self.logger.info(
            f"Training complete: val_mse={val_mse:.6f}, "
            f"r2={r2:.4f}, time={training_time:.2f}s"
        )
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        self.validate_input_shape(X)
        
        predictions = self.model.predict(X)
        
        # Ensure 2D output
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        
        return predictions
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance scores
        
        Returns:
            Feature importance array
        """
        if not self.is_trained:
            self.logger.warning("Model not trained, returning None")
            return None
        
        importance = self.model.feature_importances_
        return importance
    
    def save(self, path: str = None):
        """
        Save model to disk
        
        Args:
            path: Save path (uses config.save_path if None)
        """
        if path is None:
            path = self.config.save_path
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_model(str(path / 'model.json'))
        
        # Save scaler
        if self.scaler is not None:
            with open(path / 'scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
        
        # Save metadata
        self.save_metadata(str(path))
        
        self.logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """
        Load model from disk
        
        Args:
            path: Path to saved model
        """
        path = Path(path)
        
        # Load model
        self.model.load_model(str(path / 'model.json'))
        
        # Load scaler
        scaler_path = path / 'scaler.pkl'
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        
        # Load metadata
        metadata = self.load_metadata(str(path))
        if 'metrics' in metadata:
            self.metrics = metadata['metrics']
        
        self.is_trained = True
        self.logger.info(f"Model loaded from {path}")
