"""
MLP Regressor for Ravan Quantum-ML System
PyTorch-based Multi-Layer Perceptron for quantum observable prediction
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path
import time
import pickle

from model_base import MLModel, ModelConfig, TrainingMetrics


class MLPNetwork(nn.Module):
    """PyTorch MLP network"""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: list = [256, 128, 64],
        dropout: float = 0.2
    ):
        super(MLPNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class MLPRegressor(MLModel):
    """
    Multi-Layer Perceptron Regressor
    
    PyTorch-based neural network for predicting quantum observables
    from simulation parameters.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize MLP regressor
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        
        # Extract hyperparameters
        self.hidden_layers = config.hyperparameters.get('hidden_layers', [256, 128, 64])
        self.dropout = config.hyperparameters.get('dropout', 0.2)
        self.learning_rate = config.hyperparameters.get('learning_rate', 0.001)
        self.batch_size = config.hyperparameters.get('batch_size', 128)
        self.epochs = config.hyperparameters.get('epochs', 100)
        self.patience = config.hyperparameters.get('patience', 10)
        self.use_mixed_precision = config.hyperparameters.get('mixed_precision', False)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Create network
        self.model = MLPNetwork(
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout
        ).to(self.device)
        
        self.logger.info(
            f"MLP created: {config.input_dim} -> {self.hidden_layers} -> {config.output_dim}"
        )
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        custom_loss_fn=None,
        observable_names=None
    ) -> TrainingMetrics:
        """
        Train MLP model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            custom_loss_fn: Optional custom loss function (e.g., PhysicsInformedLoss)
            observable_names: Names of observables (required for physics-informed loss)
            
        Returns:
            Training metrics
        """
        loss_type = "physics-informed" if custom_loss_fn is not None else "standard MSE"
        self.logger.info(f"Training MLP for {self.epochs} epochs ({loss_type})...")
        start_time = time.time()
        
        # Prepare validation split if not provided
        if X_val is None or y_val is None:
            n = X_train.shape[0]
            split = max(1, int(0.8 * n))
            X_train_np, y_train_np = X_train[:split], y_train[:split]
            X_val_np, y_val_np = X_train[split:], y_train[split:]
        else:
            X_train_np, y_train_np = X_train, y_train
            X_val_np, y_val_np = X_val, y_val

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train_np).to(self.device)
        y_train_t = torch.FloatTensor(y_train_np).to(self.device)
        X_val_t = torch.FloatTensor(X_val_np).to(self.device)
        y_val_t = torch.FloatTensor(y_val_np).to(self.device)
        
        # Create data loaders with optimizations
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=False,  # Data already on GPU
            num_workers=0,  # No workers needed when data is on GPU
            persistent_workers=False  # Not applicable with num_workers=0
        )
        
        # Optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = custom_loss_fn if custom_loss_fn is not None else nn.MSELoss()
        
        # Mixed precision scaler (modern PyTorch API)
        scaler = torch.amp.GradScaler('cuda') if (self.use_mixed_precision and torch.cuda.is_available()) else None
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            epoch_train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                if self.use_mixed_precision:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(batch_X)
                        # Use custom loss if provided
                        if custom_loss_fn is not None and observable_names is not None:
                            loss = criterion(outputs, batch_y, observable_names)
                        else:
                            loss = criterion(outputs, batch_y)
                    scaler.scale(loss).backward()
                    # Gradient clipping for stability
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = self.model(batch_X)
                    # Use custom loss if provided
                    if custom_loss_fn is not None and observable_names is not None:
                        loss = criterion(outputs, batch_y, observable_names)
                    else:
                        loss = criterion(outputs, batch_y)
                    loss.backward()
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                epoch_train_loss += loss.item()
            
            epoch_train_loss /= len(train_loader)
            train_losses.append(epoch_train_loss)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                # Use custom loss if provided
                if custom_loss_fn is not None and observable_names is not None:
                    val_loss = criterion(val_outputs, y_val_t, observable_names).item()
                else:
                    val_loss = criterion(val_outputs, y_val_t).item()
                val_losses.append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            if (epoch + 1) % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch+1}/{self.epochs}: "
                    f"train_loss={epoch_train_loss:.6f}, val_loss={val_loss:.6f}"
                )
        
        training_time = time.time() - start_time
        
        # Calculate final metrics
        self.model.eval()
        with torch.no_grad():
            train_pred = self.model(X_train_t).cpu().numpy()
            val_pred = self.model(X_val_t).cpu().numpy()
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        train_mse = mean_squared_error(y_train_np, train_pred)
        val_mse = mean_squared_error(y_val_np, val_pred)
        train_mae = mean_absolute_error(y_train_np, train_pred)
        val_mae = mean_absolute_error(y_val_np, val_pred)
        r2 = r2_score(y_val_np, val_pred)
        
        self.is_trained = True
        
        metrics = TrainingMetrics(
            train_loss=train_losses,
            val_loss=val_losses,
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
        
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_t).cpu().numpy()
        
        return predictions
    
    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        n_samples: int = 50
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with uncertainty using MC Dropout
        
        Args:
            X: Input features
            n_samples: Number of MC samples
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        self.validate_input_shape(X)
        
        # Enable dropout at inference
        self.model.train()
        
        predictions_list = []
        X_t = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.model(X_t).cpu().numpy()
                predictions_list.append(pred)
        
        predictions_array = np.array(predictions_list)
        mean_pred = predictions_array.mean(axis=0)
        std_pred = predictions_array.std(axis=0)
        
        # Set back to eval mode
        self.model.eval()
        
        return mean_pred, std_pred
    
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
        
        # Save model weights
        torch.save(self.model.state_dict(), path / 'model.pt')
        
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
        
        # Load model weights
        self.model.load_state_dict(torch.load(path / 'model.pt', weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        
        # Load scaler
        scaler_path = path / 'scaler.pkl'
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        
        # Load metadata
        metadata = self.load_metadata(str(path))
        if 'metrics' in metadata:
            # Reconstruct metrics (simplified)
            self.metrics = metadata['metrics']
        
        self.is_trained = True
        self.logger.info(f"Model loaded from {path}")
