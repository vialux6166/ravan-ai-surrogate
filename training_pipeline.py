"""
Training Pipeline for Ravan Quantum-ML System
Orchestrates model training, validation, and evaluation
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from dataset import Dataset
from hdf5_storage import HDF5Storage
from model_base import ModelConfig, ModelRegistry
from mlp_regressor import MLPRegressor
from xgboost_regressor import XGBoostRegressor
from vram_manager import VRAMManager, WorkloadMode
from gpu_accelerator import GPUAccelerator


class TrainingPipeline:
    """
    Training pipeline orchestrator
    
    Manages the complete ML training workflow:
    - Data loading and preprocessing
    - Model training with multiple architectures
    - Validation and metrics computation
    - Results visualization
    """
    
    def __init__(self, gpu_accelerator: Optional[GPUAccelerator] = None):
        """
        Initialize training pipeline
        
        Args:
            gpu_accelerator: GPU accelerator instance (creates new if None)
        """
        self.logger = logging.getLogger('ravan.training_pipeline')
        
        # GPU management
        if gpu_accelerator is None:
            self.gpu = GPUAccelerator()
        else:
            self.gpu = gpu_accelerator
        
        self.vram_mgr = VRAMManager(self.gpu)
        
        # Register models
        ModelRegistry.register('mlp', MLPRegressor)
        ModelRegistry.register('xgboost', XGBoostRegressor)
        
        self.logger.info("Training pipeline initialized")
    
    def load_dataset(self, dataset_path: str) -> Dataset:
        """
        Load dataset from HDF5 file
        
        Args:
            dataset_path: Path to HDF5 dataset
            
        Returns:
            Loaded dataset
        """
        self.logger.info(f"Loading dataset from {dataset_path}")
        storage = HDF5Storage()
        dataset = storage.load_dataset(dataset_path)
        
        self.logger.info(f"Dataset loaded: {dataset.n_samples} samples")
        return dataset
    
    def prepare_data(
        self,
        dataset: Dataset,
        train_ratio: float = 0.8,
        normalize: bool = True
    ) -> tuple[Dataset, Dataset]:
        """
        Prepare data for training
        
        Args:
            dataset: Input dataset
            train_ratio: Fraction for training
            normalize: Whether to normalize features
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        self.logger.info("Preparing data...")
        
        # Check data quality
        quality = dataset.check_data_quality()
        if not quality['is_clean']:
            self.logger.warning(f"Data quality issues: {quality['issues']}")
            dataset = dataset.remove_invalid_samples()
        
        # Split data
        train_dataset, val_dataset = dataset.split(
            train_ratio=train_ratio,
            random_state=42
        )
        
        # Normalize
        if normalize:
            train_dataset, scaler_X, scaler_y = train_dataset.normalize(fit_scaler=True)
            val_dataset, _, _ = val_dataset.normalize(fit_scaler=False)
            val_dataset.scaler_X = scaler_X
            val_dataset.scaler_y = scaler_y
        
        self.logger.info(
            f"Data prepared: train={train_dataset.n_samples}, "
            f"val={val_dataset.n_samples}"
        )
        
        return train_dataset, val_dataset
    
    def train_model(
        self,
        model_type: str,
        train_dataset: Dataset,
        val_dataset: Dataset,
        hyperparameters: Optional[Dict] = None,
        save_path: Optional[str] = None
    ):
        """
        Train a single model
        
        Args:
            model_type: Type of model ('mlp', 'xgboost')
            train_dataset: Training dataset
            val_dataset: Validation dataset
            hyperparameters: Model hyperparameters
            save_path: Path to save trained model
            
        Returns:
            Trained model and metrics
        """
        self.logger.info(f"Training {model_type} model...")
        
        # Request training mode
        self.vram_mgr.request_mode(WorkloadMode.TRAINING)
        self.gpu.log_vram_usage(f"Before {model_type} training")
        
        # Default hyperparameters
        if hyperparameters is None:
            if model_type == 'mlp':
                hyperparameters = {
                    'hidden_layers': [256, 128, 64],
                    'dropout': 0.2,
                    'learning_rate': 0.001,
                    'batch_size': 128,
                    'epochs': 100,
                    'patience': 10,
                    'mixed_precision': False
                }
            elif model_type == 'xgboost':
                hyperparameters = {
                    'n_estimators': 500,
                    'max_depth': 7,
                    'learning_rate': 0.05,
                    'tree_method': 'gpu_hist',
                    'early_stopping_rounds': 50
                }
        
        # Create model config
        if save_path is None:
            save_path = f'./data/models/{model_type}_model'
        
        config = ModelConfig(
            model_type=model_type,
            hyperparameters=hyperparameters,
            input_dim=train_dataset.n_features,
            output_dim=train_dataset.n_targets,
            save_path=save_path
        )
        
        # Create and train model
        model = ModelRegistry.create(config)
        
        metrics = model.train(
            train_dataset.X,
            train_dataset.y,
            val_dataset.X,
            val_dataset.y
        )
        
        # Save model
        model.save()
        
        self.gpu.log_vram_usage(f"After {model_type} training")
        self.gpu.optimize_memory()
        
        self.logger.info(
            f"{model_type} training complete: "
            f"val_mse={metrics.val_mse:.6f}, r2={metrics.r2_score:.4f}"
        )
        
        return model, metrics
    
    def validate_predictions(
        self,
        model,
        val_dataset: Dataset,
        simulator_type: str
    ) -> Dict:
        """
        Validate model predictions against physics constraints
        
        Args:
            model: Trained model
            val_dataset: Validation dataset
            simulator_type: Type of simulator
            
        Returns:
            Validation results
        """
        self.logger.info("Validating predictions...")
        
        # Generate predictions
        predictions = model.predict(val_dataset.X)
        
        # Denormalize if needed
        if val_dataset.scaler_y is not None:
            predictions = val_dataset.denormalize_predictions(predictions)
            y_true = val_dataset.scaler_y.inverse_transform(val_dataset.y)
        else:
            y_true = val_dataset.y
        
        # Check physics constraints
        violations = []
        
        if simulator_type == 'schrodinger':
            # Check T + R = 1.0
            if 'transmission' in val_dataset.observable_names:
                T_idx = val_dataset.observable_names.index('transmission')
                R_idx = val_dataset.observable_names.index('reflection')
                
                T_pred = predictions[:, T_idx]
                R_pred = predictions[:, R_idx]
                sum_TR = T_pred + R_pred
                
                violations_TR = np.abs(sum_TR - 1.0) > 0.01
                n_violations = violations_TR.sum()
                
                if n_violations > 0:
                    violations.append(
                        f"T+R conservation violated in {n_violations} samples "
                        f"({n_violations/len(predictions)*100:.1f}%)"
                    )
        
        validation_results = {
            'n_violations': len(violations),
            'violations': violations,
            'predictions': predictions,
            'y_true': y_true
        }
        
        if violations:
            self.logger.warning(f"Physics violations: {violations}")
        else:
            self.logger.info("All predictions satisfy physics constraints")
        
        return validation_results
    
    def plot_results(
        self,
        metrics,
        validation_results,
        output_dir: str,
        model_name: str
    ):
        """
        Generate result plots
        
        Args:
            metrics: Training metrics
            validation_results: Validation results
            output_dir: Output directory for plots
            model_name: Name of model
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot 1: Training curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(metrics.train_loss, label='Train')
        ax1.plot(metrics.val_loss, label='Validation')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title(f'{model_name} Training Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Predictions vs Actual
        predictions = validation_results['predictions']
        y_true = validation_results['y_true']
        
        ax2.scatter(y_true.flatten(), predictions.flatten(), alpha=0.5, s=10)
        min_val = min(y_true.min(), predictions.min())
        max_val = max(y_true.max(), predictions.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
        ax2.set_xlabel('True Values')
        ax2.set_ylabel('Predictions')
        ax2.set_title(f'{model_name} Predictions vs Actual')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{model_name}_results.png', dpi=150)
        plt.close()
        
        self.logger.info(f"Plots saved to {output_dir}")
    
    def run_pipeline(
        self,
        dataset_path: str,
        model_types: List[str],
        output_dir: str = './results',
        train_ratio: float = 0.8
    ):
        """
        Run complete training pipeline
        
        Args:
            dataset_path: Path to dataset
            model_types: List of model types to train
            output_dir: Output directory for results
            train_ratio: Train/val split ratio
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting Training Pipeline")
        self.logger.info("=" * 60)
        
        # Load and prepare data
        dataset = self.load_dataset(dataset_path)
        train_dataset, val_dataset = self.prepare_data(dataset, train_ratio)
        
        # Get simulator type from metadata
        simulator_type = dataset.metadata.get('simulator_type', 'unknown')
        
        # Train each model
        results = {}
        for model_type in model_types:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Training {model_type.upper()}")
            self.logger.info(f"{'='*60}")
            
            model, metrics = self.train_model(
                model_type,
                train_dataset,
                val_dataset
            )
            
            # Validate predictions
            validation_results = self.validate_predictions(
                model,
                val_dataset,
                simulator_type
            )
            
            # Plot results
            self.plot_results(
                metrics,
                validation_results,
                output_dir,
                model_type
            )
            
            results[model_type] = {
                'model': model,
                'metrics': metrics,
                'validation': validation_results
            }
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Training Pipeline Complete!")
        self.logger.info("=" * 60)
        
        # Summary
        for model_type, result in results.items():
            metrics = result['metrics']
            self.logger.info(
                f"{model_type.upper()}: "
                f"val_mse={metrics.val_mse:.6f}, "
                f"r2={metrics.r2_score:.4f}, "
                f"time={metrics.training_time:.2f}s"
            )
        
        return results
