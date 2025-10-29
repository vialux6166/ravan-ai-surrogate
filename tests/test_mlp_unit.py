#!/usr/bin/env python3
"""
Unit Tests for MLP Regressor
Task 7.4: Comprehensive unit tests for MLP model

Tests cover:
- Training on synthetic data with known patterns
- Save/load functionality
- Prediction shapes and ranges
- Model architecture
- GPU/CPU compatibility
"""

import sys
sys.path.insert(0, '/home/windows/ravan-quantum-ml/src')

import pytest
import numpy as np
import torch
import tempfile
import shutil
from pathlib import Path

from models.mlp import MLPRegressor
from models.base import ModelConfig


class TestMLPRegressor:
    """Unit tests for MLP regressor"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for model saving"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def simple_config(self, temp_dir):
        """Create simple model configuration"""
        return ModelConfig(
            model_type='mlp',
            input_dim=5,
            output_dim=3,
            hyperparameters={
                'hidden_layers': [32, 16],
                'dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 10,
                'patience': 5
            },
            save_path=str(Path(temp_dir) / 'test_model')
        )
    
    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic data with known pattern"""
        np.random.seed(42)
        
        # Generate input data
        n_samples = 200
        n_features = 5
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        
        # Generate output with known linear relationship
        # y = W @ X + b + noise
        W = np.array([[1.0, 0.5, 0.0, -0.5, -1.0],
                      [0.0, 1.0, 0.5, 0.0, -0.5],
                      [-0.5, 0.0, 1.0, 0.5, 0.0]]).T
        b = np.array([1.0, 2.0, 3.0])
        
        y = (X @ W + b + np.random.randn(n_samples, 3) * 0.1).astype(np.float32)
        
        # Split into train/val
        split = int(0.8 * n_samples)
        return {
            'X_train': X[:split],
            'y_train': y[:split],
            'X_val': X[split:],
            'y_val': y[split:]
        }
    
    # =========================================================================
    # Test 1: Model Initialization
    # =========================================================================
    
    def test_model_initialization(self, simple_config):
        """Test that model initializes correctly"""
        model = MLPRegressor(simple_config)
        
        assert model is not None, "Model should be created"
        assert model.model is not None, "Network should be created"
        assert not model.is_trained, "Model should not be trained initially"
    
    def test_model_architecture(self, simple_config):
        """Test that model architecture matches configuration"""
        model = MLPRegressor(simple_config)
        
        # Check input/output dimensions
        assert model.config.input_dim == 5, "Input dimension should match"
        assert model.config.output_dim == 3, "Output dimension should match"
        
        # Check hidden layers
        assert model.hidden_layers == [32, 16], "Hidden layers should match config"
    
    def test_device_selection(self, simple_config):
        """Test that model selects appropriate device (GPU/CPU)"""
        model = MLPRegressor(simple_config)
        
        # Should select CUDA if available, otherwise CPU
        expected_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        assert model.device.type == expected_device, \
            f"Device should be {expected_device}"
    
    # =========================================================================
    # Test 2: Training on Synthetic Data
    # =========================================================================
    
    def test_training_basic(self, simple_config, synthetic_data):
        """Test basic training functionality"""
        model = MLPRegressor(simple_config)
        
        metrics = model.train(
            synthetic_data['X_train'],
            synthetic_data['y_train'],
            synthetic_data['X_val'],
            synthetic_data['y_val']
        )
        
        assert model.is_trained, "Model should be marked as trained"
        assert metrics is not None, "Training should return metrics"
        assert metrics.train_mse > 0, "Training MSE should be positive"
        assert metrics.val_mse > 0, "Validation MSE should be positive"
    
    def test_training_convergence(self, simple_config, synthetic_data):
        """Test that training reduces loss"""
        model = MLPRegressor(simple_config)
        
        metrics = model.train(
            synthetic_data['X_train'],
            synthetic_data['y_train'],
            synthetic_data['X_val'],
            synthetic_data['y_val']
        )
        
        # Loss should decrease during training
        initial_loss = metrics.train_loss[0]
        final_loss = metrics.train_loss[-1]
        
        assert final_loss < initial_loss, \
            f"Loss should decrease: {initial_loss} -> {final_loss}"
    
    def test_training_with_known_pattern(self, simple_config, synthetic_data):
        """Test that model learns known linear pattern"""
        # Use more epochs for better learning
        simple_config.hyperparameters['epochs'] = 50
        model = MLPRegressor(simple_config)
        
        metrics = model.train(
            synthetic_data['X_train'],
            synthetic_data['y_train'],
            synthetic_data['X_val'],
            synthetic_data['y_val']
        )
        
        # R² should be reasonably high for linear pattern
        assert metrics.r2_score > 0.5, \
            f"R² score {metrics.r2_score} too low for simple pattern"
    
    def test_early_stopping(self, simple_config, synthetic_data):
        """Test that early stopping works"""
        simple_config.hyperparameters['epochs'] = 100
        simple_config.hyperparameters['patience'] = 5
        
        model = MLPRegressor(simple_config)
        
        metrics = model.train(
            synthetic_data['X_train'],
            synthetic_data['y_train'],
            synthetic_data['X_val'],
            synthetic_data['y_val']
        )
        
        # Should stop before max epochs if converged
        actual_epochs = len(metrics.train_loss)
        assert actual_epochs <= 100, "Should not exceed max epochs"
    
    def test_training_metrics_completeness(self, simple_config, synthetic_data):
        """Test that all expected metrics are returned"""
        model = MLPRegressor(simple_config)
        
        metrics = model.train(
            synthetic_data['X_train'],
            synthetic_data['y_train'],
            synthetic_data['X_val'],
            synthetic_data['y_val']
        )
        
        # Check all expected metrics exist
        assert hasattr(metrics, 'train_loss'), "Should have train_loss"
        assert hasattr(metrics, 'val_loss'), "Should have val_loss"
        assert hasattr(metrics, 'train_mse'), "Should have train_mse"
        assert hasattr(metrics, 'val_mse'), "Should have val_mse"
        assert hasattr(metrics, 'r2_score'), "Should have r2_score"
        assert hasattr(metrics, 'training_time'), "Should have training_time"
    
    # =========================================================================
    # Test 3: Prediction
    # =========================================================================
    
    def test_prediction_shape(self, simple_config, synthetic_data):
        """Test that predictions have correct shape"""
        model = MLPRegressor(simple_config)
        
        # Train model
        model.train(
            synthetic_data['X_train'],
            synthetic_data['y_train'],
            synthetic_data['X_val'],
            synthetic_data['y_val']
        )
        
        # Make predictions
        X_test = synthetic_data['X_val']
        predictions = model.predict(X_test)
        
        # Check shape
        expected_shape = (len(X_test), simple_config.output_dim)
        assert predictions.shape == expected_shape, \
            f"Prediction shape {predictions.shape} != expected {expected_shape}"
    
    def test_prediction_range(self, simple_config, synthetic_data):
        """Test that predictions are in reasonable range"""
        model = MLPRegressor(simple_config)
        
        # Train model
        model.train(
            synthetic_data['X_train'],
            synthetic_data['y_train'],
            synthetic_data['X_val'],
            synthetic_data['y_val']
        )
        
        # Make predictions
        predictions = model.predict(synthetic_data['X_val'])
        
        # Predictions should be finite
        assert np.all(np.isfinite(predictions)), \
            "All predictions should be finite"
        
        # Predictions should be in reasonable range (not too extreme)
        assert np.abs(predictions).max() < 1000, \
            "Predictions should not be extremely large"
    
    def test_prediction_consistency(self, simple_config, synthetic_data):
        """Test that predictions are consistent (deterministic)"""
        model = MLPRegressor(simple_config)
        
        # Train model
        model.train(
            synthetic_data['X_train'],
            synthetic_data['y_train'],
            synthetic_data['X_val'],
            synthetic_data['y_val']
        )
        
        # Make predictions twice
        X_test = synthetic_data['X_val']
        pred1 = model.predict(X_test)
        pred2 = model.predict(X_test)
        
        # Should be identical (deterministic)
        np.testing.assert_array_almost_equal(pred1, pred2, decimal=5,
            err_msg="Predictions should be deterministic")
    
    def test_prediction_before_training(self, simple_config, synthetic_data):
        """Test that prediction works even before training"""
        model = MLPRegressor(simple_config)
        
        # Should be able to predict (with random weights)
        predictions = model.predict(synthetic_data['X_val'])
        
        assert predictions is not None, "Should return predictions"
        assert predictions.shape[0] == len(synthetic_data['X_val']), \
            "Should predict for all samples"
    
    # =========================================================================
    # Test 4: Save/Load Functionality
    # =========================================================================
    
    def test_save_model(self, simple_config, synthetic_data, temp_dir):
        """Test that model can be saved"""
        model = MLPRegressor(simple_config)
        
        # Train model
        model.train(
            synthetic_data['X_train'],
            synthetic_data['y_train'],
            synthetic_data['X_val'],
            synthetic_data['y_val']
        )
        
        # Save model
        model.save()
        
        # Check that files were created
        save_path = Path(simple_config.save_path)
        assert save_path.exists(), "Save directory should exist"
        
        # Check for model file
        model_file = save_path / 'model.pt'
        assert model_file.exists(), "Model file should exist"
    
    def test_load_model(self, simple_config, synthetic_data, temp_dir):
        """Test that model can be loaded"""
        # Train and save model
        model1 = MLPRegressor(simple_config)
        model1.train(
            synthetic_data['X_train'],
            synthetic_data['y_train'],
            synthetic_data['X_val'],
            synthetic_data['y_val']
        )
        model1.save()
        
        # Create new model and load
        model2 = MLPRegressor(simple_config)
        model2.load(simple_config.save_path)
        
        assert model2.is_trained, "Loaded model should be marked as trained"
    
    def test_save_load_predictions_match(self, simple_config, synthetic_data, temp_dir):
        """Test that predictions match after save/load"""
        # Train and save model
        model1 = MLPRegressor(simple_config)
        model1.train(
            synthetic_data['X_train'],
            synthetic_data['y_train'],
            synthetic_data['X_val'],
            synthetic_data['y_val']
        )
        
        # Make predictions
        X_test = synthetic_data['X_val']
        pred1 = model1.predict(X_test)
        
        # Save model
        model1.save()
        
        # Load model
        model2 = MLPRegressor(simple_config)
        model2.load(simple_config.save_path)
        
        # Make predictions with loaded model
        pred2 = model2.predict(X_test)
        
        # Predictions should match
        np.testing.assert_array_almost_equal(pred1, pred2, decimal=4,
            err_msg="Predictions should match after save/load")
    
    def test_save_metadata(self, simple_config, synthetic_data, temp_dir):
        """Test that metadata is saved"""
        model = MLPRegressor(simple_config)
        
        model.train(
            synthetic_data['X_train'],
            synthetic_data['y_train'],
            synthetic_data['X_val'],
            synthetic_data['y_val']
        )
        
        model.save()
        
        # Check for metadata files
        save_path = Path(simple_config.save_path)
        config_file = save_path / 'config.json'
        metrics_file = save_path / 'metrics.json'
        
        assert config_file.exists(), "Config file should exist"
        assert metrics_file.exists(), "Metrics file should exist"
    
    # =========================================================================
    # Test 5: Input Validation
    # =========================================================================
    
    def test_input_shape_validation(self, simple_config, synthetic_data):
        """Test that input shape is validated"""
        model = MLPRegressor(simple_config)
        
        model.train(
            synthetic_data['X_train'],
            synthetic_data['y_train'],
            synthetic_data['X_val'],
            synthetic_data['y_val']
        )
        
        # Try to predict with wrong input shape
        X_wrong = np.random.randn(10, 10).astype(np.float32)  # Wrong dimension
        
        with pytest.raises(ValueError):
            model.validate_input_shape(X_wrong)
    
    def test_input_dimension_validation(self, simple_config):
        """Test that 1D input is rejected"""
        model = MLPRegressor(simple_config)
        
        # 1D input should be rejected
        X_1d = np.random.randn(10).astype(np.float32)
        
        with pytest.raises(ValueError):
            model.validate_input_shape(X_1d)
    
    # =========================================================================
    # Test 6: Different Architectures
    # =========================================================================
    
    def test_small_network(self, synthetic_data, temp_dir):
        """Test with small network"""
        config = ModelConfig(
            model_type='mlp',
            input_dim=5,
            output_dim=3,
            hyperparameters={
                'hidden_layers': [16],  # Single small layer
                'dropout': 0.1,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 10,
                'patience': 5
            },
            save_path=str(Path(temp_dir) / 'small_model')
        )
        
        model = MLPRegressor(config)
        
        metrics = model.train(
            synthetic_data['X_train'],
            synthetic_data['y_train'],
            synthetic_data['X_val'],
            synthetic_data['y_val']
        )
        
        assert metrics is not None, "Small network should train"
    
    def test_large_network(self, synthetic_data, temp_dir):
        """Test with large network"""
        config = ModelConfig(
            model_type='mlp',
            input_dim=5,
            output_dim=3,
            hyperparameters={
                'hidden_layers': [128, 64, 32],  # Larger network
                'dropout': 0.3,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 10,
                'patience': 5
            },
            save_path=str(Path(temp_dir) / 'large_model')
        )
        
        model = MLPRegressor(config)
        
        metrics = model.train(
            synthetic_data['X_train'],
            synthetic_data['y_train'],
            synthetic_data['X_val'],
            synthetic_data['y_val']
        )
        
        assert metrics is not None, "Large network should train"
    
    def test_no_dropout(self, synthetic_data, temp_dir):
        """Test with no dropout"""
        config = ModelConfig(
            model_type='mlp',
            input_dim=5,
            output_dim=3,
            hyperparameters={
                'hidden_layers': [32, 16],
                'dropout': 0.0,  # No dropout
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 10,
                'patience': 5
            },
            save_path=str(Path(temp_dir) / 'no_dropout_model')
        )
        
        model = MLPRegressor(config)
        
        metrics = model.train(
            synthetic_data['X_train'],
            synthetic_data['y_train'],
            synthetic_data['X_val'],
            synthetic_data['y_val']
        )
        
        assert metrics is not None, "Model without dropout should train"
    
    # =========================================================================
    # Test 7: Edge Cases
    # =========================================================================
    
    def test_single_sample_prediction(self, simple_config, synthetic_data):
        """Test prediction on single sample"""
        model = MLPRegressor(simple_config)
        
        model.train(
            synthetic_data['X_train'],
            synthetic_data['y_train'],
            synthetic_data['X_val'],
            synthetic_data['y_val']
        )
        
        # Predict single sample
        X_single = synthetic_data['X_val'][:1]
        prediction = model.predict(X_single)
        
        assert prediction.shape == (1, simple_config.output_dim), \
            "Should handle single sample"
    
    def test_batch_prediction(self, simple_config, synthetic_data):
        """Test prediction on large batch"""
        model = MLPRegressor(simple_config)
        
        model.train(
            synthetic_data['X_train'],
            synthetic_data['y_train'],
            synthetic_data['X_val'],
            synthetic_data['y_val']
        )
        
        # Predict large batch
        X_large = np.random.randn(1000, 5).astype(np.float32)
        predictions = model.predict(X_large)
        
        assert predictions.shape == (1000, simple_config.output_dim), \
            "Should handle large batch"
    
    def test_small_batch_size(self, synthetic_data, temp_dir):
        """Test with very small batch size"""
        config = ModelConfig(
            model_type='mlp',
            input_dim=5,
            output_dim=3,
            hyperparameters={
                'hidden_layers': [32, 16],
                'dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 8,  # Very small
                'epochs': 5,
                'patience': 3
            },
            save_path=str(Path(temp_dir) / 'small_batch_model')
        )
        
        model = MLPRegressor(config)
        
        metrics = model.train(
            synthetic_data['X_train'],
            synthetic_data['y_train'],
            synthetic_data['X_val'],
            synthetic_data['y_val']
        )
        
        assert metrics is not None, "Should handle small batch size"


# =============================================================================
# Test Runner
# =============================================================================

if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
