#!/usr/bin/env python3
"""
Unit Tests for XGBoost Regressor
Task 8.4: Comprehensive unit tests for XGBoost model

Tests cover:
- Training on synthetic data
- Save/load functionality
- GPU acceleration
- Model architecture
"""

import sys
sys.path.insert(0, '/home/windows/ravan-quantum-ml')

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from xgboost_regressor import XGBoostRegressor
from model_base import ModelConfig


class TestXGBoostRegressor:
    """Unit tests for XGBoost regressor"""
    
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
            model_type='xgboost',
            input_dim=5,
            output_dim=3,
            hyperparameters={
                'n_estimators': 50,
                'max_depth': 5,
                'learning_rate': 0.1,
                'tree_method': 'hist',  # Use hist for CPU/GPU compatibility
                'early_stopping_rounds': 10
            },
            save_path=str(Path(temp_dir) / 'test_xgboost')
        )
    
    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic data with known pattern"""
        np.random.seed(42)
        
        # Generate input data
        n_samples = 200
        n_features = 5
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        
        # Generate output with known relationship
        # Non-linear pattern (better for tree-based models)
        y = np.zeros((n_samples, 3), dtype=np.float32)
        y[:, 0] = X[:, 0]**2 + X[:, 1] + np.random.randn(n_samples) * 0.1
        y[:, 1] = np.sin(X[:, 2]) + X[:, 3] + np.random.randn(n_samples) * 0.1
        y[:, 2] = X[:, 0] * X[:, 4] + np.random.randn(n_samples) * 0.1
        
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
        model = XGBoostRegressor(simple_config)
        
        assert model is not None, "Model should be created"
        assert not model.is_trained, "Model should not be trained initially"
    
    def test_model_configuration(self, simple_config):
        """Test that model configuration is set correctly"""
        model = XGBoostRegressor(simple_config)
        
        assert model.config.input_dim == 5, "Input dimension should match"
        assert model.config.output_dim == 3, "Output dimension should match"
        assert model.n_estimators == 50, "n_estimators should match config"
        assert model.max_depth == 5, "max_depth should match config"
    
    def test_gpu_configuration(self, simple_config):
        """Test GPU configuration"""
        model = XGBoostRegressor(simple_config)
        
        # Should have tree_method configured
        assert hasattr(model, 'tree_method'), "Should have tree_method attribute"
    
    # =========================================================================
    # Test 2: Training on Synthetic Data
    # =========================================================================
    
    def test_training_basic(self, simple_config, synthetic_data):
        """Test basic training functionality"""
        model = XGBoostRegressor(simple_config)
        
        metrics = model.train(
            synthetic_data['X_train'],
            synthetic_data['y_train'],
            synthetic_data['X_val'],
            synthetic_data['y_val']
        )
        
        assert model.is_trained, "Model should be marked as trained"
        assert metrics is not None, "Training should return metrics"
        assert metrics.val_mse > 0, "Validation MSE should be positive"
    
    def test_training_convergence(self, simple_config, synthetic_data):
        """Test that training improves performance"""
        model = XGBoostRegressor(simple_config)
        
        metrics = model.train(
            synthetic_data['X_train'],
            synthetic_data['y_train'],
            synthetic_data['X_val'],
            synthetic_data['y_val']
        )
        
        # R² should be positive (better than mean baseline)
        assert metrics.r2_score > 0, \
            f"R² score {metrics.r2_score} should be positive"
    
    def test_training_with_nonlinear_pattern(self, simple_config, synthetic_data):
        """Test that XGBoost learns non-linear patterns"""
        # XGBoost should handle non-linear patterns well
        simple_config.hyperparameters['n_estimators'] = 100
        model = XGBoostRegressor(simple_config)
        
        metrics = model.train(
            synthetic_data['X_train'],
            synthetic_data['y_train'],
            synthetic_data['X_val'],
            synthetic_data['y_val']
        )
        
        # Should achieve reasonable R² for non-linear pattern
        assert metrics.r2_score > 0.3, \
            f"R² score {metrics.r2_score} too low for non-linear pattern"
    
    def test_training_metrics_completeness(self, simple_config, synthetic_data):
        """Test that all expected metrics are returned"""
        model = XGBoostRegressor(simple_config)
        
        metrics = model.train(
            synthetic_data['X_train'],
            synthetic_data['y_train'],
            synthetic_data['X_val'],
            synthetic_data['y_val']
        )
        
        # Check all expected metrics exist
        assert hasattr(metrics, 'train_mse'), "Should have train_mse"
        assert hasattr(metrics, 'val_mse'), "Should have val_mse"
        assert hasattr(metrics, 'train_mae'), "Should have train_mae"
        assert hasattr(metrics, 'val_mae'), "Should have val_mae"
        assert hasattr(metrics, 'r2_score'), "Should have r2_score"
        assert hasattr(metrics, 'training_time'), "Should have training_time"
    
    def test_early_stopping(self, simple_config, synthetic_data):
        """Test that early stopping works"""
        simple_config.hyperparameters['n_estimators'] = 200
        simple_config.hyperparameters['early_stopping_rounds'] = 10
        
        model = XGBoostRegressor(simple_config)
        
        metrics = model.train(
            synthetic_data['X_train'],
            synthetic_data['y_train'],
            synthetic_data['X_val'],
            synthetic_data['y_val']
        )
        
        # Should complete (early stopping should work)
        assert metrics is not None, "Training should complete with early stopping"
    
    # =========================================================================
    # Test 3: Prediction
    # =========================================================================
    
    def test_prediction_shape(self, simple_config, synthetic_data):
        """Test that predictions have correct shape"""
        model = XGBoostRegressor(simple_config)
        
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
        model = XGBoostRegressor(simple_config)
        
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
    
    def test_prediction_consistency(self, simple_config, synthetic_data):
        """Test that predictions are consistent (deterministic)"""
        model = XGBoostRegressor(simple_config)
        
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
    
    # =========================================================================
    # Test 4: Save/Load Functionality
    # =========================================================================
    
    def test_save_model(self, simple_config, synthetic_data, temp_dir):
        """Test that model can be saved"""
        model = XGBoostRegressor(simple_config)
        
        # Train model
        model.train(
            synthetic_data['X_train'],
            synthetic_data['y_train'],
            synthetic_data['X_val'],
            synthetic_data['y_val']
        )
        
        # Save model
        model.save()
        
        # Check that directory was created
        save_path = Path(simple_config.save_path)
        assert save_path.exists(), "Save directory should exist"
    
    def test_load_model(self, simple_config, synthetic_data, temp_dir):
        """Test that model can be loaded"""
        # Train and save model
        model1 = XGBoostRegressor(simple_config)
        model1.train(
            synthetic_data['X_train'],
            synthetic_data['y_train'],
            synthetic_data['X_val'],
            synthetic_data['y_val']
        )
        model1.save()
        
        # Create new model and load
        model2 = XGBoostRegressor(simple_config)
        model2.load(simple_config.save_path)
        
        assert model2.is_trained, "Loaded model should be marked as trained"
    
    def test_save_load_predictions_match(self, simple_config, synthetic_data, temp_dir):
        """Test that predictions match after save/load"""
        # Train and save model
        model1 = XGBoostRegressor(simple_config)
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
        model2 = XGBoostRegressor(simple_config)
        model2.load(simple_config.save_path)
        
        # Make predictions with loaded model
        pred2 = model2.predict(X_test)
        
        # Predictions should match
        np.testing.assert_array_almost_equal(pred1, pred2, decimal=4,
            err_msg="Predictions should match after save/load")
    
    # =========================================================================
    # Test 5: Input Validation
    # =========================================================================
    
    def test_input_shape_validation(self, simple_config, synthetic_data):
        """Test that input shape is validated"""
        model = XGBoostRegressor(simple_config)
        
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
    
    # =========================================================================
    # Test 6: Different Hyperparameters
    # =========================================================================
    
    def test_shallow_trees(self, synthetic_data, temp_dir):
        """Test with shallow trees"""
        config = ModelConfig(
            model_type='xgboost',
            input_dim=5,
            output_dim=3,
            hyperparameters={
                'n_estimators': 50,
                'max_depth': 2,  # Shallow
                'learning_rate': 0.1,
                'tree_method': 'hist'
            },
            save_path=str(Path(temp_dir) / 'shallow_model')
        )
        
        model = XGBoostRegressor(config)
        
        metrics = model.train(
            synthetic_data['X_train'],
            synthetic_data['y_train'],
            synthetic_data['X_val'],
            synthetic_data['y_val']
        )
        
        assert metrics is not None, "Shallow trees should train"
    
    def test_deep_trees(self, synthetic_data, temp_dir):
        """Test with deep trees"""
        config = ModelConfig(
            model_type='xgboost',
            input_dim=5,
            output_dim=3,
            hyperparameters={
                'n_estimators': 50,
                'max_depth': 10,  # Deep
                'learning_rate': 0.1,
                'tree_method': 'hist'
            },
            save_path=str(Path(temp_dir) / 'deep_model')
        )
        
        model = XGBoostRegressor(config)
        
        metrics = model.train(
            synthetic_data['X_train'],
            synthetic_data['y_train'],
            synthetic_data['X_val'],
            synthetic_data['y_val']
        )
        
        assert metrics is not None, "Deep trees should train"
    
    def test_high_learning_rate(self, synthetic_data, temp_dir):
        """Test with high learning rate"""
        config = ModelConfig(
            model_type='xgboost',
            input_dim=5,
            output_dim=3,
            hyperparameters={
                'n_estimators': 30,
                'max_depth': 5,
                'learning_rate': 0.3,  # High
                'tree_method': 'hist'
            },
            save_path=str(Path(temp_dir) / 'high_lr_model')
        )
        
        model = XGBoostRegressor(config)
        
        metrics = model.train(
            synthetic_data['X_train'],
            synthetic_data['y_train'],
            synthetic_data['X_val'],
            synthetic_data['y_val']
        )
        
        assert metrics is not None, "High learning rate should train"
    
    def test_low_learning_rate(self, synthetic_data, temp_dir):
        """Test with low learning rate"""
        config = ModelConfig(
            model_type='xgboost',
            input_dim=5,
            output_dim=3,
            hyperparameters={
                'n_estimators': 50,
                'max_depth': 5,
                'learning_rate': 0.01,  # Low
                'tree_method': 'hist'
            },
            save_path=str(Path(temp_dir) / 'low_lr_model')
        )
        
        model = XGBoostRegressor(config)
        
        metrics = model.train(
            synthetic_data['X_train'],
            synthetic_data['y_train'],
            synthetic_data['X_val'],
            synthetic_data['y_val']
        )
        
        assert metrics is not None, "Low learning rate should train"
    
    # =========================================================================
    # Test 7: Edge Cases
    # =========================================================================
    
    def test_single_sample_prediction(self, simple_config, synthetic_data):
        """Test prediction on single sample"""
        model = XGBoostRegressor(simple_config)
        
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
        model = XGBoostRegressor(simple_config)
        
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
    
    def test_few_estimators(self, synthetic_data, temp_dir):
        """Test with very few estimators"""
        config = ModelConfig(
            model_type='xgboost',
            input_dim=5,
            output_dim=3,
            hyperparameters={
                'n_estimators': 10,  # Very few
                'max_depth': 5,
                'learning_rate': 0.1,
                'tree_method': 'hist'
            },
            save_path=str(Path(temp_dir) / 'few_est_model')
        )
        
        model = XGBoostRegressor(config)
        
        metrics = model.train(
            synthetic_data['X_train'],
            synthetic_data['y_train'],
            synthetic_data['X_val'],
            synthetic_data['y_val']
        )
        
        assert metrics is not None, "Should handle few estimators"
    
    def test_many_estimators(self, synthetic_data, temp_dir):
        """Test with many estimators"""
        config = ModelConfig(
            model_type='xgboost',
            input_dim=5,
            output_dim=3,
            hyperparameters={
                'n_estimators': 200,  # Many
                'max_depth': 5,
                'learning_rate': 0.1,
                'tree_method': 'hist'
            },
            save_path=str(Path(temp_dir) / 'many_est_model')
        )
        
        model = XGBoostRegressor(config)
        
        metrics = model.train(
            synthetic_data['X_train'],
            synthetic_data['y_train'],
            synthetic_data['X_val'],
            synthetic_data['y_val']
        )
        
        assert metrics is not None, "Should handle many estimators"


# =============================================================================
# Test Runner
# =============================================================================

if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
