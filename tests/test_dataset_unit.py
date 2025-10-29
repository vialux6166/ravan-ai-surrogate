#!/usr/bin/env python3
"""
Unit Tests for Dataset Class
Task 19.1: Comprehensive unit tests for Dataset management

Tests cover:
- Dataset initialization and validation
- Train/validation splitting
- Normalization and denormalization
- Data quality checks
- Statistics computation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.pipeline.dataset import Dataset


class TestDataset:
    """Unit tests for Dataset class"""
    
    @pytest.fixture
    def simple_dataset(self):
        """Create simple dataset for testing"""
        np.random.seed(42)
        
        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randn(100, 3).astype(np.float32)
        
        return Dataset(
            X=X,
            y=y,
            parameter_names=['p1', 'p2', 'p3', 'p4', 'p5'],
            observable_names=['o1', 'o2', 'o3'],
            metadata={'test': True}
        )
    
    @pytest.fixture
    def large_dataset(self):
        """Create larger dataset for testing"""
        np.random.seed(42)
        
        X = np.random.randn(1000, 10).astype(np.float32)
        y = np.random.randn(1000, 5).astype(np.float32)
        
        return Dataset(
            X=X,
            y=y,
            parameter_names=[f'param_{i}' for i in range(10)],
            observable_names=[f'obs_{i}' for i in range(5)],
            metadata={'size': 'large'}
        )
    
    # =========================================================================
    # Test 1: Dataset Initialization and Validation
    # =========================================================================
    
    def test_dataset_initialization(self, simple_dataset):
        """Test that dataset initializes correctly"""
        assert simple_dataset is not None
        assert simple_dataset.n_samples == 100
        assert simple_dataset.n_features == 5
        assert simple_dataset.n_targets == 3
    
    def test_dataset_properties(self, simple_dataset):
        """Test dataset properties"""
        assert simple_dataset.n_samples == simple_dataset.X.shape[0]
        assert simple_dataset.n_features == simple_dataset.X.shape[1]
        assert simple_dataset.n_targets == simple_dataset.y.shape[1]
    
    def test_parameter_names(self, simple_dataset):
        """Test parameter names are stored correctly"""
        assert len(simple_dataset.parameter_names) == 5
        assert simple_dataset.parameter_names == ['p1', 'p2', 'p3', 'p4', 'p5']
    
    def test_observable_names(self, simple_dataset):
        """Test observable names are stored correctly"""
        assert len(simple_dataset.observable_names) == 3
        assert simple_dataset.observable_names == ['o1', 'o2', 'o3']
    
    def test_metadata(self, simple_dataset):
        """Test metadata is stored correctly"""
        assert simple_dataset.metadata['test'] is True
    
    def test_mismatched_samples_raises_error(self):
        """Test that mismatched sample counts raise error"""
        X = np.random.randn(100, 5)
        y = np.random.randn(50, 3)  # Different number of samples
        
        with pytest.raises(ValueError, match="same number of samples"):
            Dataset(
                X=X,
                y=y,
                parameter_names=['p1', 'p2', 'p3', 'p4', 'p5'],
                observable_names=['o1', 'o2', 'o3']
            )
    
    def test_mismatched_features_raises_error(self):
        """Test that mismatched feature counts raise error"""
        X = np.random.randn(100, 5)
        y = np.random.randn(100, 3)
        
        with pytest.raises(ValueError, match="must match"):
            Dataset(
                X=X,
                y=y,
                parameter_names=['p1', 'p2', 'p3'],  # Wrong count
                observable_names=['o1', 'o2', 'o3']
            )
    
    def test_mismatched_targets_raises_error(self):
        """Test that mismatched target counts raise error"""
        X = np.random.randn(100, 5)
        y = np.random.randn(100, 3)
        
        with pytest.raises(ValueError, match="must match"):
            Dataset(
                X=X,
                y=y,
                parameter_names=['p1', 'p2', 'p3', 'p4', 'p5'],
                observable_names=['o1']  # Wrong count
            )
    
    # =========================================================================
    # Test 2: Train/Validation Splitting
    # =========================================================================
    
    def test_split_basic(self, simple_dataset):
        """Test basic train/val split"""
        train_ds, val_ds = simple_dataset.split(train_ratio=0.8)
        
        assert train_ds.n_samples == 80
        assert val_ds.n_samples == 20
        assert train_ds.n_samples + val_ds.n_samples == simple_dataset.n_samples
    
    def test_split_preserves_features(self, simple_dataset):
        """Test that split preserves feature dimensions"""
        train_ds, val_ds = simple_dataset.split()
        
        assert train_ds.n_features == simple_dataset.n_features
        assert val_ds.n_features == simple_dataset.n_features
        assert train_ds.n_targets == simple_dataset.n_targets
        assert val_ds.n_targets == simple_dataset.n_targets
    
    def test_split_preserves_names(self, simple_dataset):
        """Test that split preserves parameter and observable names"""
        train_ds, val_ds = simple_dataset.split()
        
        assert train_ds.parameter_names == simple_dataset.parameter_names
        assert val_ds.parameter_names == simple_dataset.parameter_names
        assert train_ds.observable_names == simple_dataset.observable_names
        assert val_ds.observable_names == simple_dataset.observable_names
    
    def test_split_adds_metadata(self, simple_dataset):
        """Test that split adds split metadata"""
        train_ds, val_ds = simple_dataset.split()
        
        assert train_ds.metadata['split'] == 'train'
        assert val_ds.metadata['split'] == 'validation'
    
    def test_split_different_ratios(self, large_dataset):
        """Test different split ratios"""
        ratios = [0.5, 0.7, 0.8, 0.9]
        
        for ratio in ratios:
            train_ds, val_ds = large_dataset.split(train_ratio=ratio)
            
            expected_train = int(ratio * large_dataset.n_samples)
            assert abs(train_ds.n_samples - expected_train) <= 1
            assert train_ds.n_samples + val_ds.n_samples == large_dataset.n_samples
    
    def test_split_reproducibility(self, simple_dataset):
        """Test that split is reproducible with same random_state"""
        train1, val1 = simple_dataset.split(random_state=42)
        train2, val2 = simple_dataset.split(random_state=42)
        
        np.testing.assert_array_equal(train1.X, train2.X)
        np.testing.assert_array_equal(val1.X, val2.X)
    
    def test_split_no_shuffle(self, simple_dataset):
        """Test split without shuffling"""
        train_ds, val_ds = simple_dataset.split(shuffle=False)
        
        # First 80 samples should be in train
        np.testing.assert_array_equal(train_ds.X, simple_dataset.X[:80])
        # Last 20 samples should be in val
        np.testing.assert_array_equal(val_ds.X, simple_dataset.X[80:])
    
    # =========================================================================
    # Test 3: Normalization and Denormalization
    # =========================================================================
    
    def test_normalize_basic(self, simple_dataset):
        """Test basic normalization"""
        norm_ds, scaler_X, scaler_y = simple_dataset.normalize()
        
        assert norm_ds is not None
        assert scaler_X is not None
        assert scaler_y is not None
        assert norm_ds.scaler_X is scaler_X
        assert norm_ds.scaler_y is scaler_y
    
    def test_normalize_zero_mean(self, simple_dataset):
        """Test that normalized data has zero mean"""
        norm_ds, _, _ = simple_dataset.normalize()
        
        # Check mean is close to zero for each feature
        for i in range(norm_ds.n_features):
            assert abs(np.mean(norm_ds.X[:, i])) < 0.1
        
        # Check mean is close to zero for each target
        for i in range(norm_ds.n_targets):
            assert abs(np.mean(norm_ds.y[:, i])) < 0.1
    
    def test_normalize_unit_variance(self, simple_dataset):
        """Test that normalized data has unit variance"""
        norm_ds, _, _ = simple_dataset.normalize()
        
        # Check std is close to 1 for each feature
        for i in range(norm_ds.n_features):
            assert abs(np.std(norm_ds.X[:, i]) - 1.0) < 0.1
        
        # Check std is close to 1 for each target
        for i in range(norm_ds.n_targets):
            assert abs(np.std(norm_ds.y[:, i]) - 1.0) < 0.1
    
    def test_normalize_preserves_shape(self, simple_dataset):
        """Test that normalization preserves shape"""
        norm_ds, _, _ = simple_dataset.normalize()
        
        assert norm_ds.X.shape == simple_dataset.X.shape
        assert norm_ds.y.shape == simple_dataset.y.shape
    
    def test_normalize_adds_metadata(self, simple_dataset):
        """Test that normalization adds metadata"""
        norm_ds, _, _ = simple_dataset.normalize()
        
        assert norm_ds.metadata['normalized'] is True
    
    def test_normalize_with_existing_scaler(self, simple_dataset):
        """Test normalization with existing scaler"""
        # First normalization
        norm_ds1, scaler_X, scaler_y = simple_dataset.normalize()
        
        # Create new dataset with same scalers
        new_dataset = Dataset(
            X=simple_dataset.X,
            y=simple_dataset.y,
            parameter_names=simple_dataset.parameter_names,
            observable_names=simple_dataset.observable_names,
            scaler_X=scaler_X,
            scaler_y=scaler_y
        )
        
        # Normalize with existing scaler
        norm_ds2, _, _ = new_dataset.normalize(fit_scaler=False)
        
        # Should produce same result
        np.testing.assert_array_almost_equal(norm_ds1.X, norm_ds2.X, decimal=5)
        np.testing.assert_array_almost_equal(norm_ds1.y, norm_ds2.y, decimal=5)
    
    def test_denormalize_predictions(self, simple_dataset):
        """Test denormalization of predictions"""
        norm_ds, _, scaler_y = simple_dataset.normalize()
        
        # Create some normalized predictions
        y_pred_norm = norm_ds.y[:10]
        
        # Denormalize
        y_pred_denorm = norm_ds.denormalize_predictions(y_pred_norm)
        
        # Should match original values
        np.testing.assert_array_almost_equal(
            y_pred_denorm, 
            simple_dataset.y[:10], 
            decimal=4
        )
    
    def test_denormalize_without_scaler(self, simple_dataset):
        """Test denormalization without scaler returns as-is"""
        y_pred = np.random.randn(10, 3)
        y_denorm = simple_dataset.denormalize_predictions(y_pred)
        
        # Should return same array
        np.testing.assert_array_equal(y_pred, y_denorm)
    
    # =========================================================================
    # Test 4: Statistics Computation
    # =========================================================================
    
    def test_get_feature_statistics(self, simple_dataset):
        """Test feature statistics computation"""
        stats = simple_dataset.get_feature_statistics()
        
        assert len(stats) == simple_dataset.n_features
        assert all(name in stats for name in simple_dataset.parameter_names)
    
    def test_feature_statistics_keys(self, simple_dataset):
        """Test that feature statistics have correct keys"""
        stats = simple_dataset.get_feature_statistics()
        
        for name in simple_dataset.parameter_names:
            assert 'mean' in stats[name]
            assert 'std' in stats[name]
            assert 'min' in stats[name]
            assert 'max' in stats[name]
            assert 'median' in stats[name]
    
    def test_feature_statistics_values(self, simple_dataset):
        """Test that feature statistics are computed correctly"""
        stats = simple_dataset.get_feature_statistics()
        
        # Check first feature
        col_0 = simple_dataset.X[:, 0]
        assert abs(stats['p1']['mean'] - np.mean(col_0)) < 1e-5
        assert abs(stats['p1']['std'] - np.std(col_0)) < 1e-5
        assert abs(stats['p1']['min'] - np.min(col_0)) < 1e-5
        assert abs(stats['p1']['max'] - np.max(col_0)) < 1e-5
    
    def test_get_target_statistics(self, simple_dataset):
        """Test target statistics computation"""
        stats = simple_dataset.get_target_statistics()
        
        assert len(stats) == simple_dataset.n_targets
        assert all(name in stats for name in simple_dataset.observable_names)
    
    def test_target_statistics_keys(self, simple_dataset):
        """Test that target statistics have correct keys"""
        stats = simple_dataset.get_target_statistics()
        
        for name in simple_dataset.observable_names:
            assert 'mean' in stats[name]
            assert 'std' in stats[name]
            assert 'min' in stats[name]
            assert 'max' in stats[name]
            assert 'median' in stats[name]
    
    # =========================================================================
    # Test 5: Data Quality Checks
    # =========================================================================
    
    def test_check_data_quality_clean(self, simple_dataset):
        """Test data quality check on clean data"""
        quality = simple_dataset.check_data_quality()
        
        assert quality['is_clean'] is True
        assert quality['n_issues'] == 0
        assert quality['has_nan'] is False
        assert quality['has_inf'] is False
    
    def test_check_data_quality_with_nan(self):
        """Test data quality check with NaN values"""
        X = np.random.randn(100, 5)
        X[10, 2] = np.nan  # Add NaN
        y = np.random.randn(100, 3)
        
        dataset = Dataset(
            X=X,
            y=y,
            parameter_names=['p1', 'p2', 'p3', 'p4', 'p5'],
            observable_names=['o1', 'o2', 'o3']
        )
        
        quality = dataset.check_data_quality()
        
        assert quality['is_clean'] is False
        assert quality['has_nan'] is True
        assert quality['n_issues'] > 0
    
    def test_check_data_quality_with_inf(self):
        """Test data quality check with Inf values"""
        X = np.random.randn(100, 5)
        y = np.random.randn(100, 3)
        y[20, 1] = np.inf  # Add Inf
        
        dataset = Dataset(
            X=X,
            y=y,
            parameter_names=['p1', 'p2', 'p3', 'p4', 'p5'],
            observable_names=['o1', 'o2', 'o3']
        )
        
        quality = dataset.check_data_quality()
        
        assert quality['is_clean'] is False
        assert quality['has_inf'] is True
        assert quality['n_issues'] > 0
    
    def test_check_data_quality_constant_feature(self):
        """Test data quality check with constant feature"""
        X = np.random.randn(100, 5)
        X[:, 2] = 1.0  # Constant feature
        y = np.random.randn(100, 3)
        
        dataset = Dataset(
            X=X,
            y=y,
            parameter_names=['p1', 'p2', 'p3', 'p4', 'p5'],
            observable_names=['o1', 'o2', 'o3']
        )
        
        quality = dataset.check_data_quality()
        
        assert quality['is_clean'] is False
        assert quality['n_issues'] > 0
        assert any('constant' in issue.lower() for issue in quality['issues'])
    
    def test_remove_invalid_samples_clean(self, simple_dataset):
        """Test removing invalid samples from clean data"""
        cleaned = simple_dataset.remove_invalid_samples()
        
        assert cleaned.n_samples == simple_dataset.n_samples
    
    def test_remove_invalid_samples_with_nan(self):
        """Test removing samples with NaN"""
        X = np.random.randn(100, 5)
        X[10, 2] = np.nan
        X[20, 3] = np.nan
        y = np.random.randn(100, 3)
        
        dataset = Dataset(
            X=X,
            y=y,
            parameter_names=['p1', 'p2', 'p3', 'p4', 'p5'],
            observable_names=['o1', 'o2', 'o3']
        )
        
        cleaned = dataset.remove_invalid_samples()
        
        assert cleaned.n_samples == 98  # Removed 2 samples
        assert not np.isnan(cleaned.X).any()
    
    def test_remove_invalid_samples_with_inf(self):
        """Test removing samples with Inf"""
        X = np.random.randn(100, 5)
        y = np.random.randn(100, 3)
        y[15, 1] = np.inf
        y[25, 2] = -np.inf
        
        dataset = Dataset(
            X=X,
            y=y,
            parameter_names=['p1', 'p2', 'p3', 'p4', 'p5'],
            observable_names=['o1', 'o2', 'o3']
        )
        
        cleaned = dataset.remove_invalid_samples()
        
        assert cleaned.n_samples == 98  # Removed 2 samples
        assert not np.isinf(cleaned.y).any()
    
    def test_remove_invalid_samples_adds_metadata(self):
        """Test that cleaning adds metadata"""
        X = np.random.randn(100, 5)
        X[10, 2] = np.nan
        y = np.random.randn(100, 3)
        
        dataset = Dataset(
            X=X,
            y=y,
            parameter_names=['p1', 'p2', 'p3', 'p4', 'p5'],
            observable_names=['o1', 'o2', 'o3']
        )
        
        cleaned = dataset.remove_invalid_samples()
        
        assert cleaned.metadata['cleaned'] is True
    
    # =========================================================================
    # Test 6: Summary and Utilities
    # =========================================================================
    
    def test_summary(self, simple_dataset):
        """Test dataset summary generation"""
        summary = simple_dataset.summary()
        
        assert 'Dataset Summary' in summary
        assert '100' in summary  # Number of samples
        assert '5' in summary    # Number of features
        assert '3' in summary    # Number of targets
    
    def test_summary_includes_names(self, simple_dataset):
        """Test that summary includes parameter and observable names"""
        summary = simple_dataset.summary()
        
        # Should contain some parameter names
        assert any(name in summary for name in simple_dataset.parameter_names)
        # Should contain some observable names
        assert any(name in summary for name in simple_dataset.observable_names)
    
    def test_summary_includes_metadata(self, simple_dataset):
        """Test that summary includes metadata info"""
        summary = simple_dataset.summary()
        
        assert 'Metadata' in summary or 'test' in str(simple_dataset.metadata)
    
    # =========================================================================
    # Test 7: Edge Cases
    # =========================================================================
    
    def test_single_sample_dataset(self):
        """Test dataset with single sample"""
        X = np.array([[1.0, 2.0, 3.0]])
        y = np.array([[4.0, 5.0]])
        
        dataset = Dataset(
            X=X,
            y=y,
            parameter_names=['p1', 'p2', 'p3'],
            observable_names=['o1', 'o2']
        )
        
        assert dataset.n_samples == 1
        assert dataset.n_features == 3
        assert dataset.n_targets == 2
    
    def test_single_feature_dataset(self):
        """Test dataset with single feature"""
        X = np.random.randn(100, 1)
        y = np.random.randn(100, 3)
        
        dataset = Dataset(
            X=X,
            y=y,
            parameter_names=['p1'],
            observable_names=['o1', 'o2', 'o3']
        )
        
        assert dataset.n_features == 1
    
    def test_single_target_dataset(self):
        """Test dataset with single target"""
        X = np.random.randn(100, 5)
        y = np.random.randn(100, 1)
        
        dataset = Dataset(
            X=X,
            y=y,
            parameter_names=['p1', 'p2', 'p3', 'p4', 'p5'],
            observable_names=['o1']
        )
        
        assert dataset.n_targets == 1
    
    def test_large_dataset(self, large_dataset):
        """Test operations on large dataset"""
        # Should handle large dataset without issues
        assert large_dataset.n_samples == 1000
        
        # Test split
        train, val = large_dataset.split()
        assert train.n_samples + val.n_samples == 1000
        
        # Test normalize
        norm_ds, _, _ = large_dataset.normalize()
        assert norm_ds.n_samples == 1000
    
    def test_empty_metadata(self):
        """Test dataset with empty metadata"""
        X = np.random.randn(50, 3)
        y = np.random.randn(50, 2)
        
        dataset = Dataset(
            X=X,
            y=y,
            parameter_names=['p1', 'p2', 'p3'],
            observable_names=['o1', 'o2']
        )
        
        assert dataset.metadata == {}


# =============================================================================
# Test Runner
# =============================================================================

if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
