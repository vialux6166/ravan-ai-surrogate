"""
HDF5 Dataset Storage for Ravan Quantum-ML System
Handles persistent storage of datasets with metadata
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import h5py
import numpy as np
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from dataset import Dataset


class HDF5Storage:
    """
    HDF5 storage manager for datasets
    
    Provides efficient storage and retrieval of large datasets
    with comprehensive metadata tracking.
    """
    
    def __init__(self, default_path: Optional[str] = None):
        """Initialize HDF5 storage manager
        
        Args:
            default_path: Optional default file path for convenience APIs
        """
        self.logger = logging.getLogger('ravan.hdf5')
        self.default_path = default_path
    
    def compute_dataset_hash(self, X: np.ndarray, y: np.ndarray) -> str:
        """
        Compute SHA256 hash of dataset for reproducibility
        
        Args:
            X: Feature matrix
            y: Target matrix
            
        Returns:
            SHA256 hash string
        """
        # Combine X and y into single array
        combined = np.concatenate([X.flatten(), y.flatten()])
        
        # Compute hash
        hash_obj = hashlib.sha256(combined.tobytes())
        return hash_obj.hexdigest()
    
    def save_dataset(
        self,
        dataset_or_X,
        y: Optional[np.ndarray] = None,
        filepath: Optional[str] = None,
        compression: str = 'gzip',
        compression_opts: int = 4,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Save dataset to HDF5 file
        
        Args:
            dataset_or_X: Dataset instance or feature matrix
            y: Target matrix (if dataset_or_X is not a Dataset)
            filepath: Output file path (uses default_path if None)
            compression: Compression algorithm (gzip, lzf)
            compression_opts: Compression level (0-9 for gzip)
        """
        # Accept either Dataset or (X, y) + metadata signature for tests
        # Handle legacy call pattern: save_dataset(X, y, metadata)
        if isinstance(filepath, dict) and metadata is None:
            metadata = filepath  # shift args
            filepath = None

        if isinstance(dataset_or_X, Dataset):
            dataset = dataset_or_X
        else:
            assert y is not None, "y must be provided when passing arrays"
            param_names = [f"param_{i}" for i in range(dataset_or_X.shape[1])]
            obs_names = [f"obs_{i}" for i in range(y.shape[1])]
            dataset = Dataset(dataset_or_X, y, param_names, obs_names, metadata or {})

        filepath = Path(filepath or self.default_path or 'dataset.h5')
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving dataset to {filepath}")
        
        with h5py.File(filepath, 'w') as f:
            # Create groups
            metadata_grp = f.create_group('metadata')
            params_grp = f.create_group('parameters')
            obs_grp = f.create_group('observables')
            flags_grp = f.create_group('flags')
            
            # Save metadata
            metadata_grp.attrs['n_samples'] = dataset.n_samples
            metadata_grp.attrs['n_features'] = dataset.n_features
            metadata_grp.attrs['n_targets'] = dataset.n_targets
            metadata_grp.attrs['timestamp'] = datetime.now().isoformat()
            metadata_grp.attrs['version'] = '1.0'
            
            # Compute and save dataset hash
            dataset_hash = self.compute_dataset_hash(dataset.X, dataset.y)
            metadata_grp.attrs['dataset_hash'] = dataset_hash
            
            # Save parameter and observable names
            metadata_grp.create_dataset(
                'parameter_names',
                data=np.array(dataset.parameter_names, dtype='S')
            )
            metadata_grp.create_dataset(
                'observable_names',
                data=np.array(dataset.observable_names, dtype='S')
            )
            
            # Save custom metadata
            if dataset.metadata:
                metadata_grp.attrs['custom_metadata'] = json.dumps(dataset.metadata)
            
            # Save parameters (features)
            for i, name in enumerate(dataset.parameter_names):
                params_grp.create_dataset(
                    name,
                    data=dataset.X[:, i],
                    compression=compression,
                    compression_opts=compression_opts
                )
            
            # Save observables (targets)
            for i, name in enumerate(dataset.observable_names):
                obs_grp.create_dataset(
                    name,
                    data=dataset.y[:, i],
                    compression=compression,
                    compression_opts=compression_opts
                )
            
            # Save flags (all True by default)
            flags_grp.create_dataset(
                'validation_passed',
                data=np.ones(dataset.n_samples, dtype=bool),
                compression=compression
            )
            flags_grp.create_dataset(
                'is_edge_case',
                data=np.zeros(dataset.n_samples, dtype=bool),
                compression=compression
            )
            flags_grp.create_dataset(
                'uncertainty_score',
                data=np.zeros(dataset.n_samples, dtype=float),
                compression=compression,
                compression_opts=compression_opts
            )
            
            # Save scalers if available
            if dataset.scaler_X is not None:
                scaler_grp = f.create_group('scalers')
                scaler_grp.create_dataset('X_mean', data=dataset.scaler_X.mean_)
                scaler_grp.create_dataset('X_scale', data=dataset.scaler_X.scale_)
            
            if dataset.scaler_y is not None:
                if 'scalers' not in f:
                    scaler_grp = f.create_group('scalers')
                else:
                    scaler_grp = f['scalers']
                scaler_grp.create_dataset('y_mean', data=dataset.scaler_y.mean_)
                scaler_grp.create_dataset('y_scale', data=dataset.scaler_y.scale_)
        
        self.logger.info(
            f"Dataset saved: {dataset.n_samples} samples, "
            f"hash={dataset_hash[:8]}..."
        )
    
    def load_dataset(self, filepath: str) -> Dataset:
        """
        Load dataset from HDF5 file
        
        Args:
            filepath: Input file path
            
        Returns:
            Loaded dataset
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        self.logger.info(f"Loading dataset from {filepath}")
        
        with h5py.File(filepath, 'r') as f:
            # Load metadata
            metadata_grp = f['metadata']
            n_samples = metadata_grp.attrs['n_samples']
            
            # Load parameter and observable names
            parameter_names = [
                name.decode('utf-8') if isinstance(name, bytes) else name
                for name in metadata_grp['parameter_names'][:]
            ]
            observable_names = [
                name.decode('utf-8') if isinstance(name, bytes) else name
                for name in metadata_grp['observable_names'][:]
            ]
            
            # Load custom metadata
            custom_metadata = {}
            if 'custom_metadata' in metadata_grp.attrs:
                custom_metadata = json.loads(metadata_grp.attrs['custom_metadata'])
            
            # Load parameters
            params_grp = f['parameters']
            X = np.zeros((n_samples, len(parameter_names)))
            for i, name in enumerate(parameter_names):
                X[:, i] = params_grp[name][:]
            
            # Load observables
            obs_grp = f['observables']
            y = np.zeros((n_samples, len(observable_names)))
            for i, name in enumerate(observable_names):
                y[:, i] = obs_grp[name][:]
            
            # Load scalers if available
            scaler_X = None
            scaler_y = None
            if 'scalers' in f:
                from sklearn.preprocessing import StandardScaler
                scaler_grp = f['scalers']
                
                if 'X_mean' in scaler_grp:
                    scaler_X = StandardScaler()
                    scaler_X.mean_ = scaler_grp['X_mean'][:]
                    scaler_X.scale_ = scaler_grp['X_scale'][:]
                    scaler_X.n_features_in_ = len(parameter_names)
                
                if 'y_mean' in scaler_grp:
                    scaler_y = StandardScaler()
                    scaler_y.mean_ = scaler_grp['y_mean'][:]
                    scaler_y.scale_ = scaler_grp['y_scale'][:]
                    scaler_y.n_features_in_ = len(observable_names)
            
            # Create dataset
            dataset = Dataset(
                X=X,
                y=y,
                parameter_names=parameter_names,
                observable_names=observable_names,
                metadata=custom_metadata,
                scaler_X=scaler_X,
                scaler_y=scaler_y
            )
        
        self.logger.info(f"Dataset loaded: {dataset.n_samples} samples")
        return dataset

    # Legacy convenience API for tests
    def load_training_data(self):
        """Return (X, y) from the default_path file for legacy tests."""
        if not self.default_path:
            raise ValueError("No default_path set for HDF5Storage")
        dataset = self.load_dataset(self.default_path)
        return dataset.X, dataset.y
    
    def get_dataset_info(self, filepath: str) -> Dict[str, Any]:
        """
        Get dataset information without loading full data
        
        Args:
            filepath: Input file path
            
        Returns:
            Dictionary with dataset information
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        with h5py.File(filepath, 'r') as f:
            metadata_grp = f['metadata']
            
            info = {
                'n_samples': metadata_grp.attrs['n_samples'],
                'n_features': metadata_grp.attrs['n_features'],
                'n_targets': metadata_grp.attrs['n_targets'],
                'timestamp': metadata_grp.attrs.get('timestamp', 'unknown'),
                'version': metadata_grp.attrs.get('version', 'unknown'),
                'dataset_hash': metadata_grp.attrs.get('dataset_hash', 'unknown'),
                'file_size_mb': filepath.stat().st_size / (1024**2),
                'parameter_names': [
                    name.decode('utf-8') if isinstance(name, bytes) else name
                    for name in metadata_grp['parameter_names'][:]
                ],
                'observable_names': [
                    name.decode('utf-8') if isinstance(name, bytes) else name
                    for name in metadata_grp['observable_names'][:]
                ],
            }
            
            if 'custom_metadata' in metadata_grp.attrs:
                info['custom_metadata'] = json.loads(
                    metadata_grp.attrs['custom_metadata']
                )
        
        return info
