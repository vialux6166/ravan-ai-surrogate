#!/usr/bin/env python3
"""Retrain models excluding constant purity target"""

import sys
sys.path.insert(0, '/home/windows/ravan-quantum-ml/src')

import numpy as np
from pathlib import Path
from pipeline.hdf5_storage import HDF5Storage
from pipeline.dataset import Dataset
from pipeline.training import TrainingPipeline
from gpu.accelerator import GPUAccelerator
from gpu.vram_manager import VRAMManager, WorkloadMode
from utils.logging import setup_logging

# Setup
logger = setup_logging(level='INFO')

# Load dataset
logger.info("Loading dataset...")
storage = HDF5Storage()
dataset = storage.load_dataset('/home/windows/ravan-quantum-ml/data/raw/quantum_circuit_500_20251019_153622.h5')

logger.info(f"Original dataset: {dataset.n_samples} samples, {dataset.n_features} features, {dataset.n_targets} targets")
logger.info(f"Observable names: {dataset.observable_names}")

# Find purity index
purity_idx = dataset.observable_names.index('purity')
logger.info(f"Purity is at index {purity_idx}")

# Remove purity from targets
logger.info("Removing constant purity target...")
new_observable_names = [name for name in dataset.observable_names if name != 'purity']
new_y = np.delete(dataset.y, purity_idx, axis=1)

# Create new dataset
filtered_dataset = Dataset(
    X=dataset.X,
    y=new_y,
    parameter_names=dataset.parameter_names,
    observable_names=new_observable_names,
    metadata=dataset.metadata
)

logger.info(f"Filtered dataset: {filtered_dataset.n_samples} samples, {filtered_dataset.n_features} features, {filtered_dataset.n_targets} targets")
logger.info(f"New observable names: {filtered_dataset.observable_names}")

# Initialize GPU and VRAM
logger.info("\nInitializing GPU...")
gpu = GPUAccelerator()
vram_mgr = VRAMManager(gpu)
vram_mgr.request_mode(WorkloadMode.TRAINING)

# Initialize training pipeline
pipeline = TrainingPipeline(gpu_accelerator=gpu)

# Prepare data
logger.info("\nPreparing data...")
train_dataset, val_dataset = pipeline.prepare_data(
    dataset=filtered_dataset,
    train_ratio=0.8,
    normalize=True
)

logger.info(f"Training set: {train_dataset.n_samples} samples")
logger.info(f"Validation set: {val_dataset.n_samples} samples")

# Train models
output_dir = Path('/home/windows/ravan-quantum-ml/results/enhanced_qc_500_no_purity')
output_dir.mkdir(parents=True, exist_ok=True)

results = {}
for model_type in ['mlp', 'xgboost']:
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_type} model...")
    logger.info(f"{'='*60}")
    
    model_output_dir = output_dir / model_type
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    model, metrics = pipeline.train_model(
        model_type=model_type,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        save_path=str(model_output_dir / f"{model_type}_model.pt")
    )
    
    results[model_type] = {
        'mse': metrics.val_mse,
        'mae': metrics.val_mae,
        'r2': metrics.r2_score
    }
    
    logger.info(f"\n{model_type} Results:")
    logger.info(f"  MSE: {metrics.val_mse:.6f}")
    logger.info(f"  MAE: {metrics.val_mae:.6f}")
    logger.info(f"  R²: {metrics.r2_score:.6f}")

# Final summary
logger.info(f"\n{'='*60}")
logger.info("TRAINING COMPLETE (WITHOUT PURITY)")
logger.info(f"{'='*60}")
logger.info(f"\nModel Performance Comparison:")
for model_type, metrics in results.items():
    logger.info(f"\n{model_type}:")
    logger.info(f"  MSE: {metrics['mse']:.6f}")
    logger.info(f"  MAE: {metrics['mae']:.6f}")
    logger.info(f"  R²: {metrics['r2']:.6f}")

logger.info(f"\nModels saved to: {output_dir}")
