#!/usr/bin/env python3
"""
Model Training Script for Ravan Quantum-ML System
Train ML models on quantum simulation datasets
"""

import sys
sys.path.insert(0, '/home/windows/ravan-quantum-ml/src')

import argparse
from pathlib import Path

from pipeline.training import TrainingPipeline
from utils.logging import setup_logging


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Train ML models on quantum simulation datasets'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to HDF5 dataset file'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=['mlp', 'xgboost'],
        choices=['mlp', 'xgboost'],
        help='Models to train (default: mlp xgboost)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./results',
        help='Output directory for results (default: ./results)'
    )
    
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Train/validation split ratio (default: 0.8)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    # Create pipeline
    pipeline = TrainingPipeline()
    
    # Run training
    results = pipeline.run_pipeline(
        dataset_path=args.dataset,
        model_types=args.models,
        output_dir=args.output,
        train_ratio=args.train_ratio
    )
    
    print("\n" + "="*60)
    print("Training Complete! Results saved to:", args.output)
    print("="*60)


if __name__ == '__main__':
    main()
