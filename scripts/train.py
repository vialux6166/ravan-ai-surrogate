#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error

from mlp_regressor import MLPRegressor
from model_base import ModelConfig


def set_seeds(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_csv(path: Path):
    import pandas as pd
    df = pd.read_csv(path)
    X = df[['apply_hadamard','apply_cnot','shots_norm']].values.astype(np.float32)
    y = df[['entropy','fidelity']].values.astype(np.float32)
    return X, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='data')
    p.add_argument('--output_dir', type=str, default='models/repro')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    set_seeds(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_csv(data_dir / 'train.csv')
    X_val, y_val = load_csv(data_dir / 'val.csv')
    X_test, y_test = load_csv(data_dir / 'test.csv')

    config = ModelConfig(
        model_type='mlp',
        input_dim=X_train.shape[1],
        output_dim=y_train.shape[1],
        hyperparameters={
            'hidden_layers': [512, 256, 128, 64],
            'dropout': 0.2,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'patience': 10,
            'mixed_precision': False
        }
    )
    model = MLPRegressor(config)

    metrics = model.train(X_train, y_train, X_val=X_val, y_val=y_val)

    # Evaluate on test
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)

    # Save artifacts
    model.save(str(out_dir))
    with open(out_dir / 'config.json', 'w') as f:
        json.dump({
            'input_dim': int(config.input_dim),
            'output_dim': int(config.output_dim),
            'hyperparameters': config.hyperparameters,
            'seed': args.seed
        }, f, indent=2)
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump({
            'val_mse': metrics.val_mse,
            'r2': metrics.r2_score,
            'test_r2': float(r2),
            'test_mse': float(mse)
        }, f, indent=2)

    print(f"Saved model to {out_dir}; Test R2={r2:.4f}, Test MSE={mse:.6f}")


if __name__ == '__main__':
    main()
