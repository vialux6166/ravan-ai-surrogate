#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
from pathlib import Path
import numpy as np

from quantum_circuit_simulator import QuantumCircuitSimulator
from hdf5_storage import HDF5Storage
from dataset import Dataset


def generate_samples(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    sim = QuantumCircuitSimulator(use_gpu=False)

    X_list = []
    y_list = []

    for _ in range(n):
        apply_hadamard = int(rng.integers(0, 2))
        apply_cnot = int(rng.integers(0, 2))
        shots_norm = float(rng.random())
        params = {
            'apply_hadamard': apply_hadamard,
            'apply_cnot': apply_cnot,
            'shots': int(max(100, int(shots_norm * 8192)))
        }
        res = sim.run_with_validation(params)
        entropy = float(res.observables.get('entropy', 0.0))
        fidelity = float(res.observables.get('fidelity_to_ghz', 0.0))

        X_list.append([apply_hadamard, apply_cnot, shots_norm])
        y_list.append([entropy, fidelity])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


def save_csv(x, y, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['apply_hadamard', 'apply_cnot', 'shots_norm', 'entropy', 'fidelity'])
        for xi, yi in zip(x, y):
            writer.writerow([xi[0], xi[1], xi[2], yi[0], yi[1]])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='data', type=str)
    p.add_argument('--num_samples', default=2000, type=int)
    p.add_argument('--seed', default=42, type=int)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, Y = generate_samples(args.num_samples, args.seed)

    # Split
    n = X.shape[0]
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    idx = np.arange(n)
    np.random.default_rng(args.seed).shuffle(idx)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]

    splits = {
        'train': (X[train_idx], Y[train_idx]),
        'val': (X[val_idx], Y[val_idx]),
        'test': (X[test_idx], Y[test_idx])
    }

    # Save CSVs
    for split, (x, y) in splits.items():
        save_csv(x, y, out_dir / f'{split}.csv')

    # Save HDF5
    storage = HDF5Storage(str(out_dir / 'dataset.h5'))
    for split, (x, y) in splits.items():
        ds = Dataset(X=x, y=y, parameter_names=['apply_hadamard','apply_cnot','shots_norm'], observable_names=['entropy','fidelity_to_ghz'])
        storage.save_dataset(ds, filepath=str(out_dir / f'{split}.h5'))

    print(f"Saved dataset to {out_dir}")


if __name__ == '__main__':
    main()
