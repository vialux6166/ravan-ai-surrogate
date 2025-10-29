# Ravan Quantum-ML - Reproducible Training Release

## Overview
This repo snapshot provides:
- Dataset export script (CSV + HDF5)
- Reproducible MLP training script
- Pinned requirements
- Clear steps to reproduce model results

## Environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Export Dataset
Generates train/val/test splits with parameters (apply_hadamard, apply_cnot, shots_norm) and targets (entropy, fidelity).
```bash
python scripts/export_dataset.py --out_dir data --num_samples 3000 --seed 42
```
Outputs:
- data/train.csv, data/val.csv, data/test.csv
- data/train.h5, data/val.h5, data/test.h5

## Train Model
```bash
python scripts/train.py --data_dir data --output_dir models/repro --epochs 100 --batch_size 128 --lr 1e-3 --seed 42
```
Artifacts:
- models/repro/model.pt, config.json, metrics.json

## Serve (already included in project)
```bash
uvicorn serve:app --reload
```

## Streamlit UI (with UQ & Explain)
```bash
streamlit run streamlit_dashboard_v2.py
```

## License
MIT or Apache-2.0 (choose one and rename LICENSE accordingly).
