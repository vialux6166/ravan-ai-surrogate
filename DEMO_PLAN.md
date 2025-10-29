# Demo Plan

## Demo 1: Direct Simulation (30s)
1) Start API
```bash
uvicorn serve:app --reload
```
2) Launch UI
```bash
streamlit run streamlit_dashboard_v2.py
```
3) Actions
- Tab: Direct Simulation
- Set Hadamard=1, CNOT=1, Shots≈1000
- Click Run Simulation
- Narrate metrics and UQ checkbox on/off

## Demo 2: Inverse Design (45s)
1) Tab: Inverse Design
- Set target entropy=0.95, fidelity=0.95
- Method: L-BFGS-B, Max Iter=100
- Click Run Inverse Design
2) Show
- Optimal parameters JSON
- Resulting values
- Error metrics
- Heatmaps with red X

## Demo 3: Explainability (30s)
1) Back to Direct Simulation
- Click “Explain Current Parameters”
2) Show
- Entropy and Fidelity sensitivity bars
- Explain which inputs matter most

## Recording
- Use system screen recorder at 60 FPS
- 1080p resolution recommended
- Keep cursor movements deliberate and slow
