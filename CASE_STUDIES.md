# Case Studies

## Case Study 1: High-Fidelity Bell State Design
- Objective: Achieve fidelity-to-GHZ ≥ 0.95
- Method: Inverse Design (Nelder-Mead) on [apply_hadamard, apply_cnot, shots_norm]
- Result:
  - Optimal parameters: [apply_hadamard=1.0, apply_cnot=1.0, shots_norm≈0.50]
  - Achieved metrics: entropy=0.9542, fidelity=0.9675
  - Time to solution: < 200 ms (RTX 3090, ORT GPU)
  - Speedup vs simulator: orders-of-magnitude (interactive)

## Case Study 2: Parameter Sensitivity Mapping
- Objective: Understand input influence on entropy and fidelity
- Method: Gradient-based sensitivities via /explain endpoint
- Result:
  - Entropy sensitivities: hadamard > cnot ≈ shots_norm (example session)
  - Fidelity sensitivities: cnot > hadamard > shots_norm (example session)
  - Interpretation: entangling operations dominate fidelity; entropy responds to superposition and measurement count
