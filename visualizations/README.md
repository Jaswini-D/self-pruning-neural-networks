# Visualizations

This folder is populated automatically when `main.py` is run.

## Expected Contents

| File | Description |
|---|---|
| `gate_distribution_best.png` | Gate histogram for the highest-accuracy λ |
| `gate_distribution_lambda_*.png` | Gate histogram per individual λ run |
| `training_curves_lambda_*.png` | Classification + sparsity loss curves per run |

## Interpretation Guide

A successful pruning run produces a **bimodal** gate histogram:
- Tall spike near **0** → pruned connections
- Dense cluster near **1** → retained, strong connections
- Near-empty valley in between → the gate is decisive (not indeterminate)
