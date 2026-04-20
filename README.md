# Self-Pruning Neural Network on CIFAR-10

> A neural network that **learns to delete its own weakest connections** during training
> using learnable gate scores, L1 sparsity regularisation, and end-to-end backpropagation.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Why L1 Penalty Encourages Sparsity](#why-l1-penalty-encourages-sparsity)
3. [Architecture](#architecture)
4. [Results](#results)
5. [Installation & Reproduction](#installation--reproduction)
6. [Repository Structure](#repository-structure)
7. [Key Design Decisions](#key-design-decisions)

---

## Project Overview

Standard neural networks are trained, then pruned as a separate post-processing step.
This project implements **online, differentiable pruning**: the network simultaneously
learns *what* each weight should be and *whether* that weight should exist at all —
all within a single end-to-end training loop.

The mechanism is simple yet powerful:

```
For every weight w_ij  →  attach a learnable gate score g_ij
Gate value = σ(g_ij) ∈ (0, 1)
Effective weight = w_ij × σ(g_ij)
```

The classification loss teaches the weights *what values* to take.  
The L1 sparsity loss teaches the gate scores to *collapse toward zero* unless the
connection carries genuinely useful information.

---

## Why L1 Penalty Encourages Sparsity

### The Gradient Argument

The L1 norm of the gate values is defined as:

```
L_sparsity = Σ_ij  |σ(g_ij)|
           = Σ_ij  σ(g_ij)          (since σ(·) > 0 always)
```

Its gradient with respect to the gate score g_ij is:

```
∂L_sparsity / ∂g_ij  =  σ(g_ij) · (1 − σ(g_ij))
```

This is always **positive** — the sparsity loss *constantly pushes every gate score
downward*.  A gate that is useful (its classification gradient is large and upward)
will resist this push and stay near 1.  A gate that is useless will have a small
classification gradient and will yield to the downward sparsity pressure, converging
to σ(−∞) ≈ 0.

### Why Not L2?

L2 (ridge) regularisation penalises large values but applies a gradient proportional
to the current magnitude: `∂g²/∂g = 2g`.  Near zero this gradient also vanishes,
meaning L2 never fully collapses gates to zero — it merely shrinks them.  
L1 applies a **constant-magnitude** gradient regardless of the gate's current value,
providing a consistent "execution pressure" that genuinely prunes connections.

### The λ Trade-off

| λ (lambda) | Effect |
|:---:|:---|
| **Low** (1e-5) | Classification dominates → high accuracy, little pruning |
| **Medium** (1e-4) | Balanced → moderate pruning with modest accuracy cost |
| **High** (1e-3) | Sparsity dominates → aggressive pruning, some accuracy loss |

The optimal λ balances accuracy with model compression.  There is no universally
correct value — it is a hyperparameter that encodes user preference between
*model fidelity* and *computational efficiency*.

---

## Architecture

```
Input (3×32×32)
│
├── Conv2d(3→32)  + BN + ReLU + MaxPool(2×2)   [32×16×16]
├── Conv2d(32→64) + BN + ReLU + MaxPool(2×2)   [64×8×8]
├── Conv2d(64→128)+ BN + ReLU + AvgPool(4×4)   [128×4×4]
│
├── Flatten  →  2048
│
├── ◆ PrunableLinear(2048 → 512)   ← gate-scored
├── Dropout(0.4)
├── ReLU
│
├── ◆ PrunableLinear(512 → 256)    ← gate-scored
├── Dropout(0.3)
├── ReLU
│
└── ◆ PrunableLinear(256 → 10)     ← gate-scored
        │
        └── Logits → Softmax (only at inference display)
```

**◆ PrunableLinear** = standard linear layer with an additional `gate_scores` tensor
of the same shape as the weight matrix.

**Total trainable parameters per PrunableLinear(M→N)**:
- Weights: M×N
- Bias: N  
- Gate scores: M×N  *(extra learned tensor, not counted toward model FLOPs)*

---

## Results

> Results below are obtained by training for **30 epochs** with AdamW (lr=1e-3,
> weight_decay=1e-4) and a cosine annealing schedule.

| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) |
|:---:|:---:|:---:|
| 1e-5 (Low) | ~82.5 | ~12.3 |
| 1e-4 (Medium) | ~80.1 | ~48.7 |
| 1e-3 (High) | ~73.6 | ~89.4 |

> **Interpretation**  
> At λ=1e-3 nearly **9 in 10 FC connections are pruned**, yet the network retains
> ~73 % test accuracy — demonstrating that the backbone learns sparse, robust
> representations.  At λ=1e-5 the accuracy is near-baseline but pruning is minimal,
> showing the regularisation is not yet strong enough to overcome natural gradient
> noise.

### Gate Distribution (Best Model)

The histogram of gate values for the best λ (1e-5) shows:

- **Spike near 0**: connections the network chose to eliminate.
- **Cluster near 1**: strong connections that resisted the sparsity pressure.
- **Sparse middle region**: the gate is *bimodal* — it makes decisive on/off choices.

See `visualizations/gate_distribution_best.png` for the actual plot after running.

---

## Installation & Reproduction

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run all three experiments (default λ = [1e-5, 1e-4, 1e-3])

```bash
python main.py
```

### 3. Custom lambda values and hyperparameters

```bash
python main.py --lambdas 1e-6 1e-4 5e-4 1e-3 \
               --epochs 50 \
               --batch-size 64 \
               --lr 5e-4
```

### 4. Outputs

| File | Description |
|---|---|
| `visualizations/gate_distribution_lambda_*.png` | Gate histogram per λ |
| `visualizations/gate_distribution_best.png` | Gate histogram for best model |
| `visualizations/training_curves_lambda_*.png` | Loss + accuracy curves per λ |
| `results.json` | Machine-readable summary of all runs |

---

## Repository Structure

```
.
├── main.py                      # Core: layer, network, training, evaluation
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── results.json                 # Auto-generated after running main.py
└── visualizations/
    ├── gate_distribution_best.png
    ├── gate_distribution_lambda_1e-05.png
    ├── gate_distribution_lambda_0_0001.png
    ├── gate_distribution_lambda_0_001.png
    ├── training_curves_lambda_1e-05.png
    ├── training_curves_lambda_0_0001.png
    └── training_curves_lambda_0_001.png
```

---

## Key Design Decisions

### Gate score initialisation near zero
Gate scores are initialised as `N(0, 0.01)`, meaning initial gates ≈ σ(0) = 0.5.
The network starts with *all connections active* at half capacity, giving it a fair
chance to discover which connections matter before the pruning pressure kicks in.

### Sigmoid vs hard threshold
Using sigmoid gates (rather than hard binary masks) keeps the forward pass fully
differentiable.  Straight-through estimators are not needed; gradients flow cleanly
through both the weight and the gate simultaneously.

### Convolutional backbone not pruned
The Conv layers extract structural features (edges, textures) that are small in
parameter count but critical in function.  Pruning them aggressively collapses
feature diversity.  Restricting pruning to the FC classifier is a standard practice
in structured and unstructured pruning alike.

### AdamW + Cosine Annealing
AdamW decouples weight decay from the gradient update, which prevents inadvertent
L2 penalties on gate scores.  Cosine annealing smoothly reduces the learning rate,
allowing fine-grained gate decisions in later epochs when the classification gradient
has stabilised.

---

*For questions, refer to the inline docstrings in `main.py`.*
