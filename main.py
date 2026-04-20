import os
import math
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")          # headless rendering – safe on any machine
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
VIZ_DIR = ROOT / "visualizations"
DATA_DIR = ROOT / "data"
VIZ_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# ===========================================================================
# Task 1 – PrunableLinear Layer (Custom Layer from Scratch)
# ===========================================================================

class PrunableLinear(nn.Module):
    """A fully-connected layer augmented with learnable gate scores.

    For every weight w_{ij} there is a corresponding gate score g_{ij}.
    During the forward pass:

        gate   = sigmoid(gate_score)          ∈ (0, 1)
        W_eff  = weight ⊙ gate               element-wise mask
        output = x @ W_eff.T + bias

    Both ``weight`` and ``gate_scores`` receive gradients, so the network
    learns (a) what each connection should carry and (b) whether it should
    exist at all.  Gate scores are initialised near zero so that initially
    all gates ≈ sigmoid(0) = 0.5 and the network behaves like a standard
    linear layer before pruning pressure takes effect.

    Args:
        in_features:  Number of input features.
        out_features: Number of output neurons.
        bias:         Whether to include an additive bias term (default True).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # ── Weights (standard Kaiming uniform init) ──────────────────────
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # ── Bias ─────────────────────────────────────────────────────────
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

        # ── Gate scores (same shape as weight; starts near 0 → gate≈0.5) ─
        #   Small random init prevents symmetry while keeping initial gates ≈ 0.5
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features) * 0.01)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute gated linear transformation.

        Steps
        -----
        1. gates     = sigmoid(gate_scores)   ∈ (0, 1)^{out × in}
        2. W_pruned  = weight ⊙ gates
        3. output    = F.linear(x, W_pruned, bias)

        Gradients flow through both ``weight`` and ``gate_scores`` via the
        chain rule through the sigmoid and the element-wise product.
        """
        gates: torch.Tensor = torch.sigmoid(self.gate_scores)   # (out, in)
        w_pruned: torch.Tensor = self.weight * gates             # (out, in)
        return F.linear(x, w_pruned, self.bias)                  # (batch, out)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def gate_values(self) -> torch.Tensor:
        """Return the current gate activations (detached, on CPU)."""
        return torch.sigmoid(self.gate_scores).detach().cpu()

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


# ===========================================================================
# Network Definition
# ===========================================================================

class SelfPruningNet(nn.Module):
    """Convolutional feature extractor + prunable fully-connected classifier.

    Architecture
    ------------
    • Conv block 1 : Conv2d(3→32)  → BN → ReLU → MaxPool
    • Conv block 2 : Conv2d(32→64) → BN → ReLU → MaxPool
    • Conv block 3 : Conv2d(64→128)→ BN → ReLU → AdaptiveAvgPool(4×4)
    • Flatten      : 128×4×4 = 2048
    • PrunableLinear: 2048 → 512
    • Dropout(0.4)
    • PrunableLinear: 512  → 256
    • Dropout(0.3)
    • PrunableLinear: 256  → 10   (logits over CIFAR-10 classes)

    Only the PrunableLinear layers participate in sparsity regularisation.
    The convolutional backbone provides stable feature extraction that is
    not pruned, which mirrors common practice in structured pruning.
    """

    def __init__(self) -> None:
        super().__init__()

        # ── Convolutional feature extractor ────────────────────────────
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),         # 32×32 → 16×16

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),         # 16×16 → 8×8

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # 8×8 → 4×4
        )

        # ── Prunable classifier ─────────────────────────────────────────
        self.fc1 = PrunableLinear(128 * 4 * 4, 512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = PrunableLinear(512, 256)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = PrunableLinear(256, 10)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.fc3(x)
        return x                   # raw logits

    # ------------------------------------------------------------------
    def prunable_layers(self) -> List[PrunableLinear]:
        """Return all PrunableLinear layers (used in sparsity loss)."""
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]


# ===========================================================================
# Task 2 – Sparsity Regularisation Loss
# ===========================================================================

def sparsity_loss(model: SelfPruningNet) -> torch.Tensor:
    """Compute the L1 norm of all gate activations across every prunable layer.

    L1 encourages gate values toward 0 because the gradient of |g| with
    respect to g is sign(g) = +1 for g > 0 and –1 for g < 0.  Since gates
    are constrained to (0,1) by the sigmoid, the gradient is always +1,
    giving a constant downward push on every gate.  Gates that are truly
    useful will resist this push (their classification gradient dominates);
    gates for weak connections will collapse to 0 and be effectively pruned.

    Args:
        model: The SelfPruningNet instance.

    Returns:
        Scalar tensor – sum of |gate_values| across all prunable layers.
    """
    total: torch.Tensor = torch.tensor(0.0, device=DEVICE, requires_grad=False)
    for layer in model.prunable_layers():
        gates = torch.sigmoid(layer.gate_scores)   # keep in computation graph
        total = total + gates.abs().sum()
    return total


def total_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    model: SelfPruningNet,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute combined classification + sparsity loss.

    Total Loss = CrossEntropy(logits, targets) + λ × L1_norm(gates)

    Args:
        logits:  Raw network outputs, shape (B, 10).
        targets: Ground-truth class indices, shape (B,).
        model:   The SelfPruningNet instance.
        lam:     Sparsity penalty coefficient λ.

    Returns:
        Tuple of (total_loss, cls_loss, sp_loss) – all scalar tensors.
    """
    cls_loss: torch.Tensor = F.cross_entropy(logits, targets)
    sp_loss: torch.Tensor = sparsity_loss(model)
    return cls_loss + lam * sp_loss, cls_loss, sp_loss


# ===========================================================================
# Data Loading
# ===========================================================================

def get_dataloaders(batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    """Download CIFAR-10 and return (train_loader, test_loader).

    Data augmentation on the training set:
      • Random horizontal flip
      • Random crop (pad=4)
      • Normalise with CIFAR-10 channel statistics

    The test set uses only normalisation.
    """
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR), train=True, download=True, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR), train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=256, shuffle=False, num_workers=2, pin_memory=True
    )
    return train_loader, test_loader


# ===========================================================================
# Training Loop
# ===========================================================================

def train_one_epoch(
    model: SelfPruningNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lam: float,
    epoch: int,
) -> Dict[str, float]:
    """Run one training epoch and return average losses + accuracy."""
    model.train()
    running_total = running_cls = running_sp = correct = total = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch:>3}", leave=False, ncols=90)
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)
        loss, cls, sp = total_loss(logits, labels, model, lam)
        loss.backward()

        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        bs = labels.size(0)
        running_total += loss.item() * bs
        running_cls   += cls.item()  * bs
        running_sp    += sp.item()   * bs
        correct       += logits.argmax(1).eq(labels).sum().item()
        total         += bs

        pbar.set_postfix(loss=f"{loss.item():.4f}", sp=f"{sp.item():.2f}")

    n = total
    return {
        "total_loss": running_total / n,
        "cls_loss":   running_cls   / n,
        "sp_loss":    running_sp    / n,
        "accuracy":   100.0 * correct / n,
    }


@torch.no_grad()
def evaluate(model: SelfPruningNet, loader: DataLoader) -> float:
    """Return test accuracy (%) on the given loader."""
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        preds = model(images).argmax(1)
        correct += preds.eq(labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


# ===========================================================================
# Task 3 – Evaluation & Reporting
# ===========================================================================

PRUNE_THRESHOLD = 1e-2   # gate value below this → considered pruned


def compute_sparsity(model: SelfPruningNet) -> float:
    """Return the percentage of gates below PRUNE_THRESHOLD.

    A gate g_ij < 1e-2 means the effective weight w_ij × g_ij contributes
    less than 1 % of its nominal magnitude – effectively pruned.

    Returns:
        Float in [0, 100] representing the percentage of pruned weights.
    """
    all_gates: List[torch.Tensor] = []
    for layer in model.prunable_layers():
        all_gates.append(layer.gate_values().flatten())
    all_gates_cat = torch.cat(all_gates)
    pruned = (all_gates_cat < PRUNE_THRESHOLD).sum().item()
    return 100.0 * pruned / all_gates_cat.numel()


def collect_all_gates(model: SelfPruningNet) -> np.ndarray:
    """Flatten and concatenate all gate values into a single numpy array."""
    parts: List[torch.Tensor] = []
    for layer in model.prunable_layers():
        parts.append(layer.gate_values().flatten())
    return torch.cat(parts).numpy()


def plot_gate_distribution(
    gates: np.ndarray,
    lam: float,
    sparsity: float,
    accuracy: float,
    label: str = "best",
) -> Path:
    """Save a histogram of gate value distribution to visualizations/.

    A well-pruned model shows:
      • A tall spike near 0  → many connections pruned
      • A cluster near 1     → retained strong connections
      • A valley in between  → the gate is decisive (bimodal)

    Args:
        gates:    1-D numpy array of gate values.
        lam:      The λ value used for this run.
        sparsity: Final sparsity percentage.
        accuracy: Final test accuracy (%).
        label:    String token used in the filename.

    Returns:
        Path to the saved figure.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(
        gates,
        bins=100,
        range=(0, 1),
        color="#4C72B0",
        edgecolor="white",
        linewidth=0.3,
        alpha=0.9,
    )

    # Threshold line
    ax.axvline(
        x=PRUNE_THRESHOLD,
        color="#DD4949",
        linewidth=1.6,
        linestyle="--",
        label=f"Prune threshold ({PRUNE_THRESHOLD})",
    )

    ax.set_title(
        f"Gate Value Distribution  |  λ={lam}  |  "
        f"Sparsity={sparsity:.1f}%  |  Acc={accuracy:.2f}%",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Gate Value  σ(g)  ∈ (0, 1)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    out_path = VIZ_DIR / f"gate_distribution_{label}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved gate distribution plot → {out_path}")
    return out_path


def plot_training_curves(
    history: List[Dict],
    lam: float,
    label: str,
) -> Path:
    """Save loss and accuracy training curves for a single run."""
    epochs   = [h["epoch"]    for h in history]
    cls_loss = [h["cls_loss"] for h in history]
    sp_loss  = [h["sp_loss"]  for h in history]
    train_acc= [h["train_acc"]for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # -- Loss curves --------------------------------------------------------
    axes[0].plot(epochs, cls_loss, label="Classification Loss", color="#4C72B0")
    axes[0].plot(epochs, sp_loss,  label="Sparsity Loss",       color="#DD8844")
    axes[0].set_title(f"Losses  (λ={lam})", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # -- Accuracy curve -----------------------------------------------------
    axes[1].plot(epochs, train_acc, label="Train Accuracy", color="#55A868")
    axes[1].set_title(f"Train Accuracy  (λ={lam})", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = VIZ_DIR / f"training_curves_{label}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved training curves → {out_path}")
    return out_path


def print_results_table(results: List[Dict]) -> None:
    """Print a formatted ASCII results table to stdout."""
    header = f"{'Lambda':>10} | {'Test Accuracy (%)':>18} | {'Sparsity Level (%)':>19}"
    sep    = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['lambda']:>10.4f} | "
            f"{r['test_accuracy']:>18.2f} | "
            f"{r['sparsity']:>19.2f}"
        )
    print(sep + "\n")


# ===========================================================================
# Full Experiment Runner
# ===========================================================================

def run_experiment(
    lam: float,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
) -> Dict:
    """Train a fresh SelfPruningNet for a single lambda value.

    Args:
        lam:          Sparsity penalty coefficient λ.
        train_loader: DataLoader for the training split.
        test_loader:  DataLoader for the test split.
        epochs:       Number of training epochs.
        lr:           Initial learning rate.

    Returns:
        Dictionary with keys: lambda, test_accuracy, sparsity, gates, history.
    """
    print(f"\n{'='*60}")
    print(f"  Running experiment  λ = {lam}")
    print(f"{'='*60}")

    model = SelfPruningNet().to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    history: List[Dict] = []

    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, lam, epoch)
        scheduler.step()

        # Periodic test evaluation (saves compute)
        if epoch % 5 == 0 or epoch == epochs:
            test_acc = evaluate(model, test_loader)
            print(
                f"  Epoch {epoch:>3}/{epochs}  "
                f"cls={train_stats['cls_loss']:.4f}  "
                f"sp={train_stats['sp_loss']:.2f}  "
                f"train_acc={train_stats['accuracy']:.2f}%  "
                f"test_acc={test_acc:.2f}%"
            )
        else:
            test_acc = None

        history.append({
            "epoch":     epoch,
            "cls_loss":  train_stats["cls_loss"],
            "sp_loss":   train_stats["sp_loss"],
            "train_acc": train_stats["accuracy"],
            "test_acc":  test_acc,
        })

    # Final evaluation
    final_test_acc = evaluate(model, test_loader)
    sparsity       = compute_sparsity(model)
    gates          = collect_all_gates(model)

    label = f"lambda_{str(lam).replace('.', '_')}"
    plot_gate_distribution(gates, lam, sparsity, final_test_acc, label=label)
    plot_training_curves(history, lam, label=label)

    print(f"\n[RESULT]  λ={lam}  →  Test Acc={final_test_acc:.2f}%  Sparsity={sparsity:.2f}%")

    return {
        "lambda":        lam,
        "test_accuracy": final_test_acc,
        "sparsity":      sparsity,
        "gates":         gates,
        "history":       history,
    }


# ===========================================================================
# Entry Point
# ===========================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for flexible experiment configuration."""
    parser = argparse.ArgumentParser(
        description="Self-Pruning Neural Network – CIFAR-10",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--lambdas", nargs="+", type=float,
        default=[1e-5, 1e-4, 1e-3],
        help=(
            "List of sparsity penalty coefficients λ to experiment with. "
            "At least three values are required to cover low/medium/high regimes."
        ),
    )
    parser.add_argument("--epochs",     type=int,   default=30,   help="Training epochs per run.")
    parser.add_argument("--batch-size", type=int,   default=128,  help="Mini-batch size.")
    parser.add_argument("--lr",         type=float, default=1e-3, help="Initial learning rate.")
    return parser.parse_args()


def main() -> None:
    """Orchestrate all experiments and report final results."""
    args = parse_args()

    if len(args.lambdas) < 3:
        raise ValueError(
            f"At least 3 lambda values required per project spec; got {len(args.lambdas)}."
        )

    print("\n**************************************************************")
    print("*      Self-Pruning Neural Network  |  CIFAR-10              *")
    print(f"*  λ values : {args.lambdas}")
    print(f"*  epochs   : {args.epochs}   batch_size : {args.batch_size}   lr : {args.lr}")
    print("**************************************************************\n")

    # ── Data ────────────────────────────────────────────────────────────
    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size)

    # ── Experiments ─────────────────────────────────────────────────────
    all_results: List[Dict] = []
    for lam in args.lambdas:
        result = run_experiment(lam, train_loader, test_loader, args.epochs, args.lr)
        all_results.append(result)

    # ── Results table ────────────────────────────────────────────────────
    print_results_table(all_results)

    # ── Best model gate plot (highest test accuracy) ─────────────────────
    best = max(all_results, key=lambda r: r["test_accuracy"])
    plot_gate_distribution(
        best["gates"],
        best["lambda"],
        best["sparsity"],
        best["test_accuracy"],
        label="best",
    )
    print(
        f"[INFO] Best λ={best['lambda']}  "
        f"→  Test Acc={best['test_accuracy']:.2f}%  "
        f"Sparsity={best['sparsity']:.2f}%"
    )

    # ── Persist results to JSON ──────────────────────────────────────────
    summary = [
        {k: v for k, v in r.items() if k not in ("gates", "history")}
        for r in all_results
    ]
    out_json = ROOT / "results.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Saved numeric results → {out_json}")


if __name__ == "__main__":
    main()
