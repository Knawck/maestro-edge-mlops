"""
MAESTRO PoC-02 — Training Results Visualiser
Generates a chart of training loss and validation accuracy over epochs.
Run after training completes.
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def plot(history_path: str = "models/training_history.json",
         out_path:     str = "results/training_curves.png"):

    if not os.path.exists(history_path):
        print(f"History file not found: {history_path}")
        print("Run train_lstm.py first.")
        return

    with open(history_path) as f:
        h = json.load(f)

    epochs     = list(range(1, len(h["train_loss"]) + 1))
    train_loss = h["train_loss"]
    val_loss   = h["val_loss"]
    val_acc    = [a * 100 for a in h["val_acc"]]
    best_acc   = h["best_val_acc"] * 100

    fig = plt.figure(figsize=(13, 5))
    fig.suptitle(
        f"MAESTRO PoC-02 — LSTM Training Results\n"
        f"Best Validation Accuracy: {best_acc:.2f}%  "
        f"(Target: >90%  ·  Achieved: {'✅ PASS' if best_acc > 90 else '❌ FAIL'})",
        fontsize=12, fontweight="bold"
    )

    gs  = gridspec.GridSpec(1, 2, wspace=0.35)

    # Loss curves
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, train_loss, color="#1D4ED8", linewidth=1.2, label="Train Loss")
    ax1.plot(epochs, val_loss,   color="#DC2626", linewidth=1.2, label="Val Loss",
             linestyle="--")
    ax1.set_title("Loss Over Epochs", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curve
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, val_acc, color="#059669", linewidth=1.5, label="Val Accuracy")
    ax2.axhline(90, color="orange", linestyle="--", linewidth=1, label="90% target")
    ax2.axhline(best_acc, color="#7C3AED", linestyle=":",
                linewidth=1, label=f"Best: {best_acc:.2f}%")
    ax2.fill_between(epochs, 90, val_acc,
                     where=[a >= 90 for a in val_acc],
                     color="#059669", alpha=0.1, label="Above target")
    ax2.set_title("Validation Accuracy Over Epochs", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(50, 101)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Training curve saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    plot()