"""
MAESTRO PoC-02 — LSTM Trainer

Trains the QualityLSTM model on reverting-walk network telemetry.
Uses ClearML for experiment tracking.

Key design decisions:
- Full-batch training (all data at once) works well here because the
  dataset is small enough to fit in memory and the reverting walk data
  is already well-shuffled at generation time.
- 150 epochs consistently converges above 99% accuracy.
- Normalization is critical: RTT/150, Jitter/30, Loss/20 keeps all
  features in a similar range so the LSTM doesn't over-weight large numbers.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import json
import argparse
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from model import get_model

# ── Normalization constants (must match serve_api.py exactly) ────────────────
RTT_NORM    = 150.0
JITTER_NORM =  30.0
LOSS_NORM   =  20.0
ECN_NORM    =   1.0    # already 0–1, no scaling needed


def normalize(rtt, jitter, loss, ecn):
    """Apply normalization. Same function used in training AND inference."""
    return np.array([
        rtt    / RTT_NORM,
        jitter / JITTER_NORM,
        loss   / LOSS_NORM,
        ecn    / ECN_NORM,
        0.0,    # dummy1
        0.0,    # dummy2
    ], dtype=np.float32)


def load_dataset(csv_path: str):
    """
    Reads the CSV produced by generate_training_data.py and reshapes it
    into tensors the LSTM can consume.

    Returns:
        X : torch.Tensor of shape (n_samples, 20, 6)
        y : torch.Tensor of shape (n_samples, 1)
    """
    print(f"Loading dataset from {csv_path} ...")
    df = pd.read_csv(csv_path)

    n_samples = df["sample_id"].nunique()
    seq_len   = df["frame"].nunique()

    X = np.zeros((n_samples, seq_len, 6), dtype=np.float32)
    y = np.zeros((n_samples, 1),          dtype=np.float32)

    for sample_id, group in df.groupby("sample_id"):
        group = group.sort_values("frame")
        
        # Explicitly cast to int to satisfy Pylance
        s_id = int(sample_id)  # type: ignore
        
        for t, (_, row) in enumerate(group.iterrows()):
            X[s_id, t] = normalize(
                row["rtt"], row["jitter"], row["loss"], row["ecn"]
            )
        # Apply same cast here
        y[s_id, 0] = group["label"].iloc[-1]

    print(f"X shape : {X.shape}")
    print(f"y shape : {y.shape}")
    print(f"Label balance: {y.sum():.0f} degraded / {(1-y).sum():.0f} healthy")

    return torch.tensor(X), torch.tensor(y)


def train(data_path:  str = "data/traces.csv",
          model_dir:  str = "models",
          epochs:     int = 150,
          lr:         float = 0.001,
          val_split:  float = 0.2,
          use_clearml: bool = True):

    # ── ClearML tracking ────────────────────────────────────────────────────
    task = None
    if use_clearml:
        try:
            from clearml import Task
            task = Task.init(
                project_name = "MAESTRO",
                task_name    = f"poc-02-lstm-training-{int(time.time())}",
            )
            task.connect({
                "epochs":      epochs,
                "lr":          lr,
                "val_split":   val_split,
                "rtt_norm":    RTT_NORM,
                "jitter_norm": JITTER_NORM,
                "loss_norm":   LOSS_NORM,
            })
            print("ClearML tracking active.")
        except Exception as e:
            print(f"ClearML not available ({e}). Training without tracking.")
            task = None

    # ── Load data ───────────────────────────────────────────────────────────
    X, y = load_dataset(data_path)

    # Train/validation split — stratified to keep label balance equal
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=42,
        stratify=y.numpy().flatten().astype(int)
    )

    print(f"\nTrain samples : {len(X_train)}")
    print(f"Val samples   : {len(X_val)}")

    # ── Model, loss, optimiser ───────────────────────────────────────────────
    model     = get_model()
    criterion = nn.BCELoss()           # Binary Cross Entropy for 0/1 labels
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs(model_dir, exist_ok=True)

    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    print(f"\nStarting training — {epochs} epochs\n")
    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>10}  {'Val Acc':>10}  {'Status':>10}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):

        # ── Training pass ───────────────────────────────────────────────────
        model.train()
        optimiser.zero_grad()
        preds = model(X_train)
        loss  = criterion(preds, y_train)
        loss.backward()
        optimiser.step()
        train_loss = loss.item()

        # ── Validation pass ─────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss  = criterion(val_preds, y_val).item()
            val_labels_pred = (val_preds.numpy() > 0.5).astype(int).flatten()
            val_labels_true = y_val.numpy().flatten().astype(int)
            val_acc = accuracy_score(val_labels_true, val_labels_pred)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save best model
        status = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(model_dir, "best.pt"))
            status = "✅ saved"

        # Print every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>10.6f}  "
                  f"{val_acc*100:>9.2f}%  {status}")

        # Log to ClearML
        if task:
            logger = task.get_logger()
            logger.report_scalar("Loss",     "Train", train_loss, epoch)
            logger.report_scalar("Loss",     "Val",   val_loss,   epoch)
            logger.report_scalar("Accuracy", "Val",   val_acc,    epoch)

    print("-" * 60)
    print(f"\nTraining complete.")
    print(f"Best validation accuracy : {best_val_acc * 100:.2f}%")
    print(f"Best model saved to      : {model_dir}/best.pt")

    # Save training history to JSON for the README results table
    history["best_val_acc"] = best_val_acc #type:ignore
    with open(os.path.join(model_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    if task:
        task.close()

    return best_val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        type=str,   default="data/traces.csv")
    parser.add_argument("--models",      type=str,   default="models")
    parser.add_argument("--epochs",      type=int,   default=150)
    parser.add_argument("--lr",          type=float, default=0.001)
    parser.add_argument("--no-clearml",  action="store_true",
                        help="Skip ClearML tracking (useful if not configured)")
    args = parser.parse_args()

    train(
        data_path    = args.data,
        model_dir    = args.models,
        epochs       = args.epochs,
        lr           = args.lr,
        use_clearml  = not args.no_clearml,
    )