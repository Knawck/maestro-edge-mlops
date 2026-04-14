import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
import os
import argparse
from clearml import Task
from model import get_model

# --- Configuration ---
SEQ_LEN = 20
INPUT_SIZE = 6
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 300  # Bumped from 150
MODEL_DIR = "models"
DATA_PATH = "data/traces.csv"

# Normalization constants
RTT_NORM, JITTER_NORM, LOSS_NORM, ECN_NORM = 150.0, 30.0, 20.0, 1.0

def load_dataset(csv_path):
    print(f"Loading dataset from {csv_path} ...")
    df = pd.read_csv(csv_path)
    
    n_samples = df["sample_id"].nunique()
    
    X = np.zeros((n_samples, SEQ_LEN, 6), dtype=np.float32)
    y = np.zeros((n_samples, 1), dtype=np.float32)
    
    for sample_id, group in df.groupby("sample_id"):
        group = group.sort_values("frame")
        idx = int(sample_id) # type: ignore
        
        # Capture temporal features
        for t, (_, row) in enumerate(group.iterrows()):
            X[idx, t] = [
                row["rtt"] / RTT_NORM,
                row["jitter"] / JITTER_NORM,
                row["loss"] / LOSS_NORM,
                row["ecn"] / ECN_NORM,
                0.0, 0.0 # Dummies
            ]
        y[idx, 0] = group["label"].iloc[-1]
        
    print(f"X shape : {X.shape}")
    print(f"y shape : {y.shape}")
    
    # Calculate Label Balance for Weighted Loss
    num_healthy = np.sum(y)
    num_degraded = len(y) - num_healthy
    print(f"Label balance: {int(num_degraded)} degraded (0) / {int(num_healthy)} healthy (1)")
    
    # Weight for Healthy class (Positive class)
    pos_weight = num_degraded / num_healthy if num_healthy > 0 else 1.0
    
    return torch.tensor(X), torch.tensor(y), pos_weight

def train(no_clearml=False):
    if not no_clearml:
        task = Task.init(project_name="Maestro-PoC-02", task_name="LSTM Quality Predictor - Optimized")
        logger = task.get_logger()
    else:
        logger = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y, pos_weight_val = load_dataset(DATA_PATH)
    
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = get_model().to(device)
    
    # Weighted Loss helps break the 82% plateau
    pos_weight = torch.tensor([pos_weight_val]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Scheduler: Shrinks LR when progress stalls
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    os.makedirs(MODEL_DIR, exist_ok=True)
    best_val_acc = 0.0

    print(f"\nStarting Optimized Training — {EPOCHS} epochs\n")
    print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | {'Val Acc':>10} | {'Status'}")
    print("-" * 65)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation pass
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == batch_y).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = (correct / val_size) * 100
        
        # Step the scheduler
        scheduler.step(avg_val_loss)

        if logger:
            logger.report_scalar("Loss", "Train", iteration=epoch, value=avg_train_loss)
            logger.report_scalar("Loss", "Val", iteration=epoch, value=avg_val_loss)
            logger.report_scalar("Accuracy", "Val", iteration=epoch, value=val_acc)

        status = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best.pt"))
            status = "✅ saved"

        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:6} | {avg_train_loss:12.6f} | {avg_val_loss:12.6f} | {val_acc:9.2f}% | {status}")

    print("-" * 65)
    print(f"Optimized Training Complete. Best Acc: {best_val_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-clearml", action="store_true")
    args = parser.parse_args()
    train(no_clearml=args.no_clearml)