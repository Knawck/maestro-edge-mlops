import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from clearml import Task
from model import QualityLSTM

def train():
    task = Task.init(project_name='Maestro', task_name='LSTM_Quality_Predictor_v2_Optimized')
    logger = task.get_logger()

    df = pd.read_csv("data/traces.csv")
    feat = df[["rtt", "jitter", "loss", "ecn", "d1", "d2"]].values
    
    # Precision Scaling: RTT/150, Jitter/30, Loss/20
    feat[:, 0] /= 150.0
    feat[:, 1] /= 30.0
    feat[:, 2] /= 20.0
    
    X, y = [], []
    for i in range(len(feat) - 20):
        X.append(feat[i:i+20])
        y.append(df["label"].values[i+20])
        
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32).view(-1, 1)

    model = QualityLSTM()
    criterion = nn.BCELoss()
    # LOWER LEARNING RATE FOR PRECISION
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting Deep Training (150 Epochs)...")
    for epoch in range(150):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        acc = ((outputs > 0.5).float() == y).float().mean().item()
        
        logger.report_scalar("Metrics", "Loss", iteration=epoch, value=loss.item())
        logger.report_scalar("Metrics", "Accuracy", iteration=epoch, value=acc)

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/150 | Accuracy: {acc:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/best.pt")
    task.close()

if __name__ == "__main__":
    train()