import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import mlflow
import os
import argparse
from model import QualityLSTM

def train(data_path: str, epochs: int, out_dir: str):
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    features = df[["rtt_ms", "jitter_ms", "loss_pct", "ecn_fill", "dummy_1", "dummy_2"]].values
    labels = df["label"].values
    
    # Normalize data for the neural network
    features[:, 0] /= 100.0 
    features[:, 1] /= 10.0  
    features[:, 2] /= 100.0 
    
    # Create windows of 20 frames
    X, y = [], []
    seq_len = 20
    for i in range(len(features) - seq_len):
        X.append(features[i:i+seq_len])
        y.append(labels[i+seq_len])
        
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32).view(-1, 1)

    model = QualityLSTM()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # Start MLflow Tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    with mlflow.start_run(run_name="LSTM_Base_Training"):
        print(f"Training for {epochs} epochs...")
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            # Simple accuracy calculation
            predictions = (outputs > 0.5).float()
            accuracy = (predictions == y).float().mean().item()
            
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} - Acc: {accuracy:.4f}")
                mlflow.log_metric("loss", loss.item(), step=epoch)
                mlflow.log_metric("accuracy", accuracy, step=epoch)

        os.makedirs(out_dir, exist_ok=True)
        model_path = os.path.join(out_dir, "best.pt")
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)
        print(f"Training complete. Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/traces.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--out", type=str, default="models/")
    args = parser.parse_args()
    train(args.data, args.epochs, args.out)