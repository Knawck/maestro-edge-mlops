"""
MAESTRO LSTM Edge Inference Architecture (< 2MB)
"""
import torch
import torch.nn as nn

class QualityLSTM(nn.Module):
    def __init__(self, input_size=6, hidden=64, layers=2):
        super().__init__()
        # The core memory network
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True, dropout=0.2)
        # The final decision layer
        self.fc = nn.Linear(hidden, 1)
        # Squashes the output between 0 (Bad) and 1 (Good)
        self.sig = nn.Sigmoid()

    def forward(self, x): 
        # x shape: (batch_size, sequence_length=20, features=6)
        out, _ = self.lstm(x)
        # We only care about the prediction after the FINAL frame in the sequence
        return self.sig(self.fc(out[:, -1, :]))