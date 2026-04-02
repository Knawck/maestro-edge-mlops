"""
MAESTRO LSTM Edge Inference Architecture (< 2MB)
"""
import torch
import torch.nn as nn

class QualityLSTM(nn.Module):
    def __init__(self, input_size=6, hidden=64, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x): 
        # x shape: (batch, seq_len=20, features=6)
        out, _ = self.lstm(x)
        return self.sig(self.fc(out[:, -1, :]))