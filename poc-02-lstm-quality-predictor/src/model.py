"""
MAESTRO PoC-02 — LSTM Model Architecture

A 2-layer LSTM designed for edge deployment:
- Input  : (batch, sequence_length=20, features=6)
- Output : (batch, 1) — quality score between 0.0 and 1.0

Score interpretation:
  > 0.5  → network is healthy
  < 0.5  → network is degraded, trigger fallback

Size: < 2MB saved on disk.
Inference: < 5ms on CPU.
"""

import torch
import torch.nn as nn


class QualityLSTM(nn.Module):

    def __init__(self,
                 input_size:  int = 6,
                 hidden_size: int = 64,
                 num_layers:  int = 2,
                 dropout:     float = 0.2):
        """
        Args:
            input_size  : number of features per frame (RTT, Jitter, Loss, ECN, D1, D2)
            hidden_size : neurons in each LSTM layer (64 keeps it lightweight)
            num_layers  : depth of LSTM stack
            dropout     : regularisation — prevents overfitting between layers
        """
        super(QualityLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,       # input shape is (batch, seq, features)
            dropout     = dropout,
        )

        self.fc  = nn.Linear(hidden_size, 1)
        self.sig = nn.Sigmoid()       # squashes output to 0.0–1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : tensor of shape (batch, 20, 6)
        Returns:
            tensor of shape (batch, 1) — quality scores
        """
        lstm_out, _ = self.lstm(x)
        # Take only the output from the LAST time step
        # This is the model's "verdict" after seeing all 20 frames
        last_step = lstm_out[:, -1, :]
        return self.sig(self.fc(last_step))


def get_model() -> QualityLSTM:
    """Returns a fresh untrained model instance."""
    return QualityLSTM(input_size=6, hidden_size=64, num_layers=2, dropout=0.2)


def load_model(checkpoint_path: str) -> QualityLSTM:
    """Loads a trained model from a .pt checkpoint file."""
    model = get_model()
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    model.eval()
    return model


if __name__ == "__main__":
    # Quick sanity check — run this file directly to verify the model works
    model = get_model()
    dummy_input = torch.randn(4, 20, 6)    # batch of 4 samples
    output = model(dummy_input)
    print(f"Model output shape : {output.shape}")    # should be torch.Size([4, 1])
    print(f"Sample scores      : {output.detach().numpy().flatten()}")
    print(f"All scores in 0-1  : {all(0 <= s <= 1 for s in output.detach().numpy().flatten())}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters   : {total_params:,}")