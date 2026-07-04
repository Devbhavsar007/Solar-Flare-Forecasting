import random, numpy as np, torch
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

import torch.nn as nn

class CausalLSTMForecaster(nn.Module):
    """
    LSTM-based forecaster ensuring causality.
    """
    def __init__(self, n_features: int, hidden_dim: int = 128, n_layers: int = 2, dropout: float = 0.3, n_classes: int = 4):
        super().__init__()
        # [RULE-2] bidirectional=False is strictly required
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=False
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F)
        returns: (B, n_classes)
        """
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]  # Take the last time step output
        return self.head(last_out)

def save_lstm(model: CausalLSTMForecaster, path: str = "models/causal_lstm.pt"):
    """[RULE-11, RULE-13] Save model state_dict."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_lstm(path: str = "models/causal_lstm.pt", n_features: int = 8) -> CausalLSTMForecaster:
    """[RULE-11, RULE-13] Load model state_dict."""
    model = CausalLSTMForecaster(n_features=n_features)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model
