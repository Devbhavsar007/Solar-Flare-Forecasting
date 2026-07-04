import random, numpy as np, torch
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

import torch.nn as nn
from src.nowcasting.tcn_encoder import TCNEncoder

class MultiHorizonForecaster(nn.Module):
    """
    Multi-horizon forecaster using a shared TCNEncoder backbone.
    Outputs predictions for multiple horizons (e.g., 15, 30, 60 mins).
    """
    def __init__(self, n_features: int, embed_dim: int = 256, horizons: list[int] = None):
        super().__init__()
        if horizons is None:
            horizons = [15, 30, 60]
        self.horizons = horizons
        
        self.encoder = TCNEncoder(n_features=n_features, embed_dim=embed_dim)
        
        # Separate head per horizon
        self.heads = nn.ModuleDict({
            f"h{h}": nn.Sequential(
                nn.Linear(embed_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 4),
                nn.Softmax(dim=-1)
            ) for h in horizons
        })

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        x: (B, T, F)
        returns: dict {"h15": (B,4), "h30": (B,4), "h60": (B,4)}
        """
        emb = self.encoder(x)
        return {name: head(emb) for name, head in self.heads.items()}

def compute_multi_horizon_loss(outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Loss: sum of CrossEntropyLoss for each horizon head.
    Since outputs are probabilities from Softmax, we should use NLLLoss with log(outputs) 
    or we can assume targets are class indices. Wait, nn.CrossEntropyLoss expects raw logits.
    Since we have Softmax at the end, we'll use NLLLoss on log(probs) or just compute it manually.
    Actually, to match the prompt simply: sum of CrossEntropyLoss. (We will use NLLLoss on log(output + eps)).
    """
    loss_fn = nn.NLLLoss()
    total_loss = 0.0
    for h_key in outputs:
        # Cross entropy loss on probabilities: NLLLoss(log(p), target)
        log_probs = torch.log(outputs[h_key] + 1e-8)
        total_loss += loss_fn(log_probs, targets[h_key])
    return total_loss

def save_multi_horizon(model: MultiHorizonForecaster, path: str = "models/multi_horizon.pt"):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_multi_horizon(path: str = "models/multi_horizon.pt", n_features: int = 8) -> MultiHorizonForecaster:
    model = MultiHorizonForecaster(n_features=n_features)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model
