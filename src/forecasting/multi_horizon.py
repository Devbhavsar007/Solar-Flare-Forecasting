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

def load_multi_horizon(path: str = "models/multi_horizon.pt", n_features: int = 9, embed_dim: int = 64) -> MultiHorizonForecaster:
    model = MultiHorizonForecaster(n_features=n_features, embed_dim=embed_dim)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model


def train_multi_horizon(
    X_tr: np.ndarray,
    y_tr_dict: dict,
    X_val: np.ndarray,
    y_val_dict: dict,
    n_features: int = 9,
    embed_dim: int = 64,
    n_classes: int = 4,
    lr: float = 1e-3,
    epochs: int = 80,
    batch_size: int = 256,
    patience: int = 10,
    device: str = "cpu",
    tcn_encoder_path: str = "models/tcn_encoder.pt",
    save_path: str = "models/multi_horizon.pt",
) -> MultiHorizonForecaster:
    """
    Train MultiHorizonForecaster on real windowed data.

    Targets for h15/h30/h60 are passed as dicts: {"h15": y_array, "h30": y_array, "h60": y_array}.
    Optimizes compute_multi_horizon_loss() directly.

    If tcn_encoder_path exists, initializes the encoder backbone from Task 1's
    trained TCN checkpoint (transfer learning) instead of random init.

    Args:
        X_tr: (N, T, F) training windows.
        y_tr_dict: {"h15": (N,), "h30": (N,), "h60": (N,)} training labels.
        X_val: (N, T, F) validation windows.
        y_val_dict: same format as y_tr_dict.
        tcn_encoder_path: path to Task 1's trained TCN encoder state_dict.

    Returns:
        Trained MultiHorizonForecaster.
    """
    import os
    from sklearn.metrics import f1_score

    model = MultiHorizonForecaster(n_features=n_features, embed_dim=embed_dim)

    # Transfer learning: load Task 1's trained TCN encoder backbone
    if os.path.exists(tcn_encoder_path):
        tcn_state = torch.load(tcn_encoder_path, map_location="cpu", weights_only=True)
        model.encoder.load_state_dict(tcn_state)
        print(f"  MultiHorizon: initialized encoder from {tcn_encoder_path} (transfer learning)")
    else:
        print(f"  MultiHorizon: {tcn_encoder_path} not found, training encoder from scratch")

    model.to(device)

    # Class-weighted NLLLoss — same formula
    classes, counts = np.unique(y_tr_dict["h15"], return_counts=True)
    freq = counts / counts.sum()
    weight_map = {int(c): min(1.0 / max(f, 1e-8), 5000.0) for c, f in zip(classes, freq)}
    class_weights = torch.zeros(n_classes, device=device)
    for c, w in weight_map.items():
        if c < n_classes:
            class_weights[c] = w
    class_weights[class_weights == 0] = 1.0
    print(f"  MultiHorizon class weights (from h15): {class_weights.cpu().tolist()}")

    # Weighted NLLLoss for compute_multi_horizon_loss
    loss_fn = nn.NLLLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    horizon_keys = sorted(y_tr_dict.keys())
    y_tr_tensors = {k: torch.tensor(v, dtype=torch.long) for k, v in y_tr_dict.items()}
    y_val_tensors = {k: torch.tensor(v, dtype=torch.long) for k, v in y_val_dict.items()}

    best_f1 = -1.0
    best_state = None
    epochs_no_improve = 0

    print(f"\n--- MultiHorizon Training ({epochs} max epochs, patience={patience}) ---")
    print(f"  Horizons: {horizon_keys}")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        perm = torch.randperm(len(X_tr))
        for start in range(0, len(X_tr), batch_size):
            idx = perm[start:start + batch_size]
            xb = torch.tensor(X_tr[idx.numpy()], dtype=torch.float32, device=device)
            targets = {k: y_tr_tensors[k][idx].to(device) for k in horizon_keys}

            outputs = model(xb)

            # Weighted multi-horizon loss
            total_loss = torch.tensor(0.0, device=device)
            for h_key in horizon_keys:
                log_probs = torch.log(outputs[h_key] + 1e-8)
                total_loss = total_loss + loss_fn(log_probs, targets[h_key])

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation — average macro-F1 across horizons
        model.eval()
        horizon_f1s = []
        with torch.no_grad():
            for h_key in horizon_keys:
                val_preds = []
                for start in range(0, len(X_val), batch_size):
                    xb = torch.tensor(X_val[start:start + batch_size], dtype=torch.float32, device=device)
                    outputs = model(xb)
                    val_preds.append(outputs[h_key].argmax(dim=-1).cpu().numpy())
                val_preds = np.concatenate(val_preds)
                h_f1 = f1_score(y_val_dict[h_key], val_preds, average="macro", zero_division=0.0)
                horizon_f1s.append(h_f1)

        avg_val_f1 = np.mean(horizon_f1s)
        f1_str = " | ".join(f"{k}={f:.4f}" for k, f in zip(horizon_keys, horizon_f1s))
        print(f"  Epoch {epoch+1:3d} | train_loss={avg_loss:.4f} | avg_val_F1={avg_val_f1:.4f} ({f1_str})")

        if avg_val_f1 > best_f1:
            best_f1 = avg_val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1} (best avg val macro-F1={best_f1:.4f})")
            break

    if best_state is None:
        print("  WARNING: No improvement observed. Saving final model state.")
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    save_multi_horizon(model, save_path)
    print(f"  MultiHorizon saved: {save_path} (best avg val macro-F1={best_f1:.4f})")
    print("--- MultiHorizon Training complete ---\n")

    return model
