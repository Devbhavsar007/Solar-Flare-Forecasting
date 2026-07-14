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

def load_lstm(path: str = "models/causal_lstm.pt", n_features: int = 9) -> CausalLSTMForecaster:
    """[RULE-11, RULE-13] Load model state_dict."""
    model = CausalLSTMForecaster(n_features=n_features)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model


def train_causal_lstm(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_features: int = 9,
    n_classes: int = 4,
    lr: float = 1e-3,
    epochs: int = 80,
    batch_size: int = 256,
    patience: int = 10,
    device: str = "cpu",
    save_path: str = "models/causal_lstm.pt",
) -> CausalLSTMForecaster:
    """
    Train CausalLSTMForecaster on real windowed data.

    Uses class-weighted CrossEntropyLoss (same inverse-frequency scheme as
    train_multiclass_nowcast), early stops on val macro-F1, saves best
    checkpoint via save_lstm().

    Note: CausalLSTMForecaster already has Softmax in its head, so we
    use NLLLoss on log(output) instead of CrossEntropyLoss on raw logits.

    Returns:
        Trained CausalLSTMForecaster.
    """
    from sklearn.metrics import f1_score

    model = CausalLSTMForecaster(n_features=n_features, n_classes=n_classes)
    model.to(device)

    # Class-weighted loss — same formula as train.py
    classes, counts = np.unique(y_tr, return_counts=True)
    freq = counts / counts.sum()
    weight_map = {int(c): min(1.0 / max(f, 1e-8), 5000.0) for c, f in zip(classes, freq)}
    class_weights = torch.zeros(n_classes, device=device)
    for c, w in weight_map.items():
        if c < n_classes:
            class_weights[c] = w
    class_weights[class_weights == 0] = 1.0
    print(f"  LSTM class weights: {class_weights.cpu().tolist()}")

    # Model outputs softmax probabilities, so use NLLLoss on log(probs)
    criterion = torch.nn.NLLLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    best_f1 = -1.0
    best_state = None
    epochs_no_improve = 0

    print(f"\n--- CausalLSTM Training ({epochs} max epochs, patience={patience}) ---")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        perm = torch.randperm(len(X_tr))
        for start in range(0, len(X_tr), batch_size):
            idx = perm[start:start + batch_size]
            xb = torch.tensor(X_tr[idx.numpy()], dtype=torch.float32, device=device)
            yb = y_tr_t[idx].to(device)

            probs = model(xb)
            log_probs = torch.log(probs + 1e-8)
            loss = criterion(log_probs, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_preds = []
        with torch.no_grad():
            for start in range(0, len(X_val), batch_size):
                xb = torch.tensor(X_val[start:start + batch_size], dtype=torch.float32, device=device)
                probs = model(xb)
                val_preds.append(probs.argmax(dim=-1).cpu().numpy())
        val_preds = np.concatenate(val_preds)
        val_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0.0)

        print(f"  Epoch {epoch+1:3d} | train_loss={avg_loss:.4f} | val_macro_F1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1} (best val macro-F1={best_f1:.4f})")
            break

    if best_state is None:
        print("  WARNING: No improvement observed. Saving final model state.")
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    save_lstm(model, save_path)
    print(f"  CausalLSTM saved: {save_path} (best val macro-F1={best_f1:.4f})")
    print("--- CausalLSTM Training complete ---\n")

    return model
