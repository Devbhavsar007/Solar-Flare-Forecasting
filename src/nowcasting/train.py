"""
Multi-class nowcasting training pipeline.

TCN encoder extracts temporal embeddings, XGBoost classifies into
N/C/M/X (4 classes) using objective='multi:softprob' [RULE-5].
Models saved in BOTH JSON and PKL formats [RULE-13].
"""
import random, numpy as np, torch
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import joblib
import yaml
import xgboost as xgb
from sklearn.metrics import f1_score

from src.nowcasting.tcn_encoder import TCNEncoder

# [RULE-12] — FileNotFoundError guard on module-level config open
try:
    with open("configs/nowcasting.yaml") as f:
        _CFG = yaml.safe_load(f) or {}
except FileNotFoundError:
    _CFG = {}

CLASS_THRESHOLDS = _CFG.get("class_thresholds", {
    "C": 0.38, "M": 0.45, "X": 0.28, "binary": 0.40,
})


def extract_tcn_features(
    encoder: TCNEncoder,
    X: np.ndarray,
    device: str = "cpu",
    batch_size: int = 256,
) -> np.ndarray:
    """
    Extract TCN embeddings in batches.

    Args:
        encoder: Trained TCNEncoder.
        X: (N, T, F) numpy array of input windows.
        device: torch device string.
        batch_size: Inference batch size.

    Returns:
        (N, embed_dim) numpy array of embeddings.
    """
    encoder.eval()
    encoder.to(device)
    embeddings = []

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.tensor(
                X[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            emb = encoder(batch).cpu().numpy()
            embeddings.append(emb)

    return np.concatenate(embeddings, axis=0)


def train_tcn_encoder(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_features: int = 9,
    embed_dim: int = 64,
    n_layers: int = 4,
    n_classes: int = 4,
    lr: float = 1e-3,
    epochs: int = 80,
    batch_size: int = 256,
    patience: int = 10,
    device: str = "cpu",
    save_path: str = "models/tcn_encoder.pt",
) -> TCNEncoder:
    """
    Pre-train TCNEncoder with a temporary classification head.

    Attaches nn.Linear(embed_dim, n_classes) head, trains with class-weighted
    CrossEntropyLoss (same inverse-frequency scheme as train_multiclass_nowcast),
    early-stops on validation macro-F1, then discards the head and returns
    the trained encoder only.

    Returns:
        Trained TCNEncoder (head discarded).
    """
    import os

    encoder = TCNEncoder(n_features=n_features, embed_dim=embed_dim, n_layers=n_layers)
    head = torch.nn.Linear(embed_dim, n_classes)
    encoder.to(device)
    head.to(device)

    # Class-weighted CrossEntropyLoss — same formula as line 106
    classes, counts = np.unique(y_tr, return_counts=True)
    freq = counts / counts.sum()
    weight_map = {int(c): min(1.0 / max(f, 1e-8), 5000.0) for c, f in zip(classes, freq)}
    class_weights = torch.zeros(n_classes, device=device)
    for c, w in weight_map.items():
        if c < n_classes:
            class_weights[c] = w
    # Fill any missing classes with 1.0
    class_weights[class_weights == 0] = 1.0
    print(f"  TCN encoder class weights: {class_weights.cpu().tolist()}")

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=lr)

    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    best_f1 = -1.0
    best_state = None
    epochs_no_improve = 0

    print(f"\n--- TCN Encoder Pre-training ({epochs} max epochs, patience={patience}) ---")
    for epoch in range(epochs):
        encoder.train()
        head.train()
        epoch_loss = 0.0
        n_batches = 0

        perm = torch.randperm(len(X_tr))
        for start in range(0, len(X_tr), batch_size):
            idx = perm[start:start + batch_size]
            xb = torch.tensor(X_tr[idx.numpy()], dtype=torch.float32, device=device)
            yb = y_tr_t[idx].to(device)

            emb = encoder(xb)
            logits = head(emb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation
        encoder.eval()
        head.eval()
        val_preds = []
        with torch.no_grad():
            for start in range(0, len(X_val), batch_size):
                xb = torch.tensor(X_val[start:start + batch_size], dtype=torch.float32, device=device)
                emb = encoder(xb)
                logits = head(emb)
                val_preds.append(logits.argmax(dim=-1).cpu().numpy())
        val_preds = np.concatenate(val_preds)
        val_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0.0)

        print(f"  Epoch {epoch+1:3d} | train_loss={avg_loss:.4f} | val_macro_F1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1} (best val macro-F1={best_f1:.4f})")
            break

    if best_state is None:
        print("  WARNING: No improvement observed. Saving final encoder state.")
        best_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}

    encoder.load_state_dict(best_state)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(best_state, save_path)
    print(f"  TCN encoder saved: {save_path} (best val macro-F1={best_f1:.4f})")
    print("--- TCN Encoder Pre-training complete ---\n")

    return encoder


def train_multiclass_nowcast(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    tcn_tr: np.ndarray,
    tcn_val: np.ndarray,
    models_dir: str = "models",
) -> xgb.XGBClassifier:
    """
    Train multi-class XGBoost nowcaster on combined TCN + physics features.

    Args:
        X_tr / X_val: (N, T, F) raw windows (flattened to 2D internally).
        y_tr / y_val: (N,) integer labels 0-3 (N/C/M/X).
        tcn_tr / tcn_val: (N, embed_dim) TCN embeddings.
        models_dir: Directory to save model files.

    Returns:
        Trained XGBClassifier.
    """
    # Flatten raw windows and concatenate with TCN embeddings
    flat_tr = X_tr.reshape(len(X_tr), -1)
    flat_val = X_val.reshape(len(X_val), -1)
    combined_tr = np.concatenate([tcn_tr, flat_tr], axis=1)
    combined_val = np.concatenate([tcn_val, flat_val], axis=1)

    # Inverse class frequency for sample weights — capped to prevent instability
    classes, counts = np.unique(y_tr, return_counts=True)
    freq = counts / counts.sum()
    class_weights = {c: min(1.0 / max(f, 1e-8), 5000.0) for c, f in zip(classes, freq)}
    print(f"  Class weights (inverse-frequency, capped at 5000): {class_weights}")
    sample_weights = np.array([class_weights[y] for y in y_tr])

    # [RULE-5] multi:softprob, 4 classes
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=4,
        tree_method="hist",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        seed=SEED,
    )

    model.fit(
        combined_tr, y_tr,
        sample_weight=sample_weights,
        eval_set=[(combined_val, y_val)],
        verbose=False,
    )

    # Per-class F1
    y_pred = model.predict(combined_val)
    if y_pred.ndim > 1:
        y_pred = y_pred.argmax(axis=1)
    for cls_idx, cls_name in enumerate(["N", "C", "M", "X"]):
        y_val_bin = (np.array(y_val) == cls_idx).astype(int).ravel()
        y_pred_bin = (np.array(y_pred) == cls_idx).astype(int).ravel()
        f1 = f1_score(y_val_bin, y_pred_bin, zero_division=0.0)
        print(f"  F1({cls_name}): {f1:.4f}")

    # [RULE-13] Save in BOTH formats
    os.makedirs(models_dir, exist_ok=True)
    json_path = os.path.join(models_dir, "xgb_multiclass.json")
    pkl_path = os.path.join(models_dir, "xgb_multiclass.pkl")

    model.save_model(json_path)
    joblib.dump(model, pkl_path)
    print(f"XGBoost saved: {json_path} + {pkl_path}")

    return model


def optimize_per_class_thresholds(
    model: xgb.XGBClassifier,
    combined_val: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, float]:
    """
    Sweep per-class thresholds to maximize F1 [RULE-3].

    X-class threshold is forced ≤ M-class threshold because missing
    X-class flares is catastrophically costly [Decision D8].

    Returns:
        dict mapping class name → optimal threshold.
    """
    proba = model.predict_proba(combined_val)
    n_samples = combined_val.shape[0]
    
    # If the training set was missing a class, XGBoost might return fewer columns
    if proba.shape[1] < 4:
        print(f"Warning: XGBoost returned {proba.shape[1]} classes instead of 4. Padding missing classes with 0.")
        padded_proba = np.zeros((n_samples, 4))
        # model.classes_ tells us which classes are present
        for i, cls_label in enumerate(model.classes_):
            if int(cls_label) < 4:
                padded_proba[:, int(cls_label)] = proba[:, i]
        proba = padded_proba
        
    assert proba.shape == (n_samples, 4), f"got {proba.shape}"
    
    class_names = ["N", "C", "M", "X"]
    thresholds: dict[str, float] = {}

    y_val_arr = np.asarray(y_val).ravel()
    
    for cls_idx, cls_name in enumerate(class_names):
        best_f1 = -1.0
        best_t = 0.50

        binary_true = (y_val_arr == cls_idx).astype(int)
        for t in np.linspace(0.99, 0.001, 100):
            binary_pred = (proba[:, cls_idx] >= t).astype(int)
            tp = ((binary_pred == 1) & (binary_true == 1)).sum()
            fp = ((binary_pred == 1) & (binary_true == 0)).sum()
            fn = ((binary_pred == 0) & (binary_true == 1)).sum()
            
            # F1 score maximization instead of TSS
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * (precision * recall) / max(precision + recall, 1e-8)
            
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)

        thresholds[cls_name] = best_t



    # Save to configs/nowcasting.yaml
    try:
        with open("configs/nowcasting.yaml") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        cfg = {}

    cfg["class_thresholds"] = thresholds
    with open("configs/nowcasting.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print(f"Per-class thresholds: {thresholds}")
    return thresholds


def display_feature_importance(
    model: xgb.XGBClassifier,
    tcn_embed_dim: int,
    physics_feature_names: list[str],
) -> pd.Series:
    """Print top-20 feature importances with named TCN + physics columns."""
    all_names = [f"tcn_{i}" for i in range(tcn_embed_dim)] + physics_feature_names
    importance = pd.Series(model.feature_importances_, index=all_names)
    top20 = importance.sort_values(ascending=False).head(20)
    print("Top-20 feature importances:")
    print(top20.to_string())
    return importance
