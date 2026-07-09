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

    # Inverse class frequency for sample weights
    classes, counts = np.unique(y_tr, return_counts=True)
    freq = counts / counts.sum()
    class_weights = {c: 1.0 / max(f, 1e-8) for c, f in zip(classes, freq)}
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
    Sweep per-class thresholds to maximize TSS [RULE-3].

    X-class threshold is forced ≤ M-class threshold because missing
    X-class flares is catastrophically costly [Decision D8].

    Returns:
        dict mapping class name → optimal threshold.
    """
    proba = model.predict_proba(combined_val)
    
    # Safety: if the model only learned a subset of classes, proba may have fewer
    # columns than 4, or be completely degenerate. Pad to (N, 4).
    n_samples = combined_val.shape[0]
    
    if proba.ndim == 1:
        # Model returned a single row or flat array
        if proba.shape[0] == 4:
            # Single sample case: (4,) -> (1, 4)
            proba = proba.reshape(1, -1)
        else:
            # Completely degenerate — fall back to defaults
            print("WARNING: predict_proba returned degenerate output. Using default thresholds.")
            defaults = {"N": 0.50, "C": 0.38, "M": 0.45, "X": 0.28}
            try:
                with open("configs/nowcasting.yaml") as f:
                    cfg = yaml.safe_load(f) or {}
            except FileNotFoundError:
                cfg = {}
            cfg["class_thresholds"] = defaults
            with open("configs/nowcasting.yaml", "w") as f:
                yaml.dump(cfg, f, default_flow_style=False)
            print(f"Per-class thresholds (defaults): {defaults}")
            return defaults
    
    if proba.shape[1] < 4:
        padded = np.zeros((proba.shape[0], 4))
        for i, cls in enumerate(model.classes_):
            padded[:, int(cls)] = proba[:, i]
        proba = padded
    
    # Verify proba and y_val have compatible lengths
    if proba.shape[0] != n_samples:
        print(f"WARNING: proba shape {proba.shape} doesn't match input size {n_samples}. Using default thresholds.")
        defaults = {"N": 0.50, "C": 0.38, "M": 0.45, "X": 0.28}
        return defaults
    
    class_names = ["N", "C", "M", "X"]
    thresholds: dict[str, float] = {}

    y_val_arr = np.asarray(y_val).ravel()
    
    for cls_idx, cls_name in enumerate(class_names):
        best_tss = -1.0
        best_t = 0.50

        binary_true = (y_val_arr == cls_idx).astype(int)
        for t in np.arange(0.10, 0.91, 0.05):
            binary_pred = (proba[:, cls_idx] >= t).astype(int)
            tp = ((binary_pred == 1) & (binary_true == 1)).sum()
            fp = ((binary_pred == 1) & (binary_true == 0)).sum()
            fn = ((binary_pred == 0) & (binary_true == 1)).sum()
            tn = ((binary_pred == 0) & (binary_true == 0)).sum()
            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
            tss = tpr - fpr
            if tss > best_tss:
                best_tss = tss
                best_t = float(round(t, 2))

        thresholds[cls_name] = best_t

    # [Decision D8] X-class override: threshold must be ≤ M-class
    if thresholds["X"] > thresholds["M"]:
        thresholds["X"] = min(thresholds["M"] - 0.05, thresholds["X"])
        print(f"X-class threshold overridden to {thresholds['X']:.2f} "
              f"(forced ≤ M={thresholds['M']:.2f})")

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
