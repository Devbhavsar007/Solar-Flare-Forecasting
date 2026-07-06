"""
JWALA ML Training Pipeline (scripts/train_models.py)

Ingests:
  1. Reprocessed science-quality GOES 13-15 XRS data (SWPC scaling factors stripped)
     to align with GOES-16/17 physical units.
  2. NOAA SWPC flare event catalog.

Applies:
  - 3-Way Temporal Splitting (Train: <=2021, Val: 2022, Test: 2023-2024)
  - Class-weighted loss (handled in train_multiclass_nowcast)
  - Strict ban on oversampling (no SMOTE) to prevent temporal cross-boundary leakage.
"""

import os
import hashlib
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, f1_score
import xgboost as xgb

from src.nowcasting.tcn_encoder import TCNEncoder
from src.nowcasting.train import (
    train_multiclass_nowcast, 
    optimize_per_class_thresholds, 
    extract_tcn_features
)

def compute_sha256(filepath: str) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def mock_load_goes_data():
    """
    Mock function to simulate loading reprocessed GOES 13-15 (science-quality)
    and GOES 16-17 data. In production, this would use pandas to read parquets.
    """
    print("Loading reprocessed science-quality GOES 13-15 (stripped of SWPC 0.85/0.7 scaling factors)...")
    print("Loading GOES 16-17 data...")
    # Generating dummy shape (N, T, F) -> 1000 samples, 60-min window, 4 features
    N = 1000
    X = np.random.randn(N, 60, 4)
    # Target classes: 0=N, 1=C, 2=M, 3=X (Heavily skewed towards N)
    y = np.random.choice([0, 1, 2, 3], size=N, p=[0.90, 0.08, 0.015, 0.005])
    
    # Simulating time indices for the 3-way split
    dates = pd.date_range(start="2020-01-01", periods=N, freq="3D")
    
    return X, y, dates

def build_supervised_pairs(X, y, dates):
    """
    Simulates pairing the 60-minute X-ray flux windows with the next-class labels.
    """
    print("Constructing 60-min window -> next-class supervised pairs...")
    return X, y, dates

def temporal_three_way_split(X, y, dates):
    """
    Strict contiguous time block splitting.
    Train: <= 2021
    Validate (for threshold tuning): 2022
    Test (for final reporting): >= 2023
    """
    train_mask = dates.year <= 2021
    val_mask = dates.year == 2022
    test_mask = dates.year >= 2023
    
    X_tr, y_tr = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"Split sizes -> Train: {len(y_tr)} | Val: {len(y_val)} | Test: {len(y_test)}")
    
    # Check 2022 X-class counts for validation
    val_counts = pd.Series(y_val).value_counts()
    x_class_count = val_counts.get(3, 0)
    print(f"2022 Validation Slice X-class event count: {x_class_count}")
    if x_class_count < 5:
        print("WARNING: X-class events are sparse in 2022 validation slice. The X threshold will default to crude overrides.")
        
    return X_tr, y_tr, X_val, y_val, X_test, y_test

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("configs", exist_ok=True)
    
    # 1. Load Data
    X_raw, y_raw, dates = mock_load_goes_data()
    X, y, dates = build_supervised_pairs(X_raw, y_raw, dates)
    
    # 2. Strict 3-Way Temporal Split
    X_tr, y_tr, X_val, y_val, X_test, y_test = temporal_three_way_split(X, y, dates)
    
    # 3. Extract Features
    # Note: Using mock TCN Encoder for pipeline flow
    encoder = TCNEncoder(input_dim=4, num_channels=[16, 32, 64], kernel_size=3, dropout=0.2)
    tcn_tr = extract_tcn_features(encoder, X_tr)
    tcn_val = extract_tcn_features(encoder, X_val)
    tcn_test = extract_tcn_features(encoder, X_test)
    
    # 4. Train Model (Handles class weighting internally; NO oversampling)
    print("\nTraining Multi-class Nowcaster...")
    model = train_multiclass_nowcast(
        X_tr, y_tr, 
        X_val, y_val, 
        tcn_tr, tcn_val, 
        models_dir="models"
    )
    
    # 5. Tune Thresholds on Validation Slice
    print("\nTuning thresholds against 2022 Validation slice...")
    flat_val = X_val.reshape(len(X_val), -1)
    combined_val = np.concatenate([tcn_val, flat_val], axis=1)
    thresholds = optimize_per_class_thresholds(model, combined_val, y_val)
    
    # 6. Evaluate on Untouched Test Slice (2023-2024)
    print("\nEvaluating on untouched 2023-2024 Test slice (Solar Maximum ramp-up)...")
    flat_test = X_test.reshape(len(X_test), -1)
    combined_test = np.concatenate([tcn_test, flat_test], axis=1)
    
    # Instead of predict(), we use predict_proba and the tuned thresholds to assign classes
    proba_test = model.predict_proba(combined_test)
    y_pred = np.zeros_like(y_test)
    
    # Simple hierarchy application: check from highest class downwards
    for i in range(len(y_test)):
        if proba_test[i, 3] >= thresholds.get("X", 0.5):
            y_pred[i] = 3
        elif proba_test[i, 2] >= thresholds.get("M", 0.5):
            y_pred[i] = 2
        elif proba_test[i, 1] >= thresholds.get("C", 0.5):
            y_pred[i] = 1
        else:
            y_pred[i] = 0

    class_names = ["N", "C", "M", "X"]
    recalls = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1s = f1_score(y_test, y_pred, average=None, zero_division=0)
    
    print("\nTest Set Metrics (Per-Class):")
    for idx, cls_name in enumerate(class_names):
        if idx < len(recalls):
            print(f"  {cls_name}-class | Recall: {recalls[idx]:.4f} | F1: {f1s[idx]:.4f}")
    
    # 7. Hash Generation
    print("\nGenerating model hashes...")
    try:
        with open("configs/model_hashes.yaml", "r") as f:
            hashes = yaml.safe_load(f) or {}
    except FileNotFoundError:
        hashes = {}

    pkl_path = "models/xgb_multiclass.pkl"
    if os.path.exists(pkl_path):
        hashes["xgb_multiclass.pkl"] = compute_sha256(pkl_path)
        
    with open("configs/model_hashes.yaml", "w") as f:
        yaml.dump(hashes, f)
        
    print("Updated configs/model_hashes.yaml.")

if __name__ == "__main__":
    main()
