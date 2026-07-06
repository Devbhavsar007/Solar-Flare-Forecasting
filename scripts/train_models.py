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
import glob
import hashlib
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, f1_score
import xgboost as xgb

from src.nowcasting.train import (
    train_multiclass_nowcast, 
    optimize_per_class_thresholds
)
from src.preprocessing.cross_calibration import fit_goes_solexs_calibration, apply_goes_calibration
from src.preprocessing.labels import build_multiclass_labels, create_windows
from scripts.download_noaa_catalog import download_noaa_catalog

def compute_sha256(filepath: str) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def get_real_data():
    """
    Load real GOES and SWPC catalog data. If it doesn't exist locally,
    fetch it using the ingestion scripts.
    """
    noaa_path = "data/raw/noaa_catalog.parquet"
    if not os.path.exists(noaa_path):
        print("NOAA catalog not found. Downloading...")
        master_catalogue = download_noaa_catalog(start_year=2023, end_year=2024, out_path=noaa_path)
    else:
        master_catalogue = pd.read_parquet(noaa_path)
        
    # Ensure master_catalogue has 'start_time' and 'end_time' and 'flare_class'
    # download_noaa_catalog produces: ['date', 'start', 'peak', 'end', 'goes', 'xray_class', 'noaa_region', 'peak_time']
    # We must construct start_time and end_time.
    if 'start_time' not in master_catalogue.columns:
        master_catalogue['start_time'] = pd.to_datetime(
            master_catalogue['date'].astype(str) + " " + master_catalogue['start'].astype(str),
            format="%Y%m%d %H%M", utc=True, errors="coerce"
        )
    if 'end_time' not in master_catalogue.columns:
        master_catalogue['end_time'] = pd.to_datetime(
            master_catalogue['date'].astype(str) + " " + master_catalogue['end'].astype(str),
            format="%Y%m%d %H%M", utc=True, errors="coerce"
        )
    if 'flare_class' not in master_catalogue.columns and 'xray_class' in master_catalogue.columns:
        master_catalogue['flare_class'] = master_catalogue['xray_class']
    
    # Check for GOES data
    goes_files = glob.glob("data/raw/goes/*.parquet")
    if not goes_files:
        raise FileNotFoundError(
            "No GOES parquet files found in data/raw/goes/. "
            "You must run `python src/ingestion/goes_downloader.py` "
            "to download real telemetry before training the model."
        )
    else:
        print(f"Found GOES files: {goes_files}. Loading...")
        goes_df = pd.concat([pd.read_parquet(f) for f in goes_files]).sort_index()
        if goes_df.index.tzinfo is None:
            goes_df.index = goes_df.index.tz_localize("UTC")

    # Preprocessing: Calibration Fix
    print("Applying cross-calibration for GOES 13-15 science-quality data...")
    solexs_files = glob.glob("data/raw/pradan_download/*solexs*.fits")
    if not solexs_files:
        print("WARNING: No SoLEXS FITS files found in data/raw/pradan_download/.")
        print("Skipping cross-calibration loudly. The model will train on raw GOES flux.")
        goes_df["xrs_b_calibrated"] = goes_df["xrs_b"]
        goes_df["xrs_a_calibrated"] = goes_df["xrs_a"]
        calib = {"slope": 1.0, "intercept": 0.0, "r2": 1.0, "n_samples": 0}
    else:
        from src.ingestion.fits_reader import read_solexs
        solexs_df = pd.concat([read_solexs(f) for f in solexs_files]).sort_index()
        try:
            calib = fit_goes_solexs_calibration(goes_df, solexs_df)
            goes_df["xrs_b_calibrated"] = apply_goes_calibration(goes_df["xrs_b"].values, calib)
            goes_df["xrs_a_calibrated"] = goes_df["xrs_a"]
        except Exception as e:
            print(f"Calibration fitting failed ({e}). Using raw flux.")
            goes_df["xrs_b_calibrated"] = goes_df["xrs_b"]
            goes_df["xrs_a_calibrated"] = goes_df["xrs_a"]
            calib = {"slope": 1.0, "intercept": 0.0, "r2": 1.0, "n_samples": 0}

    # Save calibration to configs
    try:
        with open("configs/nowcasting.yaml", "r") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        cfg = {}
    cfg["goes_calibration"] = calib
    with open("configs/nowcasting.yaml", "w") as f:
        yaml.dump(cfg, f)

    print("Building multi-class labels...")
    goes_df = build_multiclass_labels(goes_df, master_catalogue)
    
    feature_cols = ["xrs_a_calibrated", "xrs_b_calibrated", "xrs_a", "xrs_b"]
    X, y_now, y_fore = create_windows(goes_df, feature_cols=feature_cols, window_size=60, horizon=15)
    
    # create_windows returns indices that map to the original df
    # We need dates for the temporal split. 
    # create_windows stops at len(data) - window_size - horizon + 1.
    dates = goes_df.index[60-1 : len(goes_df) - 15]

    return X, y_now, dates

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
    
    if len(y_val) > 0:
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
    X, y, dates = get_real_data()
    
    # 2. Strict 3-Way Temporal Split
    X_tr, y_tr, X_val, y_val, X_test, y_test = temporal_three_way_split(X, y, dates)
    
    if len(X_tr) == 0 or len(X_val) == 0 or len(X_test) == 0:
        print("Not enough data to train. Exiting.")
        return

    # 3. Features (TCN is untrained and thus dropped to prevent feeding noise)
    tcn_tr = np.empty((len(X_tr), 0))
    tcn_val = np.empty((len(X_val), 0))
    tcn_test = np.empty((len(X_test), 0))
    
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
    
    proba_test = model.predict_proba(combined_test)
    y_pred = np.zeros_like(y_test)
    
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
