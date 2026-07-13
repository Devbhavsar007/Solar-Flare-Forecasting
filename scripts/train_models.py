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
import sys

# Ensure the root project directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
        master_catalogue = download_noaa_catalog(start_year=2010, end_year=2026, out_path=noaa_path)
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
        goes_df = goes_df[~goes_df.index.duplicated(keep='first')]
        if goes_df.index.tzinfo is None:
            goes_df.index = goes_df.index.tz_localize("UTC")

    # Preprocessing: Calibration Fix
    print("Applying cross-calibration for GOES 13-15 science-quality data...")
    solexs_parquet = "data/raw/solexs/solexs_all.parquet"
    if not os.path.exists(solexs_parquet):
        print("WARNING: SoLEXS parquet file not found at data/raw/solexs/solexs_all.parquet.")
        print("Skipping cross-calibration loudly. The model will train on raw GOES flux.")
        goes_df["xrs_b_calibrated"] = goes_df["xrs_b"]
        goes_df["xrs_a_calibrated"] = goes_df["xrs_a"]
        calib = {"slope": 1.0, "intercept": 0.0, "calibrated": False, "n_samples": 0}
    else:
        print(f"Loading SoLEXS data from {solexs_parquet}...")
        solexs_df = pd.read_parquet(solexs_parquet)
        if solexs_df.index.tzinfo is None:
            solexs_df.index = solexs_df.index.tz_localize("UTC")
        try:
            calib = fit_goes_solexs_calibration(
                goes_df, solexs_df, 
                noaa_catalog=master_catalogue,
                overlap_start=None, overlap_end=None
            )

            # --- Task 2: Assign pradan_version to GOES rows ---
            # SoLEXS has a pradan_version column covering 2024-02 onward.
            # Derive GOES version from SoLEXS version by resampling to 1-min
            # and forward-filling, then joining on the time index.
            # GOES rows outside SoLEXS coverage default to "v1.0".
            if "pradan_version" in solexs_df.columns:
                version_1min = solexs_df["pradan_version"].resample("1min").last().dropna()
                goes_df["pradan_version"] = version_1min.reindex(goes_df.index, method="nearest", tolerance=pd.Timedelta("2min"))
                goes_df["pradan_version"] = goes_df["pradan_version"].fillna("v1.0")
            else:
                goes_df["pradan_version"] = "v1.0"

            # Apply calibration per version group
            goes_df["xrs_b_calibrated"] = goes_df.groupby("pradan_version")["xrs_b"].transform(
                lambda s: apply_goes_calibration(s.values, calib, version=s.name)
            )
            goes_df["xrs_a_calibrated"] = goes_df["xrs_a"]

            # Verification: print per-version calibrated stats
            print("\n--- Per-version calibration verification ---")
            for v, grp in goes_df.groupby("pradan_version"):
                n = len(grp)
                if n > 0:
                    raw_med = grp["xrs_b"].median()
                    cal_med = grp["xrs_b_calibrated"].median()
                    print(f"  {v}: n={n}, raw_median={raw_med:.4e}, calibrated_median={cal_med:.4e}")
            print("---------------------------------------------\n")

        except Exception as e:
            print(f"Calibration fitting failed ({e}). Using raw flux.")
            goes_df["xrs_b_calibrated"] = goes_df["xrs_b"]
            goes_df["xrs_a_calibrated"] = goes_df["xrs_a"]
            goes_df["pradan_version"] = "v1.0"
            calib = {"slope": 1.0, "intercept": 0.0, "calibrated": False, "n_samples": 0}

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
    STEP = 15  # One window every 15 minutes to keep memory manageable on laptops
    X, y_now, y_fore, dates = create_windows(goes_df, feature_cols=feature_cols, window_size=60, horizon=15, step=STEP)

    assert len(X) == len(y_now) == len(y_fore) == len(dates), \
        f"MISALIGNED: X={len(X)} y_now={len(y_now)} y_fore={len(y_fore)} dates={len(dates)}"
    print(f"Task 1 verified: {len(X)} windows, all arrays aligned")

    return X, y_fore, y_now, dates

def temporal_three_way_split(X, y_fore, y_now, dates):
    """
    Strict contiguous time block splitting.
    Train: <= 2021
    Validate (for threshold tuning): 2022
    Test (for final reporting): >= 2023
    """
    train_mask = dates.year <= 2021
    val_mask = dates.year == 2022
    test_mask = dates.year >= 2023
    
    X_tr, y_fore_tr, y_now_tr = X[train_mask], y_fore[train_mask], y_now[train_mask]
    X_val, y_fore_val, y_now_val = X[val_mask], y_fore[val_mask], y_now[val_mask]
    X_test, y_fore_test, y_now_test = X[test_mask], y_fore[test_mask], y_now[test_mask]
    
    print(f"Split sizes -> Train: {len(y_fore_tr)} | Val: {len(y_fore_val)} | Test: {len(y_fore_test)}")
    
    if len(y_fore_val) > 0:
        val_counts = pd.Series(y_fore_val).value_counts()
        x_class_count = val_counts.get(3, 0)
        print(f"2022 Validation Slice (y_fore) X-class event count: {x_class_count}")
        if x_class_count < 5:
            print("WARNING: X-class events are sparse in 2022 validation slice. The X threshold will default to crude overrides.")
        
    return (X_tr, y_fore_tr, y_now_tr, 
            X_val, y_fore_val, y_now_val, 
            X_test, y_fore_test, y_now_test)

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("configs", exist_ok=True)
    
    # 1. Load Data
    X, y_fore, y_now, dates = get_real_data()
    
    # 2. Strict 3-Way Temporal Split
    (X_tr, y_fore_tr, y_now_tr, 
     X_val, y_fore_val, y_now_val, 
     X_test, y_fore_test, y_now_test) = temporal_three_way_split(X, y_fore, y_now, dates)
    
    if len(X_tr) == 0 or len(X_val) == 0 or len(X_test) == 0:
        print("Not enough data to train. Exiting.")
        return

    # 3. Features (Extracting embeddings using the untrained TCN)
    from src.nowcasting.tcn_encoder import TCNEncoder
    from src.nowcasting.train import extract_tcn_features
    import torch
    
    print("\nExtracting features using randomly initialized TCNEncoder...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = TCNEncoder(n_features=4, embed_dim=64, n_layers=4)
    
    tcn_tr = extract_tcn_features(encoder, X_tr, device=device)
    tcn_val = extract_tcn_features(encoder, X_val, device=device)
    tcn_test = extract_tcn_features(encoder, X_test, device=device)
    
    # Print class distribution before training
    print("\nClass Distribution (y_fore, per-split window counts):")
    counts = pd.DataFrame({
        "Train": pd.Series(y_fore_tr).value_counts(),
        "Val": pd.Series(y_fore_val).value_counts(),
        "Test": pd.Series(y_fore_test).value_counts()
    }).fillna(0).astype(int)
    print(counts)
    
    # 4. Train Model (Handles class weighting internally; NO oversampling)
    print("\nTraining Multi-class Forecaster (15-min horizon)...")
    model = train_multiclass_nowcast(
        X_tr, y_fore_tr, 
        X_val, y_fore_val, 
        tcn_tr, tcn_val, 
        models_dir="models"
    )
    
    # 5. Tune Thresholds on Validation Slice
    print("\nTuning thresholds against 2022 Validation slice...")
    flat_val = X_val.reshape(len(X_val), -1)
    combined_val = np.concatenate([tcn_val, flat_val], axis=1)
    thresholds = optimize_per_class_thresholds(model, combined_val, y_fore_val)
    
    # 6. Evaluate on Untouched Test Slice (2023-2024)
    print("\nEvaluating on untouched 2023-2024 Test slice (Solar Maximum ramp-up)...")
    flat_test = X_test.reshape(len(X_test), -1)
    combined_test = np.concatenate([tcn_test, flat_test], axis=1)
    
    proba_test = model.predict_proba(combined_test)
    n_test = len(y_fore_test)
    
    # Pad if XGBoost returned fewer than 4 classes (same fix as validation path)
    if proba_test.shape[1] < 4:
        print(f"Warning: XGBoost returned {proba_test.shape[1]} classes instead of 4 on test set. Padding.")
        padded = np.zeros((n_test, 4))
        for i, cls_label in enumerate(model.classes_):
            if int(cls_label) < 4:
                padded[:, int(cls_label)] = proba_test[:, i]
        proba_test = padded
    
    assert proba_test.shape == (n_test, 4), f"got {proba_test.shape}"
    
    # Vectorized threshold application (X > M > C priority)
    y_pred = np.zeros(n_test, dtype=int)
    y_pred[proba_test[:, 1] >= thresholds.get("C", 0.5)] = 1
    y_pred[proba_test[:, 2] >= thresholds.get("M", 0.5)] = 2
    y_pred[proba_test[:, 3] >= thresholds.get("X", 0.5)] = 3

    class_names = ["N", "C", "M", "X"]
    from sklearn.metrics import precision_score, confusion_matrix
    precisions = precision_score(y_fore_test, y_pred, average=None, zero_division=0)
    recalls = recall_score(y_fore_test, y_pred, average=None, zero_division=0)
    f1s = f1_score(y_fore_test, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_fore_test, y_pred)
    
    print("\nTest Set Metrics (Per-Class):")
    for idx, cls_name in enumerate(class_names):
        if idx < len(recalls):
            print(f"  {cls_name}-class | Precision: {precisions[idx]:.4f} | Recall: {recalls[idx]:.4f} | F1: {f1s[idx]:.4f}")
            
    print("\nConfusion Matrix (Rows: True, Cols: Pred):")
    print(cm)
    
    # Persistence baseline (15-min horizon means y_pred = y_now)
    y_pers = y_now_test
    pers_prec = precision_score(y_fore_test, y_pers, average=None, zero_division=0)
    pers_rec = recall_score(y_fore_test, y_pers, average=None, zero_division=0)
    pers_f1 = f1_score(y_fore_test, y_pers, average=None, zero_division=0)
    print("\n15-min Persistence Baseline Test Set Metrics (y_pred = y_now):")
    for idx, cls_name in enumerate(class_names):
        if idx < len(pers_rec):
            print(f"  {cls_name}-class | Precision: {pers_prec[idx]:.4f} | Recall: {pers_rec[idx]:.4f} | F1: {pers_f1[idx]:.4f}")
    
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
