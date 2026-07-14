"""
JWALA HEL1OS Binary Detector Training (scripts/train_hel1os.py)

Trains an XGBoost binary classifier (Flare vs No Flare) on HEL1OS data.
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import recall_score, f1_score, precision_score, confusion_matrix
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.labels import build_multiclass_labels, create_windows

def load_and_preprocess_hel1os():
    print("Loading HEL1OS data...")
    hel1os_path = "data/raw/hel1os/hel1os_all.parquet"
    if not os.path.exists(hel1os_path):
        raise FileNotFoundError(f"HEL1OS data not found at {hel1os_path}")
    
    hel1os_df = pd.read_parquet(hel1os_path)
    
    # Resample 1s cadence to 1min cadence for modeling
    print("Resampling HEL1OS data to 1-minute cadence...")
    hel1os_df = hel1os_df.resample('1min').mean()
    
    print("Loading NOAA master catalog...")
    noaa_path = "data/raw/noaa_catalog.parquet"
    if not os.path.exists(noaa_path):
        raise FileNotFoundError(f"NOAA catalog not found at {noaa_path}")
    
    master_catalogue = pd.read_parquet(noaa_path)
    
    print("Building labels (coincidence with GOES/SoLEXS flares)...")
    hel1os_df = build_multiclass_labels(hel1os_df, master_catalogue)
    # Convert multiclass label to binary (0 = N/B, 1 = C/M/X)
    hel1os_df['binary_label'] = (hel1os_df['label'] > 0).astype(int)
    
    print("Computing hardness ratio and rolling features...")
    # Add small epsilon to avoid division by zero
    hel1os_df['hardness_ratio'] = hel1os_df['counts_high'] / (hel1os_df['counts_low'] + 1e-8)
    
    hel1os_df['counts_low_diff'] = hel1os_df['counts_low'].diff()
    hel1os_df['counts_high_diff'] = hel1os_df['counts_high'].diff()
    
    hel1os_df['counts_low_roll_max_15m'] = hel1os_df['counts_low'].rolling(15, min_periods=1).max()
    hel1os_df['counts_high_roll_max_15m'] = hel1os_df['counts_high'].rolling(15, min_periods=1).max()
    
    # Drop rows where counts are NaN to mask out data gaps
    hel1os_df.dropna(subset=['counts_low', 'counts_high'], inplace=True)
    
    # Override 'label' so create_windows uses binary_label
    hel1os_df['label'] = hel1os_df['binary_label']
    
    return hel1os_df

def temporal_split_hel1os(X, y_fore, dates):
    """
    Train <= 2023-11-15
    Val 2023-11-15 to 2023-12-15
    Test >= 2023-12-15
    """
    train_mask = dates < pd.Timestamp('2023-11-15', tz='UTC')
    val_mask = (dates >= pd.Timestamp('2023-11-15', tz='UTC')) & (dates < pd.Timestamp('2023-12-15', tz='UTC'))
    test_mask = dates >= pd.Timestamp('2023-12-15', tz='UTC')
    
    return (X[train_mask], y_fore[train_mask],
            X[val_mask], y_fore[val_mask],
            X[test_mask], y_fore[test_mask])

def train_hel1os_xgb(X_tr, y_tr, X_val, y_val):
    print("Training HEL1OS XGBoost Binary Classifier...")
    
    X_tr_flat = X_tr.reshape(len(X_tr), -1)
    X_val_flat = X_val.reshape(len(X_val), -1)
    
    # Class weights for binary classification
    classes, counts = np.unique(y_tr, return_counts=True)
    freq = counts / counts.sum()
    weight_map = {int(c): min(1.0 / max(f, 1e-8), 50.0) for c, f in zip(classes, freq)}
    sample_weights = np.array([weight_map[int(y)] for y in y_tr])
    
    dtrain = xgb.DMatrix(X_tr_flat, label=y_tr, weight=sample_weights)
    dval = xgb.DMatrix(X_val_flat, label=y_val)
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'learning_rate': 0.05,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'hist',
        'seed': 42
    }
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=300,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=20,
        verbose_eval=50
    )
    
    return model

def main():
    os.makedirs("models", exist_ok=True)
    
    try:
        hel1os_df = load_and_preprocess_hel1os()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run scripts/extract_hel1os.py first to generate the parquet file.")
        return
    
    feature_cols = [
        'counts_low', 'counts_high', 'hardness_ratio', 
        'counts_low_diff', 'counts_high_diff',
        'counts_low_roll_max_15m', 'counts_high_roll_max_15m'
    ]
    
    print("Creating sliding windows...")
    # Using the updated create_windows that returns dict if horizon is list, or array if int
    X, y_now, y_fore, dates = create_windows(hel1os_df, feature_cols=feature_cols, window_size=60, horizon=15, step=15)
    
    print(f"Total windows: {len(X)}")
    
    X_tr, y_tr, X_val, y_val, X_test, y_test = temporal_split_hel1os(X, y_fore, dates)
    print(f"Split sizes -> Train: {len(y_tr)} | Val: {len(y_val)} | Test: {len(y_test)}")
    
    if len(X_tr) == 0:
        print("No training data available for HEL1OS. Ensure data spans Nov-Dec 2023.")
        return
        
    model = train_hel1os_xgb(X_tr, y_tr, X_val, y_val)
    
    print("Saving model...")
    model.save_model("models/hel1os_binary_xgb.json")
    
    # Evaluate
    if len(X_test) > 0:
        print("\nEvaluating HEL1OS Binary XGBoost on Test Set (Dec 15+ 2023)...")
        X_test_flat = X_test.reshape(len(X_test), -1)
        dtest = xgb.DMatrix(X_test_flat)
        preds_proba = model.predict(dtest)
        preds = (preds_proba > 0.5).astype(int)
        
        precision = precision_score(y_test, preds, zero_division=0)
        recall = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        
        print("\nTest Set Metrics (Binary Flare Detection):")
        print(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
                
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, preds))
    else:
        print("No test data available for evaluation.")

if __name__ == "__main__":
    main()
