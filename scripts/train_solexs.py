"""
JWALA SoLEXS Classifier Training (scripts/train_solexs.py)

Trains an XGBoost multi-class classifier on SoLEXS data.
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import recall_score, f1_score, precision_score, confusion_matrix
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.labels import build_multiclass_labels, create_windows

def load_and_preprocess_solexs():
    print("Loading SoLEXS data...")
    solexs_path = "data/raw/solexs/solexs_all.parquet"
    if not os.path.exists(solexs_path):
        raise FileNotFoundError(f"SoLEXS data not found at {solexs_path}")
    
    solexs_df = pd.read_parquet(solexs_path)
    
    # Resample 1s cadence to 1min cadence
    print("Resampling SoLEXS data to 1-minute cadence...")
    # Assuming 'COUNTS' is the main feature. Do not fillna(0) to avoid masking gaps.
    solexs_df = solexs_df.resample('1min').mean()
    
    print("Loading NOAA master catalog...")
    noaa_path = "data/raw/noaa_catalog.parquet"
    if not os.path.exists(noaa_path):
        raise FileNotFoundError(f"NOAA catalog not found at {noaa_path}")
    
    master_catalogue = pd.read_parquet(noaa_path)
    
    print("Building multi-class labels...")
    solexs_df = build_multiclass_labels(solexs_df, master_catalogue)
    
    print("Computing rolling features...")
    solexs_df['COUNTS_diff'] = solexs_df['COUNTS'].diff()
    solexs_df['COUNTS_roll_var_15m'] = solexs_df['COUNTS'].rolling(15, min_periods=1).var()
    solexs_df['COUNTS_roll_max_15m'] = solexs_df['COUNTS'].rolling(15, min_periods=1).max()
    solexs_df['COUNTS_roll_max_60m'] = solexs_df['COUNTS'].rolling(60, min_periods=1).max()
    
    # Drop rows where COUNTS is NaN to mask out data gaps (same approach as GOES)
    solexs_df.dropna(subset=['COUNTS'], inplace=True)
    
    return solexs_df

def temporal_split_solexs(X, y_fore, dates):
    """
    Train <= Sep 2024
    Val Oct-Nov 2024
    Test >= Dec 2024
    """
    train_mask = dates < pd.Timestamp('2024-10-01', tz='UTC')
    val_mask = (dates >= pd.Timestamp('2024-10-01', tz='UTC')) & (dates < pd.Timestamp('2024-12-01', tz='UTC'))
    test_mask = dates >= pd.Timestamp('2024-12-01', tz='UTC')
    
    return (X[train_mask], y_fore[train_mask],
            X[val_mask], y_fore[val_mask],
            X[test_mask], y_fore[test_mask])

def train_solexs_xgb(X_tr, y_tr, X_val, y_val):
    print("Training SoLEXS XGBoost Classifier...")
    
    # Flatten windows (N, 60, F) -> (N, 60*F)
    X_tr_flat = X_tr.reshape(len(X_tr), -1)
    X_val_flat = X_val.reshape(len(X_val), -1)
    
    # Class weights
    classes, counts = np.unique(y_tr, return_counts=True)
    freq = counts / counts.sum()
    weight_map = {int(c): min(1.0 / max(f, 1e-8), 5000.0) for c, f in zip(classes, freq)}
    sample_weights = np.array([weight_map[int(y)] for y in y_tr])
    
    dtrain = xgb.DMatrix(X_tr_flat, label=y_tr, weight=sample_weights)
    dval = xgb.DMatrix(X_val_flat, label=y_val)
    
    params = {
        'objective': 'multi:softprob',
        'num_class': 4,
        'eval_metric': 'mlogloss',
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'hist',
        'seed': 42
    }
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=20,
        verbose_eval=50
    )
    
    return model

def main():
    os.makedirs("models", exist_ok=True)
    
    solexs_df = load_and_preprocess_solexs()
    
    feature_cols = ['COUNTS', 'COUNTS_diff', 'COUNTS_roll_var_15m', 'COUNTS_roll_max_15m', 'COUNTS_roll_max_60m']
    
    print("Creating sliding windows...")
    X, y_now, y_fore, dates = create_windows(solexs_df, feature_cols=feature_cols, window_size=60, horizon=15, step=15)
    
    print(f"Total windows: {len(X)}")
    
    X_tr, y_tr, X_val, y_val, X_test, y_test = temporal_split_solexs(X, y_fore, dates)
    print(f"Split sizes -> Train: {len(y_tr)} | Val: {len(y_val)} | Test: {len(y_test)}")
    
    if len(X_tr) == 0:
        print("No training data available for SoLEXS. Exiting.")
        return
        
    model = train_solexs_xgb(X_tr, y_tr, X_val, y_val)
    
    print("Saving model...")
    model.save_model("models/solexs_xgb.json")
    
    # Evaluate
    if len(X_test) > 0:
        print("\nEvaluating SoLEXS XGBoost on Test Set (Dec 2024+)...")
        X_test_flat = X_test.reshape(len(X_test), -1)
        dtest = xgb.DMatrix(X_test_flat)
        preds_proba = model.predict(dtest)
        preds = np.argmax(preds_proba, axis=1)
        
        class_names = ["N", "C", "M", "X"]
        precisions = precision_score(y_test, preds, average=None, zero_division=0)
        recalls = recall_score(y_test, preds, average=None, zero_division=0)
        f1s = f1_score(y_test, preds, average=None, zero_division=0)
        
        print("\nTest Set Metrics (Per-Class):")
        for idx, cls_name in enumerate(class_names):
            if idx < len(recalls):
                print(f"  {cls_name}-class | Precision: {precisions[idx]:.4f} | Recall: {recalls[idx]:.4f} | F1: {f1s[idx]:.4f}")
                
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, preds))
    else:
        print("No test data available for evaluation.")

if __name__ == "__main__":
    main()
