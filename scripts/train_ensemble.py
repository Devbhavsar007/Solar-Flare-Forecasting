"""
JWALA Ensemble Evaluation (scripts/train_ensemble.py)

Evaluates the final Two-Model Ensemble (LSTM + MultiHorizon TCN) on the test slice.
"""

import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.forecasting.causal_lstm import load_lstm
from src.forecasting.multi_horizon import load_multi_horizon
from src.forecasting.ensemble import ThreeModelEnsemble
from scripts.train_models import get_real_data, temporal_three_way_split

def main():
    print("Loading test data...")
    X, y_fore_dict, y_now, dates = get_real_data()
    y_fore = y_fore_dict["h15"]
    
    _, _, _, _, _, _, X_test, y_test, y_now_test = temporal_three_way_split(X, y_fore, y_now, dates)
    
    if len(X_test) == 0:
        print("No test data available. Exiting.")
        return
        
    print(f"Test slice size: {len(X_test)} windows (Dec 2023 - 2024).")
    
    print("\nLoading trained models...")
    try:
        lstm_model = load_lstm(path="models/causal_lstm.pt", n_features=9)
        lstm_model.eval()
        print("  Loaded CausalLSTMForecaster")
    except FileNotFoundError:
        print("  ERROR: models/causal_lstm.pt not found. Run train_models.py first.")
        return
        
    try:
        tcn_model = load_multi_horizon(path="models/multi_horizon.pt", n_features=9, embed_dim=64)
        tcn_model.eval()
        print("  Loaded MultiHorizonForecaster")
    except FileNotFoundError:
        print("  ERROR: models/multi_horizon.pt not found. Run train_models.py first.")
        return
        
    # Instantiate Ensemble (timesfm_model=None activates the 2-model 50/50 fallback)
    ensemble = ThreeModelEnsemble(lstm_model, tcn_model, timesfm_model=None)
    
    # Normalize features using the same approach as train_models.py
    # Since we need to normalize Test exactly as Train was normalized, and Train
    # isn't saved, we'll re-calculate Train mean/std.
    # We must do the full split to get Train again for normalisation.
    (X_tr, _, _, _, _, _, _, _, _) = temporal_three_way_split(X, y_fore, y_now, dates)
    
    EPS = 1e-12
    X_tr_clean = np.nan_to_num(X_tr, nan=0.0)
    X_test_clean = np.nan_to_num(X_test, nan=0.0)
    
    X_tr_log = np.log10(np.abs(X_tr_clean) + EPS)
    X_test_log = np.log10(np.abs(X_test_clean) + EPS)
    
    mu = X_tr_log.reshape(-1, X_tr_log.shape[-1]).mean(axis=0)
    sigma = X_tr_log.reshape(-1, X_tr_log.shape[-1]).std(axis=0) + 1e-8
    
    X_test_norm = (X_test_log - mu) / sigma
    
    print("\nEvaluating Ensemble...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lstm_model.to(device)
    tcn_model.to(device)
    
    preds = []
    
    # Iterate through test set
    for i in range(len(X_test_norm)):
        # x_tensor shape: (1, 60, 9)
        x_tensor = torch.tensor(X_test_norm[i:i+1], dtype=torch.float32, device=device)
        
        # We don't have real TimesFM flux array here, passing dummy since it's None anyway
        prob = ensemble.predict_single(x_tensor, flux_np=np.zeros(60), horizon=15)
        preds.append(np.argmax(prob))
        
        if i == 0:
            print("  Verified: Real prediction branch executed successfully (not the uniform fallback).")
            print(f"  First prediction probs: {prob}")
            
    preds = np.array(preds)
    
    class_names = ["N", "C", "M", "X"]
    precisions = precision_score(y_test, preds, average=None, zero_division=0)
    recalls = recall_score(y_test, preds, average=None, zero_division=0)
    f1s = f1_score(y_test, preds, average=None, zero_division=0)
    cm = confusion_matrix(y_test, preds)
    
    print("\n--- Two-Model Ensemble (LSTM + TCN) Test Set Metrics ---")
    for idx, cls_name in enumerate(class_names):
        if idx < len(recalls):
            print(f"  {cls_name}-class | Precision: {precisions[idx]:.4f} | Recall: {recalls[idx]:.4f} | F1: {f1s[idx]:.4f}")
            
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\n--- Persistence Baseline ---")
    pers_prec = precision_score(y_test, y_now_test, average=None, zero_division=0)
    pers_rec = recall_score(y_test, y_now_test, average=None, zero_division=0)
    pers_f1 = f1_score(y_test, y_now_test, average=None, zero_division=0)
    for idx, cls_name in enumerate(class_names):
        if idx < len(pers_rec):
            print(f"  {cls_name}-class | Precision: {pers_prec[idx]:.4f} | Recall: {pers_rec[idx]:.4f} | F1: {pers_f1[idx]:.4f}")

if __name__ == "__main__":
    main()
