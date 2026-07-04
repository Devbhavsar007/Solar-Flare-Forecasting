import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from typing import Dict, List, Any

def detect_covariate_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05) -> Dict[str, Any]:
    """
    Performs Kolmogorov-Smirnov (KS) test per feature to detect covariate drift.
    A p-value < threshold indicates significant drift.
    """
    drifted_features = []
    p_values = {}
    
    # Get common numeric features
    common_cols = set(reference_df.select_dtypes(include=[np.number]).columns).intersection(
        set(current_df.select_dtypes(include=[np.number]).columns)
    )
    
    for col in common_cols:
        ref_data = reference_df[col].dropna()
        curr_data = current_df[col].dropna()
        
        if len(ref_data) > 0 and len(curr_data) > 0:
            stat, p_value = ks_2samp(ref_data, curr_data)
            p_values[col] = p_value
            if p_value < threshold:
                drifted_features.append(col)
                
    return {
        "drifted": len(drifted_features) > 0,
        "drifted_features": drifted_features,
        "p_values": p_values
    }

def detect_prediction_drift(ref_proba: np.ndarray, curr_proba: np.ndarray, threshold: float = 0.10) -> Dict[str, Any]:
    """
    Computes Population Stability Index (PSI) on predicted class distributions
    to detect prediction drift.
    """
    # Create 10 bins for probability distribution
    bins = np.linspace(0, 1, 11)
    
    # Calculate histograms
    ref_counts, _ = np.histogram(ref_proba, bins=bins)
    curr_counts, _ = np.histogram(curr_proba, bins=bins)
    
    # Convert to percentages and add a small epsilon to avoid division by zero
    epsilon = 1e-6
    ref_pct = (ref_counts / len(ref_proba)) + epsilon
    curr_pct = (curr_counts / len(curr_proba)) + epsilon
    
    # Calculate PSI
    psi_values = (curr_pct - ref_pct) * np.log(curr_pct / ref_pct)
    total_psi = np.sum(psi_values)
    
    if total_psi > 0.20:
        severity = "major"
    elif total_psi > 0.10:
        severity = "moderate"
    else:
        severity = "stable"
        
    return {
        "drifted": total_psi > threshold,
        "psi": total_psi,
        "severity": severity
    }
