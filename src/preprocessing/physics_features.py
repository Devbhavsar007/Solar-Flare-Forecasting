"""
Physics feature engineering for solar flare forecasting.
"""

import numpy as np
import pandas as pd
from scipy.signal import correlate
import yaml
import os

# [RULE-12] Module-level open() needs FileNotFoundError guard
try:
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "nowcasting.yaml")
    with open(config_path) as f:
        PHASE_CFG = yaml.safe_load(f).get("phase_detector", {})
except FileNotFoundError:
    print(f"WARNING: configs/nowcasting.yaml not found at {config_path}. Using default phase thresholds.")
    PHASE_CFG = {}

# Phase detection thresholds — calibrated during M1 EDA, stored in config.
PHASE_PEAK_FRAC_PRE       = PHASE_CFG.get("peak_frac_pre",       0.30)
PHASE_PEAK_FRAC_GRADUAL   = PHASE_CFG.get("peak_frac_gradual",   0.80)
BACKGROUND_SIGMA          = PHASE_CFG.get("background_sigma",    0.10)


def detect_solar_phase(flux_window: np.ndarray) -> int:
    """
    Classify the current position in the solar flare lifecycle.
    0=quiet  1=pre-flare  2=impulsive  3=peak  4=gradual

    The solar event phase is the single strongest categorical predictor of
    an imminent M/X peak and is fed as a feature into XGBoost.
    """
    if len(flux_window) < 5:
        return 0
    dfdt    = np.gradient(flux_window)
    current = flux_window[-1]
    peak    = flux_window.max()
    bkg_est = np.median(flux_window[:max(1, len(flux_window)//4)])
    slope   = dfdt[-1]

    if current < bkg_est * (1 + BACKGROUND_SIGMA) and abs(slope) < bkg_est * 0.05:
        return 0   # quiet
    elif slope > 0 and current < peak * PHASE_PEAK_FRAC_PRE:
        return 1   # pre-flare: rising slowly, well below peak
    elif slope > 0 and current >= peak * PHASE_PEAK_FRAC_PRE:
        return 2   # impulsive: rising steeply
    elif current >= peak * PHASE_PEAK_FRAC_GRADUAL and abs(slope) < current * 0.01:
        return 3   # near-peak
    elif slope < -current * 0.01:
        return 4   # gradual decay
    return 0


def engineer_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["flux_low", "flux_mid", "flux_high", "counts_low", "counts_high", "counts"]:
        if col in df.columns:
            df[f"d_{col}_dt"] = df[col].diff()

    # SoLEXS counts vs HEL1OS counts_low
    if "counts" in df.columns and "counts_low" in df.columns:
        df["neupert_ratio"] = (
            df["counts_low"].diff() / (df["counts"] + 1e-12)
        )
        df["channel_ratio"] = df["counts"] / (df["counts_low"] + 1e-10)
        
        w = min(60, len(df))
        if w > 0:
            corr = correlate(
                df["counts"].fillna(0).values[-w:],
                df["counts_low"].fillna(0).values[-w:]
            )
            df["instrument_lag"] = int(corr.argmax() - w + 1)

    if "counts" in df.columns:
        ratio = df["counts"] / df["counts"].shift(5).clip(lower=1e-12)
        df["doubling_time"] = 5.0 / np.log2(ratio.clip(lower=1e-6))
        
        # Calculate normalised flux avoiding NaNs
        rolling_q = df["counts"].rolling(120).quantile(0.10).clip(lower=1e-12)
        df["normalised_flux"] = df["counts"] / rolling_q
        
        flux = df["counts"].values
        df["solar_phase"] = [
            detect_solar_phase(flux[max(0, i - 20):i + 1])
            for i in range(len(flux))
        ]

    for col in ["counts", "counts_low", "counts_high"]:
        if col in df.columns:
            df[f"{col}_rollstd_5"]  = df[col].rolling(5).std()
            df[f"{col}_rollstd_15"] = df[col].rolling(15).std()
            df[f"{col}_rollmax_5"]  = df[col].rolling(5).max()

    return df.dropna()


def subtract_background(df: pd.DataFrame, window_min: int = 10) -> pd.DataFrame:
    df_clean = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        baseline = df[col].rolling(window_min, min_periods=1).median()
        df_clean[col] = (df[col] - baseline).clip(lower=0)
    return df_clean
