"""
Fits a robust calibration mapping from GOES XRS flux to SoLEXS-equivalent flux.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
import mlflow

def fit_goes_solexs_calibration(goes_df: pd.DataFrame,
                                solexs_df: pd.DataFrame,
                                overlap_start: str = "2023-09-01",
                                overlap_end: str = "2024-06-01") -> dict:
    """
    Fit a calibration mapping from GOES XRS flux to SoLEXS-equivalent flux.

    WHY LOG-SPACE: Solar flux spans ~6 decades. Linear regression in linear
    space is dominated by the top 3-5 X-class events and gives a meaningless
    calibration over the C-class range (where most training data lives).

    WHY HUBER: Robust to the outliers produced by flare peaks that saturate
    one instrument but not the other, or occur during data gaps.

    Saves calibration coefficients to MLflow for reproducibility.
    """
    goes_col   = "xrs_b" if "xrs_b" in goes_df.columns else goes_df.columns[0]
    solexs_col = "flux_high" if "flux_high" in solexs_df.columns else (
                 "counts" if "counts" in solexs_df.columns else solexs_df.columns[0])

    # Subset to the overlap period
    try:
        goes_sub = goes_df[overlap_start:overlap_end]
        solexs_sub = solexs_df[overlap_start:overlap_end]
    except KeyError:
        goes_sub = goes_df
        solexs_sub = solexs_df

    goes_1min   = goes_sub[goes_col].resample("1min").mean()
    solexs_1min = solexs_sub[solexs_col].resample("1min").mean()

    aligned = pd.concat([goes_1min, solexs_1min], axis=1).dropna()
    aligned.columns = ["goes", "solexs"]
    aligned = aligned[(aligned["goes"] > 1e-9) & (aligned["solexs"] > 1e-9)]

    if len(aligned) < 10:
        raise ValueError("Not enough overlapping data points for calibration.")

    log_goes   = np.log10(aligned["goes"].values).reshape(-1, 1)
    log_solexs = np.log10(aligned["solexs"].values)

    model = HuberRegressor(epsilon=1.5, max_iter=500)
    model.fit(log_goes, log_solexs)
    r2 = model.score(log_goes, log_solexs)

    calibration = {
        "slope":     float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "r2":        round(float(r2), 4),
        "n_samples": len(aligned),
    }

    try:
        with mlflow.start_run(run_name="goes_solexs_log_calibration"):
            mlflow.log_params(calibration)
    except Exception as e:
        print(f"Failed to log to MLflow: {e}")

    print(
        f"GOES->SoLEXS calibration (log-log):\n"
        f"  slope={calibration['slope']:.4f}  "
        f"intercept={calibration['intercept']:.4f}  "
        f"r2={r2:.4f}  n={len(aligned)}"
    )
    if r2 < 0.80:
        print("WARNING: r2 < 0.80 — calibration may be unreliable. "
              "Check overlap period for data gaps.")
    return calibration


def apply_goes_calibration(goes_flux: np.ndarray,
                           calibration: dict) -> np.ndarray:
    """Transform GOES XRS flux values to SoLEXS-equivalent scale."""
    log_goes = np.log10(np.clip(goes_flux, 1e-12, None))
    log_solexs_equiv = calibration["slope"] * log_goes + calibration["intercept"]
    return np.power(10, log_solexs_equiv)
