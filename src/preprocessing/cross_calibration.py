"""
Fits a robust calibration mapping from GOES XRS flux to SoLEXS-equivalent flux.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
import mlflow

def fit_goes_solexs_calibration(goes_df: pd.DataFrame,
                                solexs_df: pd.DataFrame,
                                noaa_catalog: pd.DataFrame = None,
                                overlap_start: str = None,
                                overlap_end: str = None) -> dict:
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

    # 1. Zero-fill Mask for SoLEXS
    # Real data has continuous zeroes for gaps. Flag runs of 0s > 2s as NaN.
    # At 1s cadence, this means counting contiguous zeroes.
    zero_mask = (solexs_df[solexs_col] == 0)
    zero_runs = zero_mask.groupby((~zero_mask).cumsum()).sum()
    # Find which groups are > 2
    bad_groups = zero_runs[zero_runs > 2].index
    # Mask those groups
    is_bad = (~zero_mask).cumsum().isin(bad_groups) & zero_mask
    solexs_df = solexs_df.copy()
    solexs_df.loc[is_bad, solexs_col] = np.nan
    
    # 2. Dynamic Overlap Date Range
    if overlap_start is None:
        overlap_start = max(goes_df.index.min(), solexs_df.index.min())
    if overlap_end is None:
        overlap_end = min(goes_df.index.max(), solexs_df.index.max())

    goes_sub = goes_df.loc[overlap_start:overlap_end].copy()
    solexs_sub = solexs_df.loc[overlap_start:overlap_end].copy()

    # 3. Outlier Clipping against NOAA Catalog
    # Exclude any SoLEXS values > 1e5 if no flare is active in the catalog at that time.
    if noaa_catalog is not None:
        # Resample first to minute cadence for easier temporal matching
        pass # We'll do it after resample to 1min to save time on 61M rows.

    goes_1min   = goes_sub[goes_col].resample("1min").mean()
    solexs_1min = solexs_sub[solexs_col].resample("1min").mean()

    if "xrs_a" in goes_sub.columns:
        goes_1min_a = goes_sub["xrs_a"].resample("1min").mean()
        if "pradan_version" in solexs_sub.columns:
            solexs_1min_version = solexs_sub["pradan_version"].resample("1min").last()
            aligned = pd.concat([goes_1min, solexs_1min, goes_1min_a, solexs_1min_version], axis=1).dropna(subset=[goes_1min.name, solexs_1min.name])
            aligned.columns = ["goes", "solexs", "goes_a", "pradan_version"]
        else:
            aligned = pd.concat([goes_1min, solexs_1min, goes_1min_a], axis=1).dropna(subset=[goes_1min.name, solexs_1min.name])
            aligned.columns = ["goes", "solexs", "goes_a"]
            aligned["pradan_version"] = "v1.0"
    else:
        if "pradan_version" in solexs_sub.columns:
            solexs_1min_version = solexs_sub["pradan_version"].resample("1min").last()
            aligned = pd.concat([goes_1min, solexs_1min, solexs_1min_version], axis=1).dropna(subset=[goes_1min.name, solexs_1min.name])
            aligned.columns = ["goes", "solexs", "pradan_version"]
        else:
            aligned = pd.concat([goes_1min, solexs_1min], axis=1).dropna()
            aligned.columns = ["goes", "solexs"]
            aligned["pradan_version"] = "v1.0"
            
    aligned = aligned[(aligned["goes"] > 1e-9) & (aligned["solexs"] > 1e-9)]

    # Apply outlier verification on the aligned 1-min data
    if noaa_catalog is not None:
        high_counts = aligned["solexs"] > 1e5
        if high_counts.any():
            # Check if any flare covers these timestamps
            valid_outliers = pd.Series(False, index=aligned.index)
            for _, flare in noaa_catalog.iterrows():
                # If start_time / end_time are missing/NaT, skip
                if pd.isna(flare.get('start_time')) or pd.isna(flare.get('end_time')):
                    continue
                mask = (aligned.index >= flare['start_time']) & (aligned.index <= flare['end_time'])
                valid_outliers |= mask
            
            # Invalid outliers are those > 1e5 AND NOT during a flare
            invalid = high_counts & (~valid_outliers)
            if invalid.any():
                print(f"WARNING: Found {invalid.sum()} 1-min bins > 10^5 counts/s without a matching NOAA flare. Clipping to NaN.")
                aligned.loc[invalid, "solexs"] = np.nan
                aligned = aligned.dropna()

    if len(aligned) < 10:
        raise ValueError("Not enough overlapping data points for calibration.")

    calibration = {}
    
    # 4. Global Fit per version
    for version, grp in aligned.groupby("pradan_version"):
        if len(grp) < 10:
            continue
            
        log_goes   = np.log10(grp["goes"].values).reshape(-1, 1)
        log_solexs = np.log10(grp["solexs"].values)

        model = HuberRegressor(epsilon=1.5, max_iter=500)
        model.fit(log_goes, log_solexs)
        r2 = model.score(log_goes, log_solexs)

        calib_v = {
            "slope":     float(model.coef_[0]),
            "intercept": float(model.intercept_),
            "r2":        round(float(r2), 4),
            "n_samples": len(grp),
            "overlap_start": str(overlap_start),
            "overlap_end": str(overlap_end),
        }
        calibration[version] = calib_v

        print(
            f"GOES->SoLEXS calibration ({version}):\n"
            f"  slope={calib_v['slope']:.4f}  "
            f"intercept={calib_v['intercept']:.4f}  "
            f"r2={r2:.4f}  n={len(grp)}"
        )
        
    if "v1.0" not in calibration:
        raise ValueError("Could not fit baseline v1.0 calibration.")
        
    try:
        with mlflow.start_run(run_name="goes_solexs_log_calibration"):
            mlflow.log_params(calibration["v1.0"])
    except Exception as e:
        print(f"Failed to log to MLflow: {e}")

    # 4.5. Secondary Regression: XRS-A vs SoLEXS (for comparison with literature)
    if "goes_a" in aligned.columns:
        aligned_a = aligned[(aligned["goes_a"] > 1e-9) & (aligned["solexs"] > 1e-9)].copy()
        if len(aligned_a) > 10:
            log_goes_a = np.log10(aligned_a["goes_a"].values).reshape(-1, 1)
            log_solexs_a = np.log10(aligned_a["solexs"].values)
            model_a = HuberRegressor(epsilon=1.5, max_iter=500)
            model_a.fit(log_goes_a, log_solexs_a)
            r2_a = model_a.score(log_goes_a, log_solexs_a)
            print(
                f"\n--- Secondary Validation against Sarwade et al. (2025) ---\n"
                f"GOES XRS-A (0.5-4Å) -> SoLEXS Counts calibration:\n"
                f"  slope={model_a.coef_[0]:.4f}  "
                f"intercept={model_a.intercept_:.4f}  "
                f"r2={r2_a:.4f}  n={len(aligned_a)}\n"
                f"Note: Sarwade et al. maps XRS-A to physical flux (W/m^2), not counts/s.\n"
                f"-----------------------------------------------------------"
            )

    # 5. Quarterly Stability Check
    print("\n--- Quarterly Stability Check ---")
    # Group by Quarter
    aligned['quarter'] = aligned.index.to_period("Q")
    for q, grp in aligned.groupby('quarter'):
        if len(grp) < 10:
            continue
        l_goes = np.log10(grp["goes"].values).reshape(-1, 1)
        l_solexs = np.log10(grp["solexs"].values)
        q_model = HuberRegressor(epsilon=1.5, max_iter=500)
        q_model.fit(l_goes, l_solexs)
        q_r2 = q_model.score(l_goes, l_solexs)
        print(f"  {q}: slope={q_model.coef_[0]:.4f}, int={q_model.intercept_:.4f}, r2={q_r2:.4f}, n={len(grp)}")
    print("---------------------------------\n")

    if "v1.0" in calibration and calibration["v1.0"]["r2"] < 0.80:
        print("WARNING: v1.0 r2 < 0.80 — calibration may be unreliable. "
              "Check overlap period for data gaps.")
    return calibration


def apply_goes_calibration(goes_flux: np.ndarray,
                           calibration: dict,
                           version: str = "v1.0") -> np.ndarray:
    """Transform GOES XRS flux values to SoLEXS-equivalent scale."""
    if version not in calibration and "slope" in calibration:
        # Backward compatibility if a single calibration dict is passed
        calib = calibration
    else:
        calib = calibration.get(version, calibration["v1.0"])
        
    log_goes = np.log10(np.clip(goes_flux, 1e-12, None))
    log_solexs_equiv = calib["slope"] * log_goes + calib["intercept"]
    return np.power(10, log_solexs_equiv)
