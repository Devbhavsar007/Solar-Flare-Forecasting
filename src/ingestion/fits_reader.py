"""
FITS reader for SoLEXS and HEL1OS instruments.

Column names are loaded from configs/fits_columns.yaml.
If a column is missing, the error message shows available names
so the developer can update the config immediately.
"""

from astropy.io import fits
from astropy.time import Time
from scipy.optimize import brentq
import pandas as pd
import numpy as np
import yaml
import os
import pandera as pa
from src.ingestion import schemas as _schemas

config_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "fits_columns.yaml")
with open(config_path) as f:
    COL = yaml.safe_load(f)


def _check_size_gate(filepath: str):
    """[T-5] Security gate: reject FITS files > 500MB."""
    if os.path.getsize(filepath) > 500 * 1024 * 1024:
        raise ValueError(f"FITS file exceeds 500MB limit: {filepath}")


def _validate_schema(df: pd.DataFrame, instrument: str, filepath: str):
    """[RULE-18] Pandera validation at ingestion boundary."""
    schema = _schemas.SOLEXS_SCHEMA if instrument == "solexs" else _schemas.HEL1OS_SCHEMA
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        raise ValueError(
            f"[RULE-18] Schema validation failed for {filepath}:\n"
            f"{exc.failure_cases.to_string()}"
        ) from exc


def apply_dead_time_correction(df: pd.DataFrame) -> pd.DataFrame:
    """Paralyzable dead-time correction for HEL1OS photon-counting detector.
    
    Model: N_obs = N_true * exp(-N_true * tau)
    Inverted numerically per time bin via Brent's method.
    The function r*exp(-r*tau) is monotonically increasing on [0, 1/tau],
    so brentq bracket is [0, 1/tau]. Bins at or above the theoretical
    saturation rate are set to NaN and interpolated.
    """
    dead_time_us = COL["hel1os"].get("dead_time_us", 2.5)
    tau = dead_time_us * 1e-6
    saturation_rate = 1.0 / (np.e * tau)
    # 1/tau is the peak of r*exp(-r*tau); the monotonic branch is [0, 1/tau]
    r_max_bracket = 1.0 / tau

    def correct_single(n_obs: float) -> float:
        rate_obs = n_obs
        if rate_obs <= 0:
            return 0.0
        if rate_obs >= saturation_rate * 0.999:
            # At or above theoretical maximum — no valid inversion exists.
            # Return NaN so the interpolation step handles it, same as
            # genuinely saturated bins. Do NOT silently pass through raw.
            return np.nan
        return brentq(
            lambda r: r * np.exp(-r * tau) - rate_obs,
            0, r_max_bracket,
            xtol=1e-7
        )

    corrected = df.copy()
    for col in ["counts_low", "counts_high"]:
        if col in corrected.columns:
            corrected[col] = corrected[col].apply(correct_single)
            
    # Interpolate saturated bins (NaN) with linear time interpolation, max 5-step gap
    corrected = corrected.interpolate(method="time", limit=5, limit_direction="both")
    return corrected


def _mjd_to_datetime_index(mjd_array: np.ndarray) -> pd.DatetimeIndex:
    """Convert Modified Julian Date array to pandas DatetimeIndex."""
    astropy_times = Time(mjd_array, format="mjd", scale="utc")
    return pd.DatetimeIndex(astropy_times.to_datetime(), name="time")


def read_hel1os(filepath: str) -> pd.DataFrame:
    """Read a HEL1OS CdTe/CZT lightcurve FITS file."""
    _check_size_gate(filepath)
    
    cfg = COL["hel1os"]
    time_col = cfg["time_col"]

    with fits.open(filepath) as hdul:
        # Read low-energy band
        ext_low = cfg["counts_low"]["ext"]
        col_low = cfg["counts_low"]["col"]
        data_low = hdul[ext_low].data
        available_low = [c.name for c in hdul[ext_low].columns]
        if col_low not in available_low:
            raise KeyError(
                f"Column '{col_low}' not found in HEL1OS ext {ext_low}.\n"
                f"Available columns: {available_low}\n"
                f"Update configs/fits_columns.yaml."
            )

        df_low = pd.DataFrame({
            "counts_low": data_low[col_low].astype(float),
        })
        # Convert MJD to datetime and round to nearest second to avoid float precision mismatch
        df_low.index = _mjd_to_datetime_index(data_low[time_col].astype(float)).round("s")

        # Read high-energy band
        ext_high = cfg["counts_high"]["ext"]
        col_high = cfg["counts_high"]["col"]
        data_high = hdul[ext_high].data
        available_high = [c.name for c in hdul[ext_high].columns]
        if col_high not in available_high:
            raise KeyError(
                f"Column '{col_high}' not found in HEL1OS ext {ext_high}.\n"
                f"Available columns: {available_high}\n"
                f"Update configs/fits_columns.yaml."
            )

        df_high = pd.DataFrame({
            "counts_high": data_high[col_high].astype(float),
        })
        df_high.index = _mjd_to_datetime_index(data_high[time_col].astype(float)).round("s")

    # Merge on rounded DatetimeIndex to align samples reliably
    df = pd.merge(df_low, df_high, left_index=True, right_index=True, how="outer")

    df = df.replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    
    # [RULE-18] schema validate
    _validate_schema(df, "hel1os", filepath)
    
    # Apply paralyzable dead-time correction
    return apply_dead_time_correction(df)


def read_solexs(filepath: str, pradan_version: str | None = None) -> pd.DataFrame:
    """Read a SoLEXS SDD lightcurve file (.lc.gz)."""
    _check_size_gate(filepath)
    
    cfg = COL["solexs"]
    ext = cfg.get("binary_table_ext", 1)

    with fits.open(filepath) as hdul:
        data = hdul[ext].data
        header = hdul[ext].header
        available = [c.name for c in hdul[ext].columns]

        time_col = cfg["time_col"]
        counts_col = cfg["counts_col"]

        for col_name in [time_col, counts_col]:
            if col_name not in available:
                raise KeyError(
                    f"Column '{col_name}' not found in SoLEXS FITS ({filepath}).\n"
                    f"Available columns: {available}\n"
                    f"Update configs/fits_columns.yaml."
                )

        time_raw = data[time_col].astype(float)
        counts = data[counts_col].astype(float)

        # Determine the time reference epoch from FITS header.
        # MJDREFI is mandatory for Aditya-L1 files; a missing/zero value
        # means the header is malformed and the silent Unix-epoch fallback
        # would produce 1970-era timestamps with no error raised.
        mjdrefi = header.get("MJDREFI", 0)
        mjdreff = header.get("MJDREFF", 0.0)
        timezero = header.get("TIMEZERO", 0.0)

        if mjdrefi <= 0:
            raise KeyError(
                f"MJDREFI missing or zero in SoLEXS FITS header ({filepath}).\n"
                f"Without a valid epoch reference, timestamps cannot be computed.\n"
                f"Available header keys: {list(header.keys())}\n"
                f"Update configs/fits_columns.yaml or inspect the file."
            )

        ref_mjd = mjdrefi + mjdreff
        ref_time = Time(ref_mjd, format="mjd", scale="utc")
        ref_timestamp = ref_time.to_datetime()
        timestamps = pd.to_datetime(
            time_raw + timezero, unit="s", origin=pd.Timestamp(ref_timestamp)
        )

    df = pd.DataFrame({"counts": counts}, index=timestamps)
    df.index.name = "time"
    
    df["pradan_version"] = pradan_version or "v1.0"

    # Sanity check: Aditya-L1 reached L1 / SoLEXS commissioning ~Sep 2023.
    # Anything before that date is a timestamp-parsing bug, not real data.
    assert timestamps.min() > pd.Timestamp("2023-09-01"), (
        f"SoLEXS timestamp out of expected mission range: "
        f"min={timestamps.min()} in {filepath}. "
        f"This likely means MJDREFI/TIMEZERO were parsed incorrectly."
    )
    
    df = df.replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    
    # [RULE-18] schema validate
    _validate_schema(df, "solexs", filepath)
    
    return df


def merge_instruments(solexs_df: pd.DataFrame,
                      hel1os_df: pd.DataFrame,
                      cadence:    str = "1min") -> pd.DataFrame:
    """Merge SoLEXS and HEL1OS DataFrames, resampling to a shared cadence.
    
    Aggregation: .mean() is correct for both instruments because:
    - SoLEXS 'COUNTS' column is actually a rate (counts/sec) at native 1s
      cadence (verified: TIME[1]-TIME[0] == 1.0s on real files). mean() of
      a rate over a window gives the average rate, which is the correct
      aggregation for resampling rates to a coarser cadence.
    - HEL1OS 'CTR' (count rate) is similarly a rate, so mean() applies.
    """
    return pd.concat([
        solexs_df.resample(cadence).mean(),
        hel1os_df.resample(cadence).mean()
    ], axis=1).dropna()
