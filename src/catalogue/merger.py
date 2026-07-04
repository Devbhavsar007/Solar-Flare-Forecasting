"""
Master Catalogue Merger — Temporal coincidence merger for SoLEXS and HEL1OS events.

Produces master_catalogue.csv with provenance columns [RULE-17].
_check_noaa is PRIVATE to this module [RULE-15].
"""
import os
import datetime
from pathlib import Path

import pandas as pd
import numpy as np

from src.nowcasting.solexs_detector import FlareEvent


COINCIDENCE_WINDOW = pd.Timedelta(minutes=2)

# Output CSV path
CATALOGUE_PATH = Path("data/processed/master_catalogue.csv")

# Required output columns in exact order
CATALOGUE_COLUMNS = [
    "start_time", "peak_time", "end_time", "flare_class",
    "peak_flux_sxr", "peak_cnt_hxr", "source", "confidence",
    "noaa_confirmed",
    "solexs_fits_path", "hel1os_fits_path",
    "model_version", "pipeline_run_id",
]


def _check_noaa(
    event_time: datetime.datetime,
    noaa_catalog: pd.DataFrame | None,
    window_min: int = 10,
) -> bool:
    """
    PRIVATE — implementation detail of merger.py only [RULE-15].

    Check whether a NOAA catalogue entry exists within ±window_min minutes
    of event_time.  Returns False if noaa_catalog is None or empty.
    """
    if noaa_catalog is None or noaa_catalog.empty:
        return False

    event_ts = pd.Timestamp(event_time)
    delta = pd.Timedelta(minutes=window_min)
    mask = (noaa_catalog["peak_time"] >= event_ts - delta) & (
        noaa_catalog["peak_time"] <= event_ts + delta
    )
    return bool(mask.any())


def check_noaa_confirmed(
    event_time: datetime.datetime,
    noaa_catalog: pd.DataFrame | None,
    window_min: int = 10,
) -> bool:
    """Public wrapper for external use. Calls _check_noaa."""
    return _check_noaa(event_time, noaa_catalog, window_min)


def merge_catalogues(
    solexs_events: list[FlareEvent],
    hel1os_events: list[FlareEvent],
    noaa_catalog: pd.DataFrame | None = None,
    pipeline_run_id: str = "",
    model_version: str = "",
    catalogue_path: Path | None = CATALOGUE_PATH,
) -> pd.DataFrame:
    """
    Merge SoLEXS and HEL1OS event lists via temporal coincidence.

    Confidence tiers:
      - dual:         ±2 min coincidence → confidence * 1.20 (cap 1.0)
      - SoLEXS_only:  no HEL1OS match   → confidence * 0.80
      - HEL1OS_only:  no SoLEXS match   → confidence * 0.60, class stays "?"

    Every row includes provenance columns [RULE-17]:
      solexs_fits_path, hel1os_fits_path, model_version, pipeline_run_id.
    """
    rows: list[dict] = []
    matched_hel1os_indices: set[int] = set()

    # --- Match SoLEXS events against HEL1OS events ---
    for sx_ev in solexs_events:
        best_match: FlareEvent | None = None
        best_idx: int = -1
        best_dt = pd.Timedelta.max

        for j, hx_ev in enumerate(hel1os_events):
            if j in matched_hel1os_indices:
                continue
            dt = abs(pd.Timestamp(sx_ev.peak_time) - pd.Timestamp(hx_ev.peak_time))
            if dt <= COINCIDENCE_WINDOW and dt < best_dt:
                best_dt = dt
                best_match = hx_ev
                best_idx = j

        if best_match is not None:
            # ---- dual detection ----
            matched_hel1os_indices.add(best_idx)
            confidence = min(sx_ev.confidence * 1.20, 1.0)
            rows.append({
                "start_time": sx_ev.start_time,
                "peak_time": sx_ev.peak_time,
                "end_time": sx_ev.end_time,
                "flare_class": sx_ev.flare_class,  # SoLEXS has GOES calibration
                "peak_flux_sxr": sx_ev.peak_flux,
                "peak_cnt_hxr": best_match.peak_flux,
                "source": "dual",
                "confidence": confidence,
                "noaa_confirmed": _check_noaa(sx_ev.peak_time, noaa_catalog),
                "solexs_fits_path": sx_ev.fits_path,
                "hel1os_fits_path": best_match.fits_path,
                "model_version": model_version,
                "pipeline_run_id": pipeline_run_id,
            })
        else:
            # ---- SoLEXS only ----
            confidence = sx_ev.confidence * 0.80
            rows.append({
                "start_time": sx_ev.start_time,
                "peak_time": sx_ev.peak_time,
                "end_time": sx_ev.end_time,
                "flare_class": sx_ev.flare_class,
                "peak_flux_sxr": sx_ev.peak_flux,
                "peak_cnt_hxr": np.nan,
                "source": "SoLEXS_only",
                "confidence": confidence,
                "noaa_confirmed": _check_noaa(sx_ev.peak_time, noaa_catalog),
                "solexs_fits_path": sx_ev.fits_path,
                "hel1os_fits_path": "",
                "model_version": model_version,
                "pipeline_run_id": pipeline_run_id,
            })

    # --- Remaining unmatched HEL1OS events ---
    for j, hx_ev in enumerate(hel1os_events):
        if j in matched_hel1os_indices:
            continue
        confidence = hx_ev.confidence * 0.60
        rows.append({
            "start_time": hx_ev.start_time,
            "peak_time": hx_ev.peak_time,
            "end_time": hx_ev.end_time,
            "flare_class": "?",  # HEL1OS cannot classify to GOES scale
            "peak_flux_sxr": np.nan,
            "peak_cnt_hxr": hx_ev.peak_flux,
            "source": "HEL1OS_only",
            "confidence": confidence,
            "noaa_confirmed": _check_noaa(hx_ev.peak_time, noaa_catalog),
            "solexs_fits_path": "",
            "hel1os_fits_path": hx_ev.fits_path,
            "model_version": model_version,
            "pipeline_run_id": pipeline_run_id,
        })

    df = pd.DataFrame(rows, columns=CATALOGUE_COLUMNS)

    # --- Persist: append + dedup (skip when catalogue_path is None) ---
    if catalogue_path is not None:
        if catalogue_path.exists():
            existing = pd.read_csv(catalogue_path, parse_dates=["peak_time"])
            df = pd.concat([existing, df], ignore_index=True)
            df = df.drop_duplicates(subset=["peak_time", "source"], keep="last")

        catalogue_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(catalogue_path, index=False)

    # --- Summary ---
    n_dual = (df["source"] == "dual").sum()
    n_sx = (df["source"] == "SoLEXS_only").sum()
    n_hx = (df["source"] == "HEL1OS_only").sum()
    n_noaa = df["noaa_confirmed"].sum()
    print(
        f"Master catalogue: {len(df)} events  "
        f"(dual={n_dual}, SoLEXS_only={n_sx}, HEL1OS_only={n_hx}, "
        f"NOAA_confirmed={n_noaa})"
    )

    return df
