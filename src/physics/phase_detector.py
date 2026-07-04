"""
Solar flare 5-phase detector.

Classifies flux windows into one of five flare phases:
  0 = quiet
  1 = pre-flare
  2 = impulsive
  3 = peak
  4 = gradual

Configuration is loaded from configs/nowcasting.yaml at module level
with a FileNotFoundError guard [RULE-12].
"""
import os
import numpy as np
import pandas as pd

# ── Module-level config load with [RULE-12] guard ───────────────
_DEFAULT_PHASE_CFG = {
    "peak_frac_pre": 0.30,
    "peak_frac_gradual": 0.80,
    "background_sigma": 0.10,
}

PHASE_CFG = _DEFAULT_PHASE_CFG.copy()

try:
    import yaml

    _cfg_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "configs",
        "nowcasting.yaml",
    )
    with open(_cfg_path, "r") as _f:
        _raw = yaml.safe_load(_f)
        if isinstance(_raw, dict) and "phase_detector" in _raw:
            PHASE_CFG.update(_raw["phase_detector"])
except FileNotFoundError:
    pass  # Use defaults — [RULE-12] guard
except Exception:
    pass  # Gracefully handle any other config issues


def detect_phase(
    flux_window: np.ndarray, peak_frac: dict | None = None
) -> int:
    """
    Detect the current flare phase from a flux window.

    Args:
        flux_window: 1-D array of flux values (time-ordered, most recent last).
        peak_frac: Optional override for phase thresholds.
                   Keys: 'peak_frac_pre', 'peak_frac_gradual', 'background_sigma'.

    Returns:
        Phase integer: 0=quiet, 1=pre-flare, 2=impulsive, 3=peak, 4=gradual.
    """
    cfg = peak_frac if peak_frac is not None else PHASE_CFG

    if len(flux_window) < 3:
        return 0

    peak_val = np.max(flux_window)
    current_val = flux_window[-1]
    mean_val = np.mean(flux_window)

    peak_frac_pre = cfg.get("peak_frac_pre", 0.30)
    peak_frac_gradual = cfg.get("peak_frac_gradual", 0.80)
    background_sigma = cfg.get("background_sigma", 0.10)

    # Quiet-sun threshold
    background = np.min(flux_window)
    quiet_threshold = background * (1.0 + background_sigma)

    if peak_val <= quiet_threshold or peak_val == 0:
        return 0  # quiet

    # Normalised current value relative to peak
    frac = current_val / peak_val if peak_val > 0 else 0.0

    # Find peak index
    peak_idx = np.argmax(flux_window)
    current_idx = len(flux_window) - 1

    if current_idx < peak_idx:
        # We are before the peak
        if frac < peak_frac_pre:
            return 1  # pre-flare
        else:
            return 2  # impulsive (rising fast toward peak)
    elif current_idx == peak_idx:
        return 3  # peak
    else:
        # We are after the peak
        if frac >= peak_frac_gradual:
            return 3  # still near peak
        else:
            return 4  # gradual (decay)


def build_phase_sequence(
    df: pd.DataFrame, flux_col: str = "flux_high", window_size: int = 10
) -> np.ndarray:
    """
    Build a phase classification sequence over a DataFrame.

    Applies detect_phase in a rolling window over the flux column.

    Args:
        df: DataFrame with a flux column.
        flux_col: Name of the flux column to use.
        window_size: Number of rows per rolling window.

    Returns:
        Integer array of shape (len(df),) with phase classifications.
    """
    flux = df[flux_col].values
    n = len(flux)
    phases = np.zeros(n, dtype=int)

    for i in range(n):
        start = max(0, i - window_size + 1)
        window = flux[start : i + 1]
        phases[i] = detect_phase(window)

    return phases
