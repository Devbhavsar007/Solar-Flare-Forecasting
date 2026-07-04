"""
MOMENT-1-large reconstruction-error anomaly detector.

Identifies pre-flare precursors using time-series reconstruction error.
Includes safety guards for MOMENT API output attribute drift and leverages
the physics phase detector to skip computationally heavy inference on
quiet sun windows, satisfying the real-time latency [SLO-1] budget.
"""
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.physics.phase_detector import build_phase_sequence

# Verified by CHECK 4 — update if the momentfm package changes the API
RECONSTRUCTION_ATTR = "reconstruction"

# Module-level singleton [RULE-16]
_MOMENT = None


def get_moment_model():
    """
    Load the MOMENT model as a module-level singleton.
    Takes ~30s to load, so we only do it once per process lifetime.
    """
    global _MOMENT
    if _MOMENT is None:
        from momentfm import MOMENTPipeline
        _MOMENT = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={"task_name": "reconstruction"}
        )
        _MOMENT.init()
    return _MOMENT


def _get_reconstruction(output) -> torch.Tensor:
    """
    Safely extract reconstruction tensor from MOMENT output object.
    
    MOMENT's output attribute has changed across versions; this function
    centralises the lookup and gives an actionable error if it changes again.
    """
    result = getattr(output, RECONSTRUCTION_ATTR, None)
    if result is None:
        available = [a for a in dir(output)
                     if not a.startswith("_") and isinstance(
                         getattr(output, a, None), torch.Tensor)]
        raise AttributeError(
            f"MOMENT output has no attribute '{RECONSTRUCTION_ATTR}'.\n"
            f"Available tensor attributes: {available}\n"
            f"Update RECONSTRUCTION_ATTR in moment_anomaly.py and "
            f"re-run CHECK 4 to confirm."
        )
    return result


def compute_reconstruction_error(flux_window: np.ndarray) -> float:
    """
    Compute MSE reconstruction error for a 512-step flux window.
    """
    model = get_moment_model()
    
    # Ensure window is exactly 512 steps, padding if necessary
    if len(flux_window) < 512:
        pad_width = 512 - len(flux_window)
        flux_window = np.pad(flux_window, (pad_width, 0), mode='edge')
    else:
        flux_window = flux_window[-512:]
        
    x = torch.tensor(flux_window, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 512)
    
    with torch.no_grad():
        output = model(x)
        
    recon = _get_reconstruction(output).squeeze()
    
    # Compute MSE between reconstruction and original input
    error = float(F.mse_loss(recon, x.squeeze()).item())
    return error


def classify_anomaly(error: float, threshold: float | None = None) -> bool:
    """
    Classify if a reconstruction error constitutes an anomaly.
    """
    if threshold is None:
        # Load default threshold from configs (placeholder logic)
        threshold = 0.05
    return error > threshold


def batch_compute_moment_scores(
    df: pd.DataFrame, 
    flux_col: str, 
    window_size: int = 512, 
    stride: int = 60
) -> pd.DataFrame:
    """
    Compute MOMENT anomaly scores across a full DataFrame.
    
    Critically, this skips quiet-sun windows using the phase detector
    to satisfy [SLO-1].
    """
    phases = build_phase_sequence(df, flux_col=flux_col, window_size=10)
    
    n = len(df)
    scores = np.full(n, np.nan)
    flux = df[flux_col].values
    
    for i in tqdm(range(window_size - 1, n, stride), desc="MOMENT Anomaly Detection"):
        # Check if this window ends in a pre-flare (1) or impulsive (2) phase
        current_phase = phases[i]
        
        # Skip quiet sun (0) to save the 30s+ inference budget per window
        if current_phase == 0:
            continue
            
        window = flux[i - window_size + 1 : i + 1]
        error = compute_reconstruction_error(window)
        scores[i] = error
        
    df["moment_reconstruction_error"] = scores
    return df
