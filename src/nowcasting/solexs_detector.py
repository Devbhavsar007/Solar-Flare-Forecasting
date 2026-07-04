import datetime
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class FlareEvent:
    start_time: datetime.datetime
    peak_time: datetime.datetime
    end_time: datetime.datetime
    peak_flux: float
    flare_class: str
    instrument: str
    confidence: float
    fits_path: str = ""


GOES_THRESHOLDS = {
    "X": 1e-4,
    "M": 1e-5,
    "C": 1e-6,
    "B": 1e-7
}


def classify_flux(peak_flux: float) -> str:
    """Classifies solar flare peak flux according to GOES scale."""
    if peak_flux >= GOES_THRESHOLDS["X"]:
        return "X"
    elif peak_flux >= GOES_THRESHOLDS["M"]:
        return "M"
    elif peak_flux >= GOES_THRESHOLDS["C"]:
        return "C"
    elif peak_flux >= GOES_THRESHOLDS["B"]:
        return "B"
    return "N"  # N for normal/quiet


def detect_solexs_flares(
    df: pd.DataFrame,
    flux_col: str = "flux_high",
    sigma_threshold: float = 3.0,
    min_duration_min: int = 3,
    fits_path: str = ""
) -> list[FlareEvent]:
    """
    Detects flares using a dual-trigger algorithm on SoLEXS data.
    
    Args:
        df: DataFrame containing FITS data with a DatetimeIndex
        flux_col: Name of the column containing flux data
        sigma_threshold: Standard deviations above background for rate-of-change trigger
        min_duration_min: Minimum duration in minutes for a valid flare
        fits_path: Provenance path to the FITS file
        
    Returns:
        List of FlareEvent objects.
    """
    if df.empty or flux_col not in df.columns:
        return []

    # Calculate rolling 10-min median background and std (excluding current point)
    rolling_window = df[flux_col].rolling("10min", min_periods=1)
    bg = rolling_window.median().shift(1).bfill()
    std = rolling_window.std().shift(1).fillna(0)
    
    # Calculate rate of change
    roc = df[flux_col].diff().fillna(0)
    
    # Dual trigger logic
    # Condition 1: Rate of change > sigma * std
    # Condition 2: Flux > 1.5 * background
    trigger_roc = roc > (sigma_threshold * std)
    trigger_flux = df[flux_col] > (1.5 * bg)
    
    is_active = trigger_roc & trigger_flux
    
    events = []
    in_event = False
    start_time = None
    peak_time = None
    peak_flux = -1.0
    start_bg = 0.0  # pre-flare background for confidence
    
    for time, row in df.iterrows():
        flux = row[flux_col]
        current_bg = bg.loc[time]
        
        if is_active.loc[time] and not in_event:
            in_event = True
            start_time = time
            start_bg = current_bg  # capture pre-flare background
            peak_time = time
            peak_flux = flux
            
        elif in_event:
            if flux > peak_flux:
                peak_flux = flux
                peak_time = time
                
            # Event ends when flux drops below 1.5 * background
            if flux < 1.5 * current_bg:
                end_time = time
                duration = (end_time - start_time).total_seconds() / 60.0
                
                if duration >= min_duration_min:
                    confidence = float(np.clip(peak_flux / (5 * start_bg), 0.0, 1.0))
                    flare_class = classify_flux(peak_flux)
                    
                    events.append(FlareEvent(
                        start_time=start_time,
                        peak_time=peak_time,
                        end_time=end_time,
                        peak_flux=float(peak_flux),
                        flare_class=flare_class,
                        instrument="SoLEXS",
                        confidence=confidence,
                        fits_path=fits_path
                    ))
                    
                in_event = False
                
    # Handle event still ongoing at end of data
    if in_event:
        end_time = df.index[-1]
        duration = (end_time - start_time).total_seconds() / 60.0
        current_bg = bg.iloc[-1]
        
        if duration >= min_duration_min:
            confidence = float(np.clip(peak_flux / (5 * start_bg), 0.0, 1.0))
            flare_class = classify_flux(peak_flux)
            
            events.append(FlareEvent(
                start_time=start_time,
                peak_time=peak_time,
                end_time=end_time,
                peak_flux=float(peak_flux),
                flare_class=flare_class,
                instrument="SoLEXS",
                confidence=confidence,
                fits_path=fits_path
            ))
            
    return events
