import pandas as pd
import numpy as np

# Import only FlareEvent, not _check_noaa or other internal details [RULE-15]
from src.nowcasting.solexs_detector import FlareEvent


def detect_hel1os_flares(
    df: pd.DataFrame,
    counts_col: str = "counts_low",
    sigma_threshold: float = 4.0,
    min_duration_min: int = 2,
    fits_path: str = ""
) -> list[FlareEvent]:
    """
    Detects flares using a dual-trigger algorithm on HEL1OS data.
    
    Args:
        df: DataFrame containing FITS data with a DatetimeIndex
        counts_col: Name of the column containing counts data
        sigma_threshold: Standard deviations above background for rate-of-change trigger
        min_duration_min: Minimum duration in minutes for a valid flare
        fits_path: Provenance path to the FITS file
        
    Returns:
        List of FlareEvent objects.
    """
    if df.empty or counts_col not in df.columns:
        return []

    # Calculate rolling 10-min median background and std (excluding current point)
    rolling_window = df[counts_col].rolling("10min", min_periods=1)
    bg = rolling_window.median().shift(1).bfill()
    std = rolling_window.std().shift(1).fillna(0)
    
    # Calculate rate of change
    roc = df[counts_col].diff().fillna(0)
    
    # Dual trigger logic
    # Condition 1: Rate of change > sigma * std
    # Condition 2: Counts > 1.5 * background
    trigger_roc = roc > (sigma_threshold * std)
    trigger_flux = df[counts_col] > (1.5 * bg)
    
    is_active = trigger_roc & trigger_flux
    
    events = []
    in_event = False
    start_time = None
    peak_time = None
    peak_flux = -1.0
    start_bg = 0.0  # pre-flare background for confidence
    
    for time, row in df.iterrows():
        flux = row[counts_col]
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
                    # confidence capped at 1.0
                    confidence = float(np.clip(peak_flux / (5 * start_bg) if start_bg > 0 else 1.0, 0.0, 1.0))
                    
                    # HEL1OS cannot classify on GOES scale
                    flare_class = "?"
                    
                    events.append(FlareEvent(
                        start_time=start_time,
                        peak_time=peak_time,
                        end_time=end_time,
                        peak_flux=float(peak_flux),
                        flare_class=flare_class,
                        instrument="HEL1OS",
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
            confidence = float(np.clip(peak_flux / (5 * start_bg) if start_bg > 0 else 1.0, 0.0, 1.0))
            
            events.append(FlareEvent(
                start_time=start_time,
                peak_time=peak_time,
                end_time=end_time,
                peak_flux=float(peak_flux),
                flare_class="?",
                instrument="HEL1OS",
                confidence=confidence,
                fits_path=fits_path
            ))
            
    return events
