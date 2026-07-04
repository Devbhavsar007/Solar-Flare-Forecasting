"""
Chronos-Bolt probabilistic uncertainty intervals for solar flare forecasting.

Uses a module-level singleton [RULE-16] so that the ~20s model load only
happens once per process lifetime, keeping the real-time pipeline viable.
"""
import numpy as np
import torch
from typing import Optional

# Module-level singleton [RULE-16] — Chronos takes ~20s to load.
_CHRONOS_PIPELINE = None


def get_chronos():
    """
    Return the cached Chronos-Bolt pipeline, loading it on first call.

    WHY SINGLETON: Chronos takes ~20s to load. Loading inside an agent
    function that runs every 60s cadence would break [SLO-1] latency.
    """
    global _CHRONOS_PIPELINE
    if _CHRONOS_PIPELINE is None:
        import time
        t0 = time.perf_counter()
        from chronos import BaseChronosPipeline
        _CHRONOS_PIPELINE = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-small",
            dtype=torch.float32,
            device_map="cpu",
        )
        print(f"Chronos loaded in {time.perf_counter() - t0:.1f}s — singleton cached.")
    return _CHRONOS_PIPELINE


def chronos_forecast_interval(
    flux_window: np.ndarray,
    prediction_length: int = 15,
    num_samples: int = 200,
    q: tuple = (0.10, 0.90),
) -> dict:
    """
    Generate probabilistic forecast intervals using Chronos-Bolt.

    Uses predict() — NOT predict_quantiles() [RULE-7]. Quantiles are
    computed manually from the sample distribution via np.quantile().

    Args:
        flux_window: 1D array of recent flux values (context window).
        prediction_length: Number of future steps to forecast.
        num_samples: Number of Monte Carlo samples for interval estimation.
        q: Tuple of (lower, upper) quantile levels.

    Returns:
        dict with keys 'q10', 'q90', 'median', 'std'.
    """
    pipeline = get_chronos()

    # Chronos expects a 2D context tensor: (batch, context_length)
    context = torch.tensor(flux_window, dtype=torch.float32).unsqueeze(0)

    # predict() returns (batch, num_samples, prediction_length) [RULE-7]
    forecast = pipeline.predict(
        context,
        prediction_length=prediction_length,
        num_samples=num_samples,
    )

    # Convert to numpy for quantile computation
    samples = forecast[0].numpy()  # (num_samples, prediction_length)

    q10 = float(np.quantile(samples, q[0]))
    q90 = float(np.quantile(samples, q[1]))

    return {
        "q10": q10,
        "q90": q90,
        "median": float(np.median(samples)),
        "std": float(np.std(samples.mean(axis=1))),
    }

