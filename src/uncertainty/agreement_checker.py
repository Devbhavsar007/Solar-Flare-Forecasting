"""
Dual uncertainty agreement checker.

Compares independent uncertainty signals from MAPIE (conformal prediction
set size) and Chronos-Bolt (forecast standard deviation) to flag events
where both models indicate high uncertainty.
"""


def check_dual_agreement(
    mapie_set_size: int,
    chronos_std: float,
    mapie_threshold: int = 2,
    chronos_std_threshold: float = 1e-7,
) -> str:
    """
    Check if MAPIE and Chronos both signal high uncertainty.

    Args:
        mapie_set_size: Number of classes in the MAPIE conformal prediction set.
        chronos_std: Standard deviation of Chronos forecast sample means.
        mapie_threshold: If set size >= this, MAPIE signals high uncertainty.
        chronos_std_threshold: If std >= this, Chronos signals high uncertainty.

    Returns:
        "HIGH_UNCERTAINTY" if both signals exceed thresholds, else "NORMAL".
    """
    mapie_high = mapie_set_size >= mapie_threshold
    chronos_high = chronos_std >= chronos_std_threshold

    if mapie_high and chronos_high:
        return "HIGH_UNCERTAINTY"
    return "NORMAL"
