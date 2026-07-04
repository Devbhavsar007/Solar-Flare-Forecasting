"""
LangGraph pipeline state definitions for JWALA.

Uses Annotated reducers to handle fan-in from parallel detector agents
(Decision D7). The _merge_catalogue_lists reducer concatenates detected
events from independent SoLEXS and HEL1OS detector nodes.
"""
import operator
from typing import TypedDict, Annotated, Optional


def _merge_catalogue_lists(a: Optional[list], b: Optional[list]) -> list:
    """Reducer to merge detected events from parallel detectors."""
    if a is None:
        a = []
    if b is None:
        b = []
    return a + b


def _merge_timing_dicts(a: Optional[dict], b: Optional[dict]) -> dict:
    """Reducer to merge timing dictionaries from different agents [SLO-1]."""
    if a is None:
        a = {}
    if b is None:
        b = {}
    return {**a, **b}


class SolarPipelineState(TypedDict):
    """
    Stateful container for the LangGraph solar flare pipeline.

    Fields use Annotated reducers so that parallel branches
    (e.g. SoLEXS and HEL1OS detection) merge correctly into a single
    downstream node without data loss.
    """
    # Provenance [RULE-17]
    pipeline_run_id: str
    model_version: str

    # Input paths
    solexs_fits_path: Optional[str]
    hel1os_fits_path: Optional[str]

    # Event tracking — Annotated reducer handles fan-in from parallel detectors
    detected_events: Annotated[list, _merge_catalogue_lists]

    # Nowcast results
    nowcast_class: Optional[str]
    nowcast_confidence: Optional[float]
    nowcast_proba: Optional[dict]

    # Forecast results
    forecast_probs: Optional[dict]

    # Uncertainty results
    uncertainty_intervals: Optional[dict]

    # Alert and reporting
    alert_triggered: bool
    llm_report: Optional[str]
    shap_explanation: Optional[dict]

    # Error tracking — appending reducer
    errors: Annotated[list, operator.add]

    # Timing instrumentation [SLO-1] — every agent logs elapsed seconds
    timing: Annotated[dict, _merge_timing_dicts]
