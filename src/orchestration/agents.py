"""
LangGraph agent node functions for the JWALA solar flare pipeline.

Each agent function receives and returns a SolarPipelineState dict.
Every agent logs its wall-clock time to state["timing"] [SLO-1].

Module-level singletons [RULE-16] ensure heavyweight models load once.
Module-level config opens use FileNotFoundError guards [RULE-12].
Prometheus metrics imported from src/monitoring/metrics.py only [RULE-20].
"""
import os
import time
import uuid
import yaml
import numpy as np
import torch

from src.orchestration.state import SolarPipelineState
from src.nowcasting.solexs_detector import FlareEvent, detect_solexs_flares
from src.nowcasting.hel1os_detector import detect_hel1os_flares
from src.catalogue.merger import merge_catalogues
from src.monitoring.metrics import (
    ALERT_COUNTER,
    DATA_FRESHNESS,
    NOWCAST_CONFIDENCE,
    INFERENCE_LATENCY,
)

# ---------------------------------------------------------------------------
# Module-level config loading [RULE-12]
# ---------------------------------------------------------------------------
try:
    with open("configs/nowcasting.yaml") as f:
        _CFG = yaml.safe_load(f) or {}
except FileNotFoundError:
    _CFG = {}

try:
    with open("configs/version.yaml") as f:
        _MODEL_VERSION = yaml.safe_load(f).get("model_version", "unknown")
except FileNotFoundError:
    _MODEL_VERSION = "unknown"

# Class thresholds from config
_CLASS_THRESHOLDS = _CFG.get("class_thresholds", {
    "C": 0.38, "M": 0.45, "X": 0.28, "binary": 0.40
})


# ---------------------------------------------------------------------------
# Singleton model loaders [RULE-16]
# ---------------------------------------------------------------------------
_XGB_MODEL = None


def _get_xgb_model():
    """Load XGBoost model from .pkl (joblib) [RULE-13]. Cached as singleton."""
    global _XGB_MODEL
    if _XGB_MODEL is None:
        import joblib
        pkl_path = "models/xgb_multiclass.pkl"
        if os.path.exists(pkl_path):
            _XGB_MODEL = joblib.load(pkl_path)
        # If model file doesn't exist yet, return None (stub mode)
    return _XGB_MODEL


_LSTM_MODEL = None


def _get_lstm_model():
    """Load CausalLSTM from state_dict [RULE-11, RULE-13]. Cached as singleton."""
    global _LSTM_MODEL
    if _LSTM_MODEL is None:
        pt_path = "models/causal_lstm.pt"
        if os.path.exists(pt_path):
            from src.forecasting.causal_lstm import CausalLSTMForecaster
            model = CausalLSTMForecaster(n_features=9)
            model.load_state_dict(
                torch.load(pt_path, map_location="cpu", weights_only=True)
            )
            model.eval()
            _LSTM_MODEL = model
    return _LSTM_MODEL


_ONNX_NOWCASTER = None


def _get_onnx_nowcaster():
    """Load ONNX TCN + XGBoost nowcaster [T-2 hash verification]. Cached."""
    global _ONNX_NOWCASTER
    if _ONNX_NOWCASTER is None:
        onnx_path = "models/tcn_encoder.onnx"
        xgb = _get_xgb_model()
        if os.path.exists(onnx_path) and xgb is not None:
            from src.deployment.onnx_export import ONNXNowcaster
            _ONNX_NOWCASTER = ONNXNowcaster(onnx_path, xgb)
    return _ONNX_NOWCASTER


# ---------------------------------------------------------------------------
# Agent functions
# ---------------------------------------------------------------------------

def ingestion_agent(state: SolarPipelineState) -> dict:
    """
    Read FITS file paths and initialise provenance fields [RULE-17].

    Sets pipeline_run_id (UUID4) and model_version for downstream agents.
    Logs data freshness to Prometheus [RULE-20].
    """
    t0 = time.perf_counter()
    updates: dict = {"errors": []}

    try:
        run_id = str(uuid.uuid4())
        updates["pipeline_run_id"] = run_id
        updates["model_version"] = _MODEL_VERSION

        # Log data freshness if FITS paths are present
        solexs_path = state.get("solexs_fits_path", "")
        hel1os_path = state.get("hel1os_fits_path", "")

        if solexs_path and os.path.exists(solexs_path):
            mtime = os.path.getmtime(solexs_path)
            DATA_FRESHNESS.set(time.time() - mtime)

    except Exception as exc:
        updates["errors"] = [f"ingestion_agent: {exc}"]
    finally:
        updates["timing"] = {"ingestion": round(time.perf_counter() - t0, 3)}
    return updates


def solexs_detect_agent(state: SolarPipelineState) -> dict:
    """
    Run independent SoLEXS flare detection [RULE-1].

    Operates on a single-instrument dataframe. Never touches HEL1OS data.
    Detected events are fed into the Annotated reducer for fan-in at merge.
    """
    t0 = time.perf_counter()
    updates: dict = {"errors": []}

    try:
        # In production, would read the FITS file here. For orchestration,
        # we expect detected_events to be populated by the caller or
        # we detect from pre-loaded data in the state.
        solexs_path = state.get("solexs_fits_path", "")
        events: list[FlareEvent] = []

        # If a solexs dataframe were provided via state, detect flares.
        # Stub: the actual FITS reading + detection is wired in production.
        updates["detected_events"] = events

    except Exception as exc:
        updates["errors"] = [f"solexs_detect_agent: {exc}"]
        updates["detected_events"] = []
    finally:
        updates["timing"] = {"solexs_detect": round(time.perf_counter() - t0, 3)}
    return updates


def hel1os_detect_agent(state: SolarPipelineState) -> dict:
    """
    Run independent HEL1OS flare detection [RULE-1].

    Operates on a single-instrument dataframe. Never touches SoLEXS data.
    Detected events are fed into the Annotated reducer for fan-in at merge.
    """
    t0 = time.perf_counter()
    updates: dict = {"errors": []}

    try:
        hel1os_path = state.get("hel1os_fits_path", "")
        events: list[FlareEvent] = []

        # Stub: actual detection wired in production.
        updates["detected_events"] = events

    except Exception as exc:
        updates["errors"] = [f"hel1os_detect_agent: {exc}"]
        updates["detected_events"] = []
    finally:
        updates["timing"] = {"hel1os_detect": round(time.perf_counter() - t0, 3)}
    return updates


def merge_agent(state: SolarPipelineState) -> dict:
    """
    Merge detected events from both instruments via temporal coincidence.

    Passes pipeline_run_id and model_version to merge_catalogues [RULE-17].
    """
    t0 = time.perf_counter()
    updates: dict = {"errors": []}

    try:
        all_events = state.get("detected_events", [])

        solexs_events = [e for e in all_events if e.instrument == "SoLEXS"]
        hel1os_events = [e for e in all_events if e.instrument == "HEL1OS"]

        if solexs_events or hel1os_events:
            merge_catalogues(
                solexs_events,
                hel1os_events,
                pipeline_run_id=state.get("pipeline_run_id", ""),
                model_version=state.get("model_version", ""),
            )

    except Exception as exc:
        updates["errors"] = [f"merge_agent: {exc}"]
    finally:
        updates["timing"] = {"merge": round(time.perf_counter() - t0, 3)}
    return updates


def preprocess_agent(state: SolarPipelineState) -> dict:
    """
    Engineer physics features for downstream models.

    Placeholder — full implementation wired when FITS data flows through.
    """
    t0 = time.perf_counter()
    updates: dict = {"errors": []}

    try:
        # In production: call engineer_physics_features, detect_solar_phase
        pass

    except Exception as exc:
        updates["errors"] = [f"preprocess_agent: {exc}"]
    finally:
        updates["timing"] = {"preprocess": round(time.perf_counter() - t0, 3)}
    return updates


def moment_score_agent(state: SolarPipelineState) -> dict:
    """
    Compute MOMENT reconstruction anomaly score [RULE-16 singleton].

    Uses get_moment_model() (public API, not _load_moment) [RULE-15].
    """
    t0 = time.perf_counter()
    updates: dict = {"errors": []}

    try:
        # In production: call compute_moment_anomaly_score on flux window
        pass

    except Exception as exc:
        updates["errors"] = [f"moment_score_agent: {exc}"]
    finally:
        updates["timing"] = {"moment_score": round(time.perf_counter() - t0, 3)}
    return updates


def nowcast_agent(state: SolarPipelineState) -> dict:
    """
    Run ONNX TCN + XGBoost multi-class nowcast [RULE-5].

    Uses singleton ONNXNowcaster [RULE-16]. Logs confidence to
    Prometheus NOWCAST_CONFIDENCE gauge [RULE-20].
    """
    t0 = time.perf_counter()
    updates: dict = {"errors": []}

    try:
        nowcaster = _get_onnx_nowcaster()
        if nowcaster is not None:
            # In production: pass actual window and features
            # result = nowcaster.predict(window, feats)
            # updates["nowcast_class"] = result["class"]
            # updates["nowcast_confidence"] = result["confidence"]
            # updates["nowcast_proba"] = result["proba"]
            # NOWCAST_CONFIDENCE.set(result["confidence"])
            pass
        else:
            updates["nowcast_class"] = "N"
            updates["nowcast_confidence"] = 0.0
            updates["nowcast_proba"] = {"N": 1.0, "C": 0.0, "M": 0.0, "X": 0.0}

    except Exception as exc:
        updates["errors"] = [f"nowcast_agent: {exc}"]
    finally:
        updates["timing"] = {"nowcast": round(time.perf_counter() - t0, 3)}
    return updates


def forecast_agent(state: SolarPipelineState) -> dict:
    """
    Run three-model ensemble forecast (LSTM + TCN + TimesFM) [RULE-2, RULE-6].

    Uses singleton LSTM model [RULE-16].
    """
    t0 = time.perf_counter()
    updates: dict = {"errors": []}

    try:
        # In production: call ThreeModelEnsemble.predict_single()
        pass

    except Exception as exc:
        updates["errors"] = [f"forecast_agent: {exc}"]
    finally:
        updates["timing"] = {"forecast": round(time.perf_counter() - t0, 3)}
    return updates


def uncertainty_agent(state: SolarPipelineState) -> dict:
    """
    Compute Chronos-Bolt probabilistic intervals [RULE-7, RULE-16].

    Uses get_chronos() singleton loader to avoid reloading each cycle.
    """
    t0 = time.perf_counter()
    updates: dict = {"errors": []}

    try:
        # In production: call chronos_forecast_interval() on flux window
        pass

    except Exception as exc:
        updates["errors"] = [f"uncertainty_agent: {exc}"]
    finally:
        updates["timing"] = {"uncertainty": round(time.perf_counter() - t0, 3)}
    return updates


def shap_agent(state: SolarPipelineState) -> dict:
    """
    Generate SHAP explanations for the nowcast prediction.
    """
    t0 = time.perf_counter()
    updates: dict = {"errors": []}

    try:
        # In production: compute SHAP values for the nowcast
        pass

    except Exception as exc:
        updates["errors"] = [f"shap_agent: {exc}"]
    finally:
        updates["timing"] = {"shap": round(time.perf_counter() - t0, 3)}
    return updates


def llm_report_agent(state: SolarPipelineState) -> dict:
    """
    Generate a natural language report using DSPy + Ollama [RULE-10].

    Uses dspy.context() per-call context manager for thread safety.
    Falls back to structured template if LLM is unavailable.
    """
    t0 = time.perf_counter()
    updates: dict = {"errors": []}

    nowcast_class = state.get("nowcast_class", "N")
    confidence = state.get("nowcast_confidence", 0.0)
    pipeline_run_id = state.get("pipeline_run_id", "N/A")
    model_version = state.get("model_version", "N/A")
    peak_flux = 0.0  # Placeholder, usually taken from nowcast/forecast
    lead_time_min = 15
    q10 = 0.0
    q90 = 0.0

    try:
        from src.intelligence.dspy_reporter import SolarFlareReporter
        from src.intelligence.graphrag_retriever import retrieve_flare_context
        
        # 1. Retrieve Context from GraphRAG
        context = retrieve_flare_context(f"What are the implications of a {nowcast_class}-class solar flare?", top_k=3)
        
        # 2. Generate LLM Report
        reporter = SolarFlareReporter()
        report = reporter.forward(
            flare_class=nowcast_class,
            peak_flux=peak_flux,
            lead_time_min=lead_time_min,
            confidence=confidence,
            graphrag_context=context,
            uncertainty_q10=q10,
            uncertainty_q90=q90,
        )
        updates["llm_report"] = report

    except Exception as exc:
        updates["errors"] = [f"llm_report_agent DSPy failed: {exc}"]
        # Fallback to structured message
        from src.intelligence.structured_fallback import build_alert_message
        report = build_alert_message(
            flare_class=nowcast_class,
            peak_flux=peak_flux,
            lead_min=lead_time_min,
            q10=q10,
            q90=q90,
            pipeline_run_id=pipeline_run_id,
            model_version=model_version
        )
        updates["llm_report"] = report
    finally:
        updates["timing"] = {"llm_report": round(time.perf_counter() - t0, 3)}
    return updates


def alert_router(state: SolarPipelineState) -> str:
    """
    Conditional edge: route to LLM report for M/X flares, else end.

    Fires Prometheus ALERT_COUNTER [RULE-20] when an alert triggers.

    Returns:
        "llm_report" if alert should fire, "end" otherwise.
    """
    nowcast_class = state.get("nowcast_class", "N")
    confidence = state.get("nowcast_confidence", 0.0)

    # Alert on M or X class with sufficient confidence
    if nowcast_class in ("M", "X") and confidence >= 0.3:
        ALERT_COUNTER.labels(
            flare_class=nowcast_class,
            source="pipeline",
        ).inc()
        return "llm_report"

    return "end"
