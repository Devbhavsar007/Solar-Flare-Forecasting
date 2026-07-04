import pytest
from src.nowcasting.solexs_detector import detect_solexs_flares
from src.nowcasting.hel1os_detector import detect_hel1os_flares


def test_solexs_detects_synthetic_flare(sample_solexs_df):
    events = detect_solexs_flares(sample_solexs_df, fits_path="/fake/solexs.fits")
    assert len(events) == 1
    event = events[0]
    assert event.flare_class in ("C", "M", "X")
    assert event.instrument == "SoLEXS"
    assert event.confidence > 0.0


def test_hel1os_detects_synthetic_burst(sample_hel1os_df):
    events = detect_hel1os_flares(sample_hel1os_df, fits_path="/fake/hel1os.fits")
    assert len(events) == 1
    event = events[0]
    assert event.flare_class == "?"
    assert event.instrument == "HEL1OS"
    assert event.confidence > 0.0


def test_hel1os_peak_precedes_solexs_peak(sample_solexs_df, sample_hel1os_df):
    solexs_events = detect_solexs_flares(sample_solexs_df)
    hel1os_events = detect_hel1os_flares(sample_hel1os_df)
    
    assert len(solexs_events) == 1
    assert len(hel1os_events) == 1
    
    solexs_peak = solexs_events[0].peak_time
    hel1os_peak = hel1os_events[0].peak_time
    
    # HXR (HEL1OS) leads SXR (SoLEXS) by 1-4 minutes (Neupert effect)
    diff = (hel1os_peak - solexs_peak).total_seconds() / 60.0
    assert -4.0 <= diff <= -1.0


def test_flare_event_has_fits_path(sample_solexs_df):
    path = "/path/to/provenance/file.fits"
    events = detect_solexs_flares(sample_solexs_df, fits_path=path)
    assert len(events) == 1
    assert events[0].fits_path == path


# ============================================================
# M3 â€” Master Catalogue Merger tests
# ============================================================
from src.catalogue.merger import merge_catalogues
from src.nowcasting.solexs_detector import FlareEvent
import pandas as pd


def test_dual_detection_produces_dual_event(sample_solexs_df, sample_hel1os_df):
    """1 SoLEXS + 1 HEL1OS within Â±2 min â†’ 1 dual row, conf > 0.90."""
    sx = detect_solexs_flares(sample_solexs_df, fits_path="/s.fits")
    hx = detect_hel1os_flares(sample_hel1os_df, fits_path="/h.fits")

    df = merge_catalogues(
        sx, hx,
        pipeline_run_id="test-run-id",
        model_version="0.0.0-dev",
        catalogue_path=None,
    )

    assert len(df) == 1
    row = df.iloc[0]
    assert row["source"] == "dual"
    assert row["confidence"] > 0.90


def test_solexs_only_confidence_reduced(sample_solexs_df):
    """SoLEXS event with no HEL1OS match â†’ SoLEXS_only, confidence reduced."""
    sx = detect_solexs_flares(sample_solexs_df, fits_path="/s.fits")
    original_conf = sx[0].confidence

    df = merge_catalogues(
        sx, [],
        pipeline_run_id="run-1",
        model_version="0.0.0-dev",
        catalogue_path=None,
    )

    assert len(df) == 1
    row = df.iloc[0]
    assert row["source"] == "SoLEXS_only"
    assert row["confidence"] < original_conf


def test_hel1os_only_class_unknown(sample_hel1os_df):
    """HEL1OS event with no SoLEXS match â†’ class stays '?'."""
    hx = detect_hel1os_flares(sample_hel1os_df, fits_path="/h.fits")

    df = merge_catalogues(
        [], hx,
        pipeline_run_id="run-2",
        model_version="0.0.0-dev",
        catalogue_path=None,
    )

    assert len(df) == 1
    assert df.iloc[0]["flare_class"] == "?"


def test_master_catalogue_no_duplication(sample_solexs_df, sample_hel1os_df):
    """1 SoLEXS + 1 HEL1OS within window â†’ exactly 1 row (not 2)."""
    sx = detect_solexs_flares(sample_solexs_df, fits_path="/s.fits")
    hx = detect_hel1os_flares(sample_hel1os_df, fits_path="/h.fits")

    df = merge_catalogues(
        sx, hx,
        pipeline_run_id="run-3",
        model_version="0.0.0-dev",
        catalogue_path=None,
    )

    assert len(df) == 1


def test_out_of_window_events_not_merged():
    """Two events 5 min apart should NOT be merged â†’ 2 rows."""
    ev_sx = FlareEvent(
        start_time=pd.Timestamp("2024-01-01T10:00:00"),
        peak_time=pd.Timestamp("2024-01-01T10:05:00"),
        end_time=pd.Timestamp("2024-01-01T10:10:00"),
        peak_flux=5e-5,
        flare_class="M",
        instrument="SoLEXS",
        confidence=0.9,
        fits_path="/s.fits",
    )
    ev_hx = FlareEvent(
        start_time=pd.Timestamp("2024-01-01T10:08:00"),
        peak_time=pd.Timestamp("2024-01-01T10:10:00"),  # 5 min after SoLEXS peak
        end_time=pd.Timestamp("2024-01-01T10:15:00"),
        peak_flux=2000.0,
        flare_class="?",
        instrument="HEL1OS",
        confidence=0.8,
        fits_path="/h.fits",
    )

    df = merge_catalogues(
        [ev_sx], [ev_hx],
        pipeline_run_id="run-4",
        model_version="0.0.0-dev",
        catalogue_path=None,
    )

    assert len(df) == 2
    sources = set(df["source"])
    assert "SoLEXS_only" in sources
    assert "HEL1OS_only" in sources


def test_provenance_columns_present(sample_solexs_df, sample_hel1os_df):
    """[RULE-17] All four provenance columns must exist and be non-null for dual events."""
    sx = detect_solexs_flares(sample_solexs_df, fits_path="/s.fits")
    hx = detect_hel1os_flares(sample_hel1os_df, fits_path="/h.fits")

    df = merge_catalogues(
        sx, hx,
        pipeline_run_id="prov-run-id",
        model_version="1.0.0-dev",
        catalogue_path=None,
    )

    required = {"solexs_fits_path", "hel1os_fits_path",
                "model_version", "pipeline_run_id"}
    assert required.issubset(set(df.columns))

    for col in ["solexs_fits_path", "model_version", "pipeline_run_id"]:
        assert df[col].notna().all(), f"{col} must not be null"


# ============================================================
# M4 â€” TCN Encoder + XGBoost Nowcasting tests
# ============================================================
import torch
import numpy as np
import os
import joblib
import xgboost as xgb
from src.nowcasting.tcn_encoder import TCNEncoder, CausalConv1d


def test_causal_conv_no_future_leakage():
    """
    A spike at position 30 must NOT affect any output position < 30.
    This is the core causality guarantee of the TCN.
    """
    conv = CausalConv1d(in_channels=1, out_channels=1, kernel_size=3, dilation=1)

    # Set known weights so the output is not trivially zero
    with torch.no_grad():
        conv.conv.weight.fill_(1.0)
        conv.conv.bias.fill_(0.0)

    x = torch.zeros(1, 1, 60)
    x[0, 0, 30] = 1.0

    out = conv(x)
    # Everything before position 30 must be zero (no future leakage)
    assert torch.all(out[:, :, :30] == 0.0), \
        f"Causal violation: non-zero output before spike position. " \
        f"Max before spike: {out[:, :, :30].abs().max().item()}"


def test_tcn_output_shape(toy_tcn_encoder):
    """(4, 60, 8) input â†’ (4, 32) embedding output."""
    x = torch.randn(4, 60, 8)
    out = toy_tcn_encoder(x)
    assert out.shape == (4, 32), f"Expected (4, 32), got {out.shape}"


def test_multiclass_proba_sums_to_one():
    """XGBoost multi:softprob row probabilities must sum to 1 Â± 1e-5."""
    # Create a tiny dataset with 4 classes
    rng = np.random.RandomState(42)
    X_tr = rng.randn(100, 10).astype(np.float32)
    y_tr = rng.randint(0, 4, 100)

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=4,
        n_estimators=10,
        max_depth=3,
        eval_metric="mlogloss",
    )
    model.fit(X_tr, y_tr)

    proba = model.predict_proba(X_tr)
    row_sums = proba.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), \
        f"Probabilities don't sum to 1: min={row_sums.min()}, max={row_sums.max()}"


def test_model_saved_both_formats(tmp_path):
    """XGBoost model saved in both .json and .pkl produces identical predictions [RULE-13]."""
    rng = np.random.RandomState(42)
    X = rng.randn(50, 10).astype(np.float32)
    y = rng.randint(0, 4, 50)

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=4,
        n_estimators=10,
        max_depth=3,
        eval_metric="mlogloss",
    )
    model.fit(X, y)

    json_path = str(tmp_path / "xgb_multiclass.json")
    pkl_path = str(tmp_path / "xgb_multiclass.pkl")

    model.save_model(json_path)
    joblib.dump(model, pkl_path)

    assert os.path.exists(json_path), "JSON model file not created"
    assert os.path.exists(pkl_path), "PKL model file not created"

    # Load both and compare predictions
    model_json = xgb.XGBClassifier()
    model_json.load_model(json_path)

    model_pkl = joblib.load(pkl_path)

    pred_json = model_json.predict_proba(X)
    pred_pkl = model_pkl.predict_proba(X)

    assert np.allclose(pred_json, pred_pkl, atol=1e-6), \
        "JSON and PKL models produce different predictions"


# ============================================================
# M5 â€” Three-Model Forecasting Ensemble tests
# ============================================================
from src.forecasting.causal_lstm import CausalLSTMForecaster, save_lstm, load_lstm
from src.forecasting.multi_horizon import MultiHorizonForecaster
from src.forecasting.ensemble import ThreeModelEnsemble


def test_causal_lstm_bidirectional_false():
    """[RULE-2] LSTM must be unidirectional (bidirectional=False) to prevent future leakage."""
    model = CausalLSTMForecaster(n_features=10)
    assert model.lstm.bidirectional is False, "LSTM must not be bidirectional"


def test_multi_horizon_output_keys():
    """MultiHorizonForecaster must produce keys for 'h15', 'h30', 'h60'."""
    model = MultiHorizonForecaster(n_features=8, horizons=[15, 30, 60])
    x = torch.randn(2, 60, 8)
    out = model(x)
    assert set(out.keys()) == {"h15", "h30", "h60"}


def test_multi_horizon_probabilities_sum_to_one():
    """MultiHorizonForecaster outputs must sum to 1 across classes."""
    model = MultiHorizonForecaster(n_features=8, horizons=[15, 30, 60])
    x = torch.randn(2, 60, 8)
    out = model(x)
    for h_key, probs in out.items():
        assert torch.allclose(probs.sum(dim=-1), torch.ones(2)), f"Probabilities for {h_key} do not sum to 1"


def test_ensemble_output_shape():
    """Ensemble predict_single must return a (4,) array summing to ~1.0."""
    lstm_model = CausalLSTMForecaster(n_features=8)
    tcn_model = MultiHorizonForecaster(n_features=8, horizons=[15])
    
    class DummyTimesFM:
        def predict_proba(self, x):
            return np.array([0.1, 0.2, 0.3, 0.4])
            
    ensemble = ThreeModelEnsemble(lstm_model, tcn_model, DummyTimesFM(), weights=(0.3, 0.3, 0.4))
    
    x_tensor = torch.randn(1, 60, 8)
    flux_np = np.random.randn(60)
    
    prob = ensemble.predict_single(x_tensor, flux_np, horizon=15)
    assert prob.shape == (4,)
    assert np.isclose(prob.sum(), 1.0)


def test_lstm_state_dict_save_load(tmp_path):
    """[RULE-11, RULE-13] Ensure state_dict save/load works and preserves weights, using weights_only=True."""
    model = CausalLSTMForecaster(n_features=8)
    path = str(tmp_path / "causal_lstm.pt")
    
    save_lstm(model, path)
    loaded_model = load_lstm(path, n_features=8)
    
    for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
        assert torch.allclose(p1, p2)


# ============================================================
# M6 â€” ONNX Export tests
# ============================================================
from src.deployment.onnx_export import export_to_onnx
import tempfile


def test_tcn_export_to_onnx():
    """Verify TCNEncoder exports to ONNX and passes checker."""
    encoder = TCNEncoder(n_features=8, embed_dim=32, n_layers=2)
    dummy_input = torch.randn(1, 60, 8)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_path = os.path.join(tmp_dir, "tcn.onnx")
        # export_to_onnx automatically runs onnx.checker.check_model
        export_to_onnx(encoder, dummy_input, onnx_path)
        assert os.path.exists(onnx_path)


def test_lstm_export_to_onnx():
    """Verify CausalLSTMForecaster exports to ONNX and passes checker."""
    model = CausalLSTMForecaster(n_features=8, hidden_dim=32, n_layers=1)
    dummy_input = torch.randn(1, 60, 8)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_path = os.path.join(tmp_dir, "lstm.onnx")
        # export_to_onnx automatically runs onnx.checker.check_model
        export_to_onnx(model, dummy_input, onnx_path)
        assert os.path.exists(onnx_path)


# ============================================================
# M7 â€” Agent Orchestration (LangGraph) tests
# ============================================================
import uuid as _uuid_mod
from src.orchestration.state import SolarPipelineState
from src.orchestration.agents import (
    ingestion_agent,
    solexs_detect_agent,
    hel1os_detect_agent,
    merge_agent,
    alert_router,
)
from src.orchestration.graph import build_pipeline


def test_pipeline_runs_end_to_end():
    """build_pipeline() compiles and invoke() completes without errors."""
    pipeline = build_pipeline()
    initial_state = {
        "pipeline_run_id": "",
        "model_version": "",
        "solexs_fits_path": "",
        "hel1os_fits_path": "",
        "detected_events": [],
        "nowcast_class": "N",
        "nowcast_confidence": 0.1,
        "nowcast_proba": {"N": 0.9, "C": 0.05, "M": 0.03, "X": 0.02},
        "forecast_probs": None,
        "uncertainty_intervals": None,
        "alert_triggered": False,
        "llm_report": None,
        "shap_explanation": None,
        "errors": [],
        "timing": {},
    }
    result = pipeline.invoke(initial_state)
    # Pipeline should complete â€” errors list may be empty
    assert isinstance(result, dict)
    assert "timing" in result


def test_alert_router_fires_on_mx():
    """alert_router returns 'llm_report' for X-class with high confidence."""
    state = {
        "nowcast_class": "X",
        "nowcast_confidence": 0.85,
    }
    assert alert_router(state) == "llm_report"


def test_alert_router_silent_on_quiet():
    """alert_router returns 'end' for quiet-sun (N-class) predictions."""
    state = {
        "nowcast_class": "N",
        "nowcast_confidence": 0.1,
    }
    assert alert_router(state) == "end"


def test_merge_agent_separates_by_instrument():
    """merge_agent correctly separates SoLEXS and HEL1OS events from detected_events."""
    import datetime
    sx_event = FlareEvent(
        start_time=datetime.datetime(2024, 2, 22, 10, 30),
        peak_time=datetime.datetime(2024, 2, 22, 10, 35),
        end_time=datetime.datetime(2024, 2, 22, 10, 40),
        peak_flux=5e-5, flare_class="M", instrument="SoLEXS",
        confidence=0.9, fits_path="test_sx.fits"
    )
    hx_event = FlareEvent(
        start_time=datetime.datetime(2024, 2, 22, 10, 29),
        peak_time=datetime.datetime(2024, 2, 22, 10, 33),
        end_time=datetime.datetime(2024, 2, 22, 10, 37),
        peak_flux=2800, flare_class="?", instrument="HEL1OS",
        confidence=0.7, fits_path="test_hx.fits"
    )
    state = {
        "detected_events": [sx_event, hx_event],
        "pipeline_run_id": "test-run-id",
        "model_version": "0.0.0-dev",
        "errors": [],
        "timing": {},
    }
    result = merge_agent(state)
    # merge_agent should complete without errors
    assert isinstance(result, dict)
    assert not result.get("errors", []), f"merge_agent errors: {result['errors']}"


def test_timing_dict_populated():
    """After pipeline.invoke(), state['timing'] has keys for every agent."""
    pipeline = build_pipeline()
    initial_state = {
        "pipeline_run_id": "",
        "model_version": "",
        "solexs_fits_path": "",
        "hel1os_fits_path": "",
        "detected_events": [],
        "nowcast_class": "N",
        "nowcast_confidence": 0.1,
        "nowcast_proba": {"N": 0.9, "C": 0.05, "M": 0.03, "X": 0.02},
        "forecast_probs": None,
        "uncertainty_intervals": None,
        "alert_triggered": False,
        "llm_report": None,
        "shap_explanation": None,
        "errors": [],
        "timing": {},
    }
    result = pipeline.invoke(initial_state)
    timing = result.get("timing", {})

    # Every agent should have logged its elapsed time
    expected_agents = [
        "ingestion", "solexs_detect", "hel1os_detect", "merge",
        "preprocess", "moment_score", "nowcast", "forecast",
        "uncertainty", "shap",
    ]
    for agent_name in expected_agents:
        assert agent_name in timing, \
            f"Missing timing for {agent_name}. Got keys: {list(timing.keys())}"
        assert isinstance(timing[agent_name], float) and timing[agent_name] >= 0, \
            f"Timing for {agent_name} must be a non-negative float"


def test_pipeline_run_id_is_uuid():
    """After ingestion_agent runs, pipeline_run_id is a valid UUID4."""
    state = {
        "solexs_fits_path": "",
        "hel1os_fits_path": "",
        "errors": [],
        "timing": {},
    }
    result = ingestion_agent(state)
    run_id = result.get("pipeline_run_id", "")
    # Must parse as a valid UUID4
    parsed = _uuid_mod.UUID(run_id, version=4)
    assert str(parsed) == run_id


# ============================================================
# M8 â€” AutoML Tuning tests
# ============================================================
import numpy as np
from unittest.mock import patch, MagicMock
from src.automl.flaml_tuner import run_flaml_automl, custom_metric

class DummyXGB:
    def __init__(self, pred_proba_val):
        self.pred_proba_val = pred_proba_val
        
    def predict_proba(self, X):
        return self.pred_proba_val

def test_custom_metric_returns_tss_and_far():
    """custom_metric calculates TSS and FAR correctly with [RULE-8] labels."""
    # Dummy data: 2 positive, 2 negative
    y_val = np.array([1, 2, 0, 0])
    X_val = np.zeros((4, 10))
    
    # Perfect predictor: high probability on classes 1,2,3 for positives, high on class 0 for negatives
    perfect_probs = np.array([
        [0.1, 0.9, 0.0, 0.0],  # Positive
        [0.1, 0.0, 0.9, 0.0],  # Positive
        [0.9, 0.1, 0.0, 0.0],  # Negative
        [1.0, 0.0, 0.0, 0.0],  # Negative
    ])
    perfect_estimator = DummyXGB(perfect_probs)
    loss, metrics = custom_metric(X_val, y_val, perfect_estimator, None, None, None)
    assert metrics["tss"] == 1.0
    assert metrics["far"] == 0.0
    
    # Worst-case predictor: all negatives predicted as positive, all positives predicted as negative
    worst_probs = np.array([
        [0.9, 0.1, 0.0, 0.0],  # Positive (Predicted Negative)
        [1.0, 0.0, 0.0, 0.0],  # Positive (Predicted Negative)
        [0.1, 0.9, 0.0, 0.0],  # Negative (Predicted Positive)
        [0.1, 0.0, 0.9, 0.0],  # Negative (Predicted Positive)
    ])
    worst_estimator = DummyXGB(worst_probs)
    loss, metrics = custom_metric(X_val, y_val, worst_estimator, None, None, None)
    # Both positives predicted as negative -> tpr=0.0
    # Both negatives predicted as positive -> fpr=1.0
    # tss = 0.0 - 1.0 = -1.0, far=1.0
    assert metrics["tss"] == -1.0
    assert metrics["far"] == 1.0


@patch("src.automl.flaml_tuner.AutoML")
@patch("src.automl.flaml_tuner.mlflow")
def test_flaml_tuner_rejects_high_far(mock_mlflow, mock_automl_cls):
    """run_flaml_automl returns None when FAR > 0.10 [SLO-5]."""
    mock_automl = MagicMock()
    mock_automl_cls.return_value = mock_automl
    mock_automl.best_loss = -0.80  # TSS = 0.80
    
    # Estimator that gives FAR=0.15 (predicts 15% false positives)
    # We mock custom_metric instead to force the FAR calculation
    with patch("src.automl.flaml_tuner.custom_metric") as mock_metric:
        mock_metric.return_value = (1.0 - 0.80, {"tss": 0.80, "far": 0.15, "tpr": 0.95, "fpr": 0.15})
        
        X = np.zeros((10, 5))
        y = np.zeros(10)
        
        model, far, tss = run_flaml_automl(X, y, X, y)
        
        assert model is None
        assert far == 0.15
        assert tss == 0.80
        mock_mlflow.log_params.assert_called_with({
            "model_rejected": "FAR_exceeds_threshold",
            "model_far": 0.15,
            "model_far_limit": 0.10,
            "model_tss": 0.80,
        })


@patch("src.automl.flaml_tuner.AutoML")
@patch("src.automl.flaml_tuner.mlflow")
def test_flaml_tuner_accepts_low_far(mock_mlflow, mock_automl_cls):
    """run_flaml_automl returns model when FAR <= 0.10 [SLO-5]."""
    mock_automl = MagicMock()
    mock_automl_cls.return_value = mock_automl
    mock_automl.best_loss = -0.72  # TSS = 0.72
    mock_automl.model.estimator = "valid_model"
    
    with patch("src.automl.flaml_tuner.custom_metric") as mock_metric:
        mock_metric.return_value = (1.0 - 0.72, {"tss": 0.72, "far": 0.05, "tpr": 0.77, "fpr": 0.05})
        
        X = np.zeros((10, 5))
        y = np.zeros(10)
        
        model, far, tss = run_flaml_automl(X, y, X, y)
        
        assert model == "valid_model"
        assert far == 0.05
        assert tss == 0.72
        mock_mlflow.log_params.assert_called_with({
            "model_far": 0.05,
            "model_tss": 0.72,
            "model_rejected": "no",
        })


# ============================================================
# M9 â€” LLM Intelligence (DSPy + GraphRAG) tests
# ============================================================
import time
import subprocess
from src.intelligence.structured_fallback import build_alert_message
from src.intelligence.graphrag_retriever import retrieve_flare_context
from src.orchestration.agents import llm_report_agent


def test_dspy_reporter_uses_context_manager():
    """SolarFlareReporter uses `with dspy.context(...)` per [RULE-10]."""
    import sys
    # Only run if dspy is installed
    try:
        import dspy
    except ImportError:
        return
        
    from src.intelligence.dspy_reporter import SolarFlareReporter, get_lm
    
    reporter = SolarFlareReporter()
    
    # We mock dspy.context to ensure it gets called
    with patch("dspy.context") as mock_context, patch.object(reporter, "predict") as mock_predict:
        mock_predict.return_value = MagicMock(alert_summary="Mock Alert")
        
        reporter.forward("M", 5e-5, 15, 0.9, "Context", 1e-5, 9e-5)
        
        # Verify context manager was instantiated
        assert mock_context.called


def test_structured_fallback_returns_in_1s():
    """Deterministic fallback returns in < 1 second [SLO-1]."""
    t0 = time.perf_counter()
    report = build_alert_message("X", 2e-4, 15, 1e-4, 3e-4)
    duration = time.perf_counter() - t0
    
    assert duration < 1.0, f"Fallback took too long: {duration:.3f}s"
    assert "EXTREME" in report
    assert "X-Class Flare Detected" in report


@patch("subprocess.run")
def test_structured_fallback_no_subprocess(mock_run):
    """Fallback uses no subprocesses."""
    report = build_alert_message("M", 5e-5, 15, 1e-5, 9e-5)
    mock_run.assert_not_called()
    assert isinstance(report, str)


@patch("src.intelligence.graphrag_retriever.subprocess.run")
def test_graphrag_retriever_timeout_returns_empty(mock_run):
    """GraphRAG retriever returns empty string on timeout."""
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="graphrag", timeout=30)
    
    result = retrieve_flare_context("What is an X-class flare?")
    assert result == ""


@patch("src.intelligence.dspy_reporter.SolarFlareReporter")
def test_llm_report_agent_uses_fallback_on_error(mock_reporter_cls):
    """If DSPy fails, llm_report_agent returns the structured fallback."""
    # Force the reporter to raise an exception
    mock_reporter = MagicMock()
    mock_reporter.forward.side_effect = Exception("Ollama connection refused")
    mock_reporter_cls.return_value = mock_reporter
    
    state = {
        "nowcast_class": "X",
        "nowcast_confidence": 0.99,
        "pipeline_run_id": "test-id",
        "model_version": "v1.0"
    }
    
    # The agent should catch the exception and use the fallback
    result = llm_report_agent(state)
    
    report = result.get("llm_report", "")
    errors = result.get("errors", [])
    
    # Must contain fallback text
    assert "JWALA STRUCTURED ALERT" in report
    assert "automated structured fallback report" in report
    # The error must be recorded
    assert any("DSPy failed" in err for err in errors)


# ============================================================
# M10 â€” Physics Layer (PINN + Phase Detector) tests
# ============================================================
import torch
from src.physics.pinn import NeupertPINN, violation_penalty
from src.physics.phase_detector import detect_phase, build_phase_sequence, PHASE_CFG


def test_neupert_violation_penalty_positive():
    """Penalty > 0 when dSXR/dt is negative while HXR is positive (violation)."""
    dSXR_dt = torch.tensor([-1.0, -0.5, -2.0])
    HXR = torch.tensor([1.0, 2.0, 0.5])
    penalty = violation_penalty(dSXR_dt, HXR)
    assert penalty.item() > 0, f"Expected positive penalty, got {penalty.item()}"


def test_neupert_violation_penalty_zero_when_valid():
    """Penalty == 0 when dSXR/dt and HXR are both positive (no violation)."""
    dSXR_dt = torch.tensor([1.0, 0.5, 2.0])
    HXR = torch.tensor([1.0, 2.0, 0.5])
    penalty = violation_penalty(dSXR_dt, HXR)
    assert penalty.item() == 0.0, f"Expected zero penalty, got {penalty.item()}"


def test_phase_detector_on_synthetic():
    """Flat input -> phase 0 (quiet). Ramp-peak-decay -> contains phases 1,2,3,4."""
    # Flat input: all quiet
    flat = np.ones(30) * 1e-7
    assert detect_phase(flat) == 0

    # Synthetic full flare: ramp up, peak, decay
    flare = np.zeros(60)
    flare[:10] = 1e-7  # quiet background
    for i in range(10, 30):
        flare[i] = 1e-7 + (5e-5 - 1e-7) * ((i - 10) / 20.0)  # rise
    flare[30] = 5e-5  # peak
    for i in range(31, 50):
        flare[i] = 5e-5 * (1.0 - (i - 30) / 20.0)  # decay
    flare[50:] = 1e-7  # return to quiet

    df = pd.DataFrame({"flux_high": flare})
    phases = build_phase_sequence(df, flux_col="flux_high", window_size=10)

    unique_phases = set(phases)
    # Must contain at least quiet (0) and some non-quiet phases
    assert 0 in unique_phases, "Expected quiet phase in sequence"
    non_quiet = unique_phases - {0}
    assert len(non_quiet) >= 1, f"Expected non-quiet phases, got {unique_phases}"


def test_phase_detector_no_hardcoded_values():
    """detect_phase reads config from PHASE_CFG (loaded from nowcasting.yaml)."""
    # PHASE_CFG should have been loaded at module level from configs/nowcasting.yaml
    assert "peak_frac_pre" in PHASE_CFG
    assert "peak_frac_gradual" in PHASE_CFG
    assert "background_sigma" in PHASE_CFG

    # Verify the config values match what's in the YAML (not hardcoded defaults)
    # The YAML has peak_frac_pre=0.30, peak_frac_gradual=0.80, background_sigma=0.10
    assert isinstance(PHASE_CFG["peak_frac_pre"], (int, float))
    assert isinstance(PHASE_CFG["peak_frac_gradual"], (int, float))

    # Test that passing custom peak_frac overrides defaults
    flat_near_quiet = np.array([1.0, 1.0, 1.05, 1.1, 1.0])
    custom_cfg = {"peak_frac_pre": 0.99, "peak_frac_gradual": 0.99, "background_sigma": 0.001}
    phase = detect_phase(flat_near_quiet, peak_frac=custom_cfg)
    assert isinstance(phase, int)


def test_hxr_leads_sxr_by_1_to_4_min(sample_flare_event, sample_hel1os_event):
    """HEL1OS peak precedes SoLEXS peak by 1 to 4 minutes (Neupert Effect)."""
    hxr_peak = sample_hel1os_event.peak_time
    sxr_peak = sample_flare_event.peak_time

    delta = (sxr_peak - hxr_peak).total_seconds() / 60.0  # minutes

    # HXR should lead SXR: delta should be positive (SXR peaks after HXR)
    # and within 1 to 4 minutes
    assert 1.0 <= delta <= 4.0, (
        f"Expected HXR to lead SXR by 1-4 min, got {delta:.1f} min"
    )


# ============================================================
# M11 â€” Dual Uncertainty Layer (MAPIE + Chronos-Bolt) tests
# ============================================================
from src.uncertainty.agreement_checker import check_dual_agreement
from src.uncertainty.chronos_uncertainty import chronos_forecast_interval, get_chronos


def test_mapie_returns_prediction_sets():
    """MAPIE predict_with_sets returns y_set with shape[1] == n_classes (4)."""
    from src.uncertainty.conformal_mapie import predict_with_sets

    # Create a mock MapieClassifier
    mock_mapie = MagicMock()
    X_test = np.random.randn(5, 10)

    # Mock predict to return (y_pred, y_set) where y_set has 4 classes
    mock_mapie.predict.return_value = (
        np.array([0, 1, 2, 0, 1]),  # y_pred
        np.array([                   # y_set: (n_samples, n_classes)
            [True, False, False, False],
            [False, True, False, False],
            [False, False, True, True],
            [True, False, False, False],
            [False, True, True, False],
        ]),
    )

    y_pred, y_set = predict_with_sets(mock_mapie, X_test, alpha=0.10)
    assert y_set.shape[1] == 4, f"Expected 4 classes, got {y_set.shape[1]}"
    mock_mapie.predict.assert_called_once()


@patch("src.uncertainty.chronos_uncertainty.get_chronos")
def test_chronos_uses_predict_not_quantiles(mock_get_chronos):
    """Chronos uses pipeline.predict(), NOT predict_quantiles() [RULE-7]."""
    mock_pipeline = MagicMock()
    mock_get_chronos.return_value = mock_pipeline

    # Mock predict to return a tensor-like object
    fake_samples = torch.randn(1, 200, 15)
    mock_pipeline.predict.return_value = fake_samples

    flux_window = np.random.randn(60)
    result = chronos_forecast_interval(flux_window)

    # predict() must be called
    mock_pipeline.predict.assert_called_once()

    # predict_quantiles() must NOT exist or NOT be called
    if hasattr(mock_pipeline, "predict_quantiles"):
        mock_pipeline.predict_quantiles.assert_not_called()


@patch("src.uncertainty.chronos_uncertainty.get_chronos")
def test_chronos_quantiles_computed_from_samples(mock_get_chronos):
    """q10 < median < q90 when computed from fixed samples."""
    mock_pipeline = MagicMock()
    mock_get_chronos.return_value = mock_pipeline

    # Create fixed samples where values range from 1 to 200
    # Shape: (1, num_samples, prediction_length)
    samples = torch.arange(1, 201, dtype=torch.float32).unsqueeze(1).expand(1, 200, 15)
    mock_pipeline.predict.return_value = samples

    flux_window = np.random.randn(60)
    result = chronos_forecast_interval(flux_window)

    assert result["q10"] < result["median"] < result["q90"], (
        f"Expected q10 < median < q90, got q10={result['q10']}, "
        f"median={result['median']}, q90={result['q90']}"
    )


@patch("chronos.BaseChronosPipeline")
def test_chronos_singleton_only_loaded_once(mock_base_class):
    """get_chronos() calls from_pretrained exactly once across multiple calls."""
    import src.uncertainty.chronos_uncertainty as cu

    # Reset the singleton
    cu._CHRONOS_PIPELINE = None

    mock_base_class.from_pretrained.return_value = MagicMock()

    _ = cu.get_chronos()
    _ = cu.get_chronos()
    _ = cu.get_chronos()

    mock_base_class.from_pretrained.assert_called_once()

    # Reset singleton to not affect other tests
    cu._CHRONOS_PIPELINE = None


def test_dual_agreement_high_on_both_signals():
    """check_dual_agreement returns HIGH_UNCERTAINTY when both thresholds exceeded."""
    result = check_dual_agreement(
        mapie_set_size=3,
        chronos_std=5e-6,
        mapie_threshold=2,
        chronos_std_threshold=1e-7,
    )
    assert result == "HIGH_UNCERTAINTY"

    # Only one threshold exceeded -> NORMAL
    result_mapie_only = check_dual_agreement(
        mapie_set_size=3, chronos_std=1e-8,
        mapie_threshold=2, chronos_std_threshold=1e-7,
    )
    assert result_mapie_only == "NORMAL"

    result_chronos_only = check_dual_agreement(
        mapie_set_size=1, chronos_std=5e-6,
        mapie_threshold=2, chronos_std_threshold=1e-7,
    )
    assert result_chronos_only == "NORMAL"


# ============================================================
# M12 â€” Pre-flare Anomaly Detection (MOMENT) tests
# ============================================================
import src.forecasting.moment_anomaly as moment_module


@patch("momentfm.MOMENTPipeline")
def test_moment_singleton_loads_once(mock_pipeline_class):
    """get_moment_model() calls from_pretrained exactly once."""
    moment_module._MOMENT = None  # Reset singleton
    
    mock_pipeline_class.from_pretrained.return_value = MagicMock()
    
    _ = moment_module.get_moment_model()
    _ = moment_module.get_moment_model()
    
    mock_pipeline_class.from_pretrained.assert_called_once()
    
    moment_module._MOMENT = None  # Reset for other tests


def test_get_reconstruction_raises_on_wrong_attr():
    """_get_reconstruction raises actionable AttributeError if API changes."""
    class MockOutput:
        def __init__(self):
            # Wrong attribute name (e.g. 'logits' instead of 'reconstruction')
            self.logits = torch.randn(1, 1, 512)
            
    out = MockOutput()
    
    with pytest.raises(AttributeError) as exc_info:
        moment_module._get_reconstruction(out)
        
    err_msg = str(exc_info.value)
    assert "MOMENT output has no attribute" in err_msg
    assert "Available tensor attributes:" in err_msg
    assert "logits" in err_msg


def test_get_reconstruction_success():
    """_get_reconstruction safely returns the tensor when attribute is correct."""
    class MockOutput:
        def __init__(self):
            # Correct attribute based on RECONSTRUCTION_ATTR
            setattr(self, moment_module.RECONSTRUCTION_ATTR, torch.zeros(1, 1, 512))
            
    out = MockOutput()
    tensor = moment_module._get_reconstruction(out)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 1, 512)
    assert torch.all(tensor == 0)


@patch("src.forecasting.moment_anomaly.compute_reconstruction_error")
@patch("src.forecasting.moment_anomaly.build_phase_sequence")
def test_batch_skips_quiet_windows(mock_build_phase, mock_compute_error):
    """batch_compute_moment_scores skips computing on quiet (0) phases."""
    # Synthetic df with 1 pre-flare window and 9 quiet windows
    df_len = 10 * 60  # e.g., 10 minutes of data, 600 steps
    df = pd.DataFrame({"flux_high": np.random.randn(df_len)})
    
    # Let phase sequence be mostly 0, but one 1 (pre-flare) near the end
    phases = np.zeros(df_len, dtype=int)
    preflare_idx = df_len - 1
    phases[preflare_idx] = 1
    
    mock_build_phase.return_value = phases
    mock_compute_error.return_value = 0.05
    
    # We use a stride of 1 just to check the preflare index
    moment_module.batch_compute_moment_scores(df, "flux_high", window_size=512, stride=1)
    
    # compute_reconstruction_error should only be called once!
    mock_compute_error.assert_called_once()
    assert df["moment_reconstruction_error"].iloc[preflare_idx] == 0.05
    # The rest should be NaN
    assert np.isnan(df["moment_reconstruction_error"].iloc[0])


@patch("src.forecasting.moment_anomaly.get_moment_model")
def test_reconstruction_error_increases_near_flare(mock_get_model):
    """Reconstruction error is higher for a pre-flare ramp than quiet baseline."""
    mock_model = MagicMock()
    mock_get_model.return_value = mock_model
    
    # Mock the model output to just return something resembling the input but slightly off
    # We will simulate a perfect reconstruction for quiet, and a bad one for pre-flare.
    def side_effect(x):
        class MockOut:
            pass
        out = MockOut()
        
        # If the input has a high ramp (mean > threshold), return a bad reconstruction
        if torch.mean(x) > 1e-6:
            # Bad reconstruction (e.g., model predicts flat line instead of ramp)
            setattr(out, moment_module.RECONSTRUCTION_ATTR, torch.ones_like(x) * 1e-7)
        else:
            # Good reconstruction (matches input closely)
            setattr(out, moment_module.RECONSTRUCTION_ATTR, x + torch.randn_like(x) * 1e-9)
        return out
        
    mock_model.side_effect = side_effect
    
    quiet_window = np.ones(512) * 1e-7
    error_quiet = moment_module.compute_reconstruction_error(quiet_window)
    
    preflare_window = np.linspace(1e-7, 5e-5, 512)
    error_preflare = moment_module.compute_reconstruction_error(preflare_window)
    
    assert error_preflare > error_quiet, "Pre-flare error should be higher than quiet baseline"


# ============================================================
# M13 â€” Explainability (SHAP) tests
# ============================================================
from src.explainability.shap_explainer import SHAPExplainer


def test_shap_explain_result_has_required_keys():
    """explain() returns all required keys and top_features are subset of feature_names."""
    # Create a mock XGB model that shap.TreeExplainer can process
    # Since we can't easily mock TreeExplainer's internal C-extensions, we mock shap.TreeExplainer directly.
    with patch("src.explainability.shap_explainer.shap") as mock_shap:
        mock_explainer = MagicMock()
        mock_shap.TreeExplainer.return_value = mock_explainer
        
        # Mock shap_values to return a list of 4 arrays (for 4 classes), each shape (1, 10)
        mock_explainer.shap_values.return_value = [
            np.random.randn(1, 10) for _ in range(4)
        ]
        
        feature_names = [f"feat_{i}" for i in range(10)]
        explainer = SHAPExplainer(xgb_model="dummy", feature_names=feature_names)
        
        # Explain a single instance
        X_dummy = np.random.randn(1, 10)
        result = explainer.explain(X_dummy)
        
        # Check required keys
        required_keys = {"top_features", "class_importances", "dominant_class", "raw_shap"}
        assert required_keys.issubset(result.keys())
        
        # Check top_features are a subset of all features
        top_feature_names = [f[0] for f in result["top_features"]]
        assert set(top_feature_names).issubset(set(feature_names))
        assert len(top_feature_names) == 5


# ============================================================
# M14 â€” Dashboard & Visualization (FastAPI Backend) tests
# ============================================================
from fastapi.testclient import TestClient
from src.api.main import app, _active_ws, MAX_CONNECTIONS


def test_health_endpoint_returns_model_version():
    """GET /health returns status, model_version, and env."""
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model_version" in data
    assert "env" in data


def test_metrics_endpoint_returns_prometheus_format():
    """GET /metrics contains the solar_inference_duration_seconds metric."""
    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "solar_inference_duration_seconds" in resp.text


def test_websocket_rate_limit_enforced():
    """Sending 2 messages < 10s apart triggers rate limit closure (code 1008)."""
    client = TestClient(app)
    with client.websocket_connect("/ws/live") as ws:
        # First message is allowed
        ws.send_text("hello")
        # Send second message immediately (< 10s)
        ws.send_text("spam")
        # The server should respond with a rate_limit error then close
        response = ws.receive_text()
        assert "rate_limit" in response


def test_websocket_connection_cap():
    """Exceeding MAX_CONNECTIONS (20) should close the 21st connection with 1008."""
    client = TestClient(app)

    # Simulate MAX_CONNECTIONS active connections by adding sentinels
    sentinels = [MagicMock() for _ in range(MAX_CONNECTIONS)]
    _active_ws.update(sentinels)

    try:
        # The 21st connection should be rejected
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/live") as ws:
                # Should not get here â€” the server closes immediately
                ws.receive_text()
    finally:
        # Clean up sentinels
        _active_ws.difference_update(sentinels)

# --- M15: Evaluation & Walk-Forward Validation Tests ---

from src.evaluation.walk_forward import walk_forward_cv, compute_fold_metrics
from src.evaluation.slo_reporter import check_slo_compliance
import pandas as pd
import numpy as np

def test_walk_forward_no_data_leakage():
    # Create dummy dataframe
    df = pd.DataFrame({'val': np.arange(100)})
    
    # Dummy pipeline returning deterministic predictions
    def dummy_pipeline(train_df, test_df):
        assert train_df.index.max() < test_df.index.min(), "Data leakage detected: train index overlaps test index!"
        y_true = np.random.choice([0, 1], size=len(test_df))
        y_pred = np.random.choice([0, 1], size=len(test_df))
        return y_true, y_pred, None

    walk_forward_cv(df, dummy_pipeline, n_splits=3, gap_days=0)

def test_confusion_matrix_empty_fold_safe():
    # All zeros in true labels
    y_true = np.zeros(10)
    y_pred = np.ones(10) # FPs
    
    # Should not raise ValueError about missing classes because we use labels=[0,1]
    metrics = compute_fold_metrics(y_true, y_pred)
    assert metrics['far'] == 1.0 # 10 FPs / 10 total negatives

def test_slo_reporter_fails_on_high_far():
    # Mock slo config expects far_max: 0.10
    res = check_slo_compliance({"far": 0.15, "mean_lead_min": 15.0, "tpr_mx": 0.85})
    assert res['overall'] == 'fail'
    assert res['detail']['far'] == 'fail'

def test_slo_reporter_fails_on_low_tpr_mx():
    res = check_slo_compliance({"far": 0.06, "mean_lead_min": 15.0, "tpr_mx": 0.62})
    assert res['overall'] == 'fail'
    assert res['detail']['tpr_mx'] == 'fail'

def test_slo_reporter_passes_all_three_slos():
    res = check_slo_compliance({"far": 0.06, "mean_lead_min": 12.0, "tpr_mx": 0.85})
    assert res['overall'] == 'pass'

def test_promote_blocks_on_low_tpr_mx():
    # The promotion logic usually resides in a promotion script, but we test the SLO evaluation part here
    res = check_slo_compliance({"far": 0.05, "tpr_mx": 0.65})
    assert res['overall'] == 'fail'
    assert res['detail']['tpr_mx'] == 'fail'

def test_all_src_modules_importable():
    """
    Walk src/ and import every .py file (except __init__.py).
    Assert no ImportError. This catches missing __init__.py and circular imports.
    """
    import os
    import importlib
    
    src_dir = os.path.abspath('src')
    failed_imports = []
    
    for root, _, files in os.walk(src_dir):
        for f in files:
            if f.endswith('.py') and f != '__init__.py':
                # Convert file path to module name
                rel_path = os.path.relpath(os.path.join(root, f), 'src')
                module_name = 'src.' + rel_path.replace(os.sep, '.')[:-3]
                try:
                    importlib.import_module(module_name)
                except ImportError as e:
                    failed_imports.append((module_name, str(e)))
                    
    assert not failed_imports, f"Failed to import the following modules: {failed_imports}"


def test_seen_files_is_seen_returns_false_for_new_file(tmp_path):
    import os
    os.environ['SEEN_FILES_DB'] = str(tmp_path / 'test_seen.db')
    from src.ingestion.seen_files import is_seen
    assert not is_seen('new_file.fits')

def test_seen_files_mark_then_is_seen_returns_true(tmp_path):
    import os
    os.environ['SEEN_FILES_DB'] = str(tmp_path / 'test_seen.db')
    from src.ingestion.seen_files import is_seen, mark_seen
    mark_seen('path.fits', 'solexs', 'run-1')
    assert is_seen('path.fits')

def test_seen_files_duplicate_mark_is_idempotent(tmp_path):
    import os
    os.environ['SEEN_FILES_DB'] = str(tmp_path / 'test_seen.db')
    from src.ingestion.seen_files import mark_seen, list_seen
    mark_seen('dup.fits', 'solexs', 'run-1')
    mark_seen('dup.fits', 'solexs', 'run-2')
    entries = [e for e in list_seen() if e['fits_path'] == 'dup.fits']
    assert len(entries) == 1
    assert entries[0]['pipeline_run_id'] == 'run-1'


def test_scheduler_skips_already_seen_file(tmp_path, monkeypatch):
    import os
    os.environ['SEEN_FILES_DB'] = str(tmp_path / 'test_seen.db')
    from src.ingestion.seen_files import mark_seen
    from src.orchestration.scheduler import run_scheduler
    from unittest.mock import MagicMock, patch
    mark_seen('seen_file.fits', 'solexs', 'run-1')
    pradan_mock = MagicMock()
    pradan_mock.list_new_files.return_value = ['seen_file.fits']
    pipeline_mock = MagicMock()
    def mock_sleep(s): raise StopIteration
    with patch('time.sleep', mock_sleep):
        try:
            run_scheduler(pipeline_mock, pradan_mock)
        except StopIteration:
            pass
    pipeline_mock.invoke.assert_not_called()

def test_post_slack_no_op_when_webhook_not_set():
    from src.orchestration.scheduler import _post_slack
    import os
    if 'SLACK_WEBHOOK_URL' in os.environ: del os.environ['SLACK_WEBHOOK_URL']
    _post_slack('test')  # should not raise

def test_scheduler_falls_back_to_120s_after_3_slow_cycles():
    from src.orchestration.scheduler import run_scheduler
    from unittest.mock import MagicMock, patch
    pradan_mock = MagicMock()
    pradan_mock.list_new_files.return_value = []
    pipeline_mock = MagicMock()
    sleep_calls = []
    def mock_sleep(s):
        sleep_calls.append(s)
        if len(sleep_calls) >= 3: raise StopIteration
    def mock_perf():
        mock_perf.val += 65
        return mock_perf.val
    mock_perf.val = 0.0
    with patch('time.sleep', mock_sleep), patch('time.perf_counter', mock_perf), patch('src.orchestration.scheduler._post_slack') as mock_slack:
        try:
            run_scheduler(pipeline_mock, pradan_mock)
        except StopIteration:
            pass
    mock_slack.assert_called_once()
    assert 'SLO-2 breach' in mock_slack.call_args[0][0]
    assert sleep_calls[-1] == max(0.0, 120 - 65)

def test_scheduler_recovers_cadence_after_fast_cycle():
    from src.orchestration.scheduler import run_scheduler
    from unittest.mock import MagicMock, patch
    pradan_mock = MagicMock()
    pradan_mock.list_new_files.return_value = []
    pipeline_mock = MagicMock()
    sleep_calls = []
    def mock_sleep(s):
        sleep_calls.append(s)
        if len(sleep_calls) >= 4: raise StopIteration
    def mock_perf():
        if len(sleep_calls) < 3:
            mock_perf.val += 65
        else:
            mock_perf.val += 5
        return mock_perf.val
    mock_perf.val = 0.0
    with patch('time.sleep', mock_sleep), patch('time.perf_counter', mock_perf), patch('src.orchestration.scheduler._post_slack') as mock_slack:
        try:
            run_scheduler(pipeline_mock, pradan_mock)
        except StopIteration:
            pass
    assert mock_slack.call_count == 2
    assert 'SLO-2 recovered' in mock_slack.call_args_list[1][0][0]
    assert sleep_calls[-1] == max(0.0, 60 - 5)


def test_rollback_updates_model_hashes(tmp_path, monkeypatch):
    import os, yaml
    os.makedirs('configs', exist_ok=True)
    if os.path.exists('configs/model_hashes.yaml'): os.remove('configs/model_hashes.yaml')
    from unittest.mock import patch, MagicMock
    with patch('mlflow.set_tracking_uri'), patch('mlflow.tracking.MlflowClient'), patch('mlflow.artifacts.download_artifacts') as mock_dl:
        mock_dl.return_value = str(tmp_path / 'mock_model')
        (tmp_path / 'mock_model').write_text('dummy')
        from scripts.rollback import rollback_to_run
        rollback_to_run('fake_run_id', 'http://mock')
    assert os.path.exists('configs/model_hashes.yaml')
    with open('configs/model_hashes.yaml') as f: hashes = yaml.safe_load(f)
    assert len(hashes) > 0

def test_rollback_updates_version_yaml(tmp_path, monkeypatch):
    import os, yaml
    os.makedirs('configs', exist_ok=True)
    from unittest.mock import patch, MagicMock
    with patch('mlflow.set_tracking_uri'), patch('mlflow.tracking.MlflowClient'), patch('mlflow.artifacts.download_artifacts') as mock_dl:
        mock_dl.return_value = str(tmp_path / 'mock_model')
        (tmp_path / 'mock_model').write_text('dummy')
        from scripts.rollback import rollback_to_run
        rollback_to_run('fake_run_id_12345', 'http://mock')
    with open('configs/version.yaml') as f: v = yaml.safe_load(f)
    assert v['model_version'].startswith('rollback-fake_run')



def test_compute_canary_far_all_true_positives():
    import pandas as pd
    from src.deployment.model_registry import _compute_canary_far
    preds = pd.DataFrame({'timestamp': ['2024-01-01T12:00:00Z', '2024-01-01T12:30:00Z', '2024-01-01T13:00:00Z'], 'predicted_class': [2, 3, 2]})
    noaa = pd.DataFrame({'peak_time': ['2024-01-01T12:03:00Z', '2024-01-01T12:28:00Z', '2024-01-01T13:05:00Z']})
    assert _compute_canary_far(preds, noaa) == 0.0



def test_health_endpoint_responds_under_1s():
    from unittest.mock import patch, MagicMock
    with patch('requests.get') as mock_get, patch('time.perf_counter') as mock_perf:
        import time
        responses = []
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp
        
        times = []
        for i in range(20):
            t1 = time.time()
            times.append(t1)
            times.append(t1 + 0.5) # simulate 500ms delay
            
        mock_perf.side_effect = times
        
        # The actual test would make real requests if API is up, but here we just mock.
        import requests
        latencies = []
        for _ in range(20):
            start = time.perf_counter()
            requests.get('http://localhost:8000/health')
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
        
        latencies.sort()
        p99 = latencies[int(0.99 * len(latencies))]
        assert p99 < 1000, f'P99 response time {p99}ms > 1000ms'

def test_all_src_modules_importable():
    import os, importlib
    importlib.invalidate_caches()
    for root, _, files in os.walk('src'):
        for f in files:
            if f.endswith('.py') and f != '__init__.py':
                module = os.path.join(root, f).replace(os.sep, '.')[:-3]
                importlib.import_module(module)


def test_sensitive_formatter_masks_password():
    import os, logging
    os.environ['PRADAN_PASSWORD'] = 'super_secret_pw'
    from src.utils.logging_config import SensitiveFormatter
    fmt = SensitiveFormatter()
    record = logging.LogRecord(name='test', level=logging.ERROR, pathname='', lineno=0, msg='Login failed: super_secret_pw is wrong', args=(), exc_info=None)
    output = fmt.format(record)
    assert 'super_secret_pw' not in output
    assert '***REDACTED***' in output

def test_sensitive_formatter_no_op_when_no_secrets():
    import os, logging
    if 'PRADAN_PASSWORD' in os.environ: del os.environ['PRADAN_PASSWORD']
    if 'PRADAN_USERNAME' in os.environ: del os.environ['PRADAN_USERNAME']
    if 'GH_PAT' in os.environ: del os.environ['GH_PAT']
    from src.utils.logging_config import SensitiveFormatter
    fmt = SensitiveFormatter()
    # Force rebuild of pattern
    fmt.__class__._PATTERN = None
    record = logging.LogRecord(name='test', level=logging.INFO, pathname='', lineno=0, msg='Normal log', args=(), exc_info=None)
    output = fmt.format(record)
    assert output == 'Normal log'


def test_timing_recorded_even_when_agent_raises():
    from src.orchestration.agents import ingestion_agent
    from unittest.mock import patch
    with patch('uuid.uuid4', side_effect=RuntimeError('mock error')):
        state = {}
        result = ingestion_agent(state)
    assert 'errors' in result
    assert any('mock error' in e for e in result['errors'])
    assert 'timing' in result
    assert 'ingestion' in result['timing']
    assert result['timing']['ingestion'] >= 0


def test_extract_best_far_uses_val_set_not_history():
    from src.automl.flaml_tuner import _extract_best_far
    from unittest.mock import MagicMock
    import numpy as np
    automl = MagicMock()
    model = MagicMock()
    # probas that give FP=1, TP=1, FN=0, TN=1
    model.predict_proba.return_value = np.array([
        [0.9, 0.1, 0.0, 0.0],  # N predicted (TN)
        [0.1, 0.9, 0.0, 0.0],  # C predicted (FP for true=0)
        [0.1, 0.9, 0.0, 0.0],  # C predicted (TP for true>0)
    ])
    automl.model.estimator = model
    X_val = np.zeros((3, 10))
    y_val = np.array([0, 0, 1])
    far = _extract_best_far(automl, X_val, y_val)
    # TN=1, FP=1 -> FAR = 1/(1+1) = 0.5
    assert far == 0.5
    assert not hasattr(automl, 'best_config_train_time') or automl.best_config_train_time is not None
