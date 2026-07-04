import pytest
import pandas as pd
import numpy as np
import datetime
from src.nowcasting.solexs_detector import FlareEvent


@pytest.fixture
def sample_solexs_df():
    """
    Simulates 120 minutes of SoLEXS data.
    Background flux is ~1e-7 (B class).
    An M-class flare (peak 5e-5) occurs between t=55 and t=75.
    Peak is at t=65.
    """
    times = pd.date_range("2024-02-22T10:00:00", periods=120, freq="1min")
    flux = np.full(120, 1e-7)
    
    # Add noise to background
    flux += np.random.normal(0, 1e-9, 120)
    
    # Flare profile: rises from 55 to 65, decays from 65 to 75
    # Peak flux = 5e-5
    for i in range(55, 66):
        flux[i] = 1e-7 + (5e-5 - 1e-7) * ((i - 55) / 10.0)
    for i in range(66, 76):
        flux[i] = 1e-7 + (5e-5 - 1e-7) * ((75 - i) / 9.0)
        
    df = pd.DataFrame({"flux_high": flux}, index=times)
    return df


@pytest.fixture
def sample_hel1os_df():
    """
    Simulates 120 minutes of HEL1OS data.
    Background counts is ~50.
    A burst occurs between t=55 and t=70.
    Peak is at t=63 (which is 2 minutes before SoLEXS peak at t=65).
    This satisfies both the Neupert effect (-1 to -4 min) and the
    ±2 min coincidence window for the merger.
    """
    times = pd.date_range("2024-02-22T10:00:00", periods=120, freq="1min")
    counts = np.full(120, 50.0)
    
    # Add Poisson-like noise to background
    counts += np.random.normal(0, 5, 120)
    
    # Burst profile: rises from 55 to 63, decays from 63 to 70
    # Peak counts = 2800
    for i in range(55, 64):
        counts[i] = 50 + (2800 - 50) * ((i - 55) / 8.0)
    for i in range(64, 71):
        counts[i] = 50 + (2800 - 50) * ((70 - i) / 6.0)
        
    df = pd.DataFrame({"counts_low": counts}, index=times)
    return df


@pytest.fixture
def sample_flare_event():
    return FlareEvent(
        start_time=pd.to_datetime("2024-02-22T10:55:00"),
        peak_time=pd.to_datetime("2024-02-22T11:05:00"),  # t=65
        end_time=pd.to_datetime("2024-02-22T11:15:00"),
        peak_flux=5e-5,
        flare_class="M",
        instrument="SoLEXS",
        confidence=1.0,
        fits_path="/path/to/solexs.fits"
    )


@pytest.fixture
def sample_hel1os_event():
    return FlareEvent(
        start_time=pd.to_datetime("2024-02-22T10:55:00"),
        peak_time=pd.to_datetime("2024-02-22T11:03:00"),  # t=63
        end_time=pd.to_datetime("2024-02-22T11:10:00"),
        peak_flux=2800.0,
        flare_class="?",
        instrument="HEL1OS",
        confidence=1.0,
        fits_path="/path/to/hel1os.fits"
    )


# ============================================================
# M4 — TCN Encoder fixtures
# ============================================================
from src.nowcasting.tcn_encoder import TCNEncoder


@pytest.fixture
def toy_tcn_encoder():
    """Small TCN encoder for unit tests (n_features=8, embed_dim=32, 2 layers)."""
    return TCNEncoder(n_features=8, embed_dim=32, n_layers=2)


# ============================================================
# M6 — ONNX Export fixtures
# ============================================================
import os
from src.deployment.onnx_export import export_to_onnx, ONNXNowcaster
import xgboost as xgb

@pytest.fixture(scope="function")
def toy_onnx_nowcaster(tmp_path, toy_tcn_encoder):
    """
    [RULE-14] Must be function-scoped to avoid Hypothesis session incompatibility.
    """
    onnx_path = str(tmp_path / "toy_tcn.onnx")
    dummy_input = torch.randn(1, 60, 8)
    
    export_to_onnx(toy_tcn_encoder, dummy_input, onnx_path)
    
    # Create a dummy XGBoost model
    X_dummy = np.random.randn(10, 32 + 60*8)  # TCN output (32) + flat window (480)
    y_dummy = np.random.randint(0, 4, 10)
    
    xgb_model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=4,
        n_estimators=2,
        max_depth=2,
    )
    xgb_model.fit(X_dummy, y_dummy)
    
    return ONNXNowcaster(onnx_path, xgb_model)
