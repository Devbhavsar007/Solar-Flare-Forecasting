import os
import tempfile
import torch
import numpy as np
import xgboost as xgb
from hypothesis import given, settings, strategies as st
from src.deployment.onnx_export import export_to_onnx, ONNXNowcaster
from src.nowcasting.tcn_encoder import TCNEncoder

# ====================================================================
# [RULE-14] Singleton pattern for Hypothesis tests that need heavy models
# ====================================================================
_TOY_NOWCASTER = None

def _get_toy_nowcaster():
    global _TOY_NOWCASTER
    if _TOY_NOWCASTER is None:
        # Create temporary directory for ONNX model
        tmp_dir = tempfile.mkdtemp()
        onnx_path = os.path.join(tmp_dir, "toy_tcn.onnx")
        
        # 1. Create and export toy TCN
        encoder = TCNEncoder(n_features=8, embed_dim=32, n_layers=2)
        dummy_input = torch.randn(1, 60, 8)
        export_to_onnx(encoder, dummy_input, onnx_path)
        
        # 2. Create dummy XGBoost model
        X_dummy = np.random.randn(10, 32 + 60*8)  # TCN (32) + flat (480)
        y_dummy = np.random.randint(0, 4, 10)
        
        xgb_model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=4,
            n_estimators=2,
            max_depth=2,
        )
        xgb_model.fit(X_dummy, y_dummy)
        
        # 3. Instantiate the ONNX Nowcaster
        _TOY_NOWCASTER = ONNXNowcaster(onnx_path, xgb_model)
        
    return _TOY_NOWCASTER


import hypothesis.extra.numpy as npst

@settings(max_examples=20, deadline=None)
@given(npst.arrays(np.float32, shape=(60, 8), elements=st.floats(0.0, 1e-3, allow_nan=False)))
def test_onnx_nowcaster_output_valid(window):
    """
    Test ONNX Nowcaster with Hypothesis.
    Uses singleton to avoid creating the model per example.
    """
    nowcaster = _get_toy_nowcaster()
    
    # Run prediction
    result = nowcaster.predict(
        window[np.newaxis].astype(np.float32), 
        np.zeros((1, 0), dtype=np.float32)
    )
    
    assert result["class"] in ("N", "C", "M", "X")
    assert 0.0 <= result["confidence"] <= 1.0
    assert abs(sum(result["proba"].values()) - 1.0) < 1e-4
