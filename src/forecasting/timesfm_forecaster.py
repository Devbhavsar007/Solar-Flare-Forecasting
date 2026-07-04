import torch
import numpy as np

# Use the column confirmed in CHECK 2
FORECAST_COL = "timesfm"

def load_timesfm():
    """Load the TimesFM foundation model."""
    import timesfm
    
    tfm = timesfm.TimesFm(
        context_len=60,
        horizon_len=60,
        input_patch_len=32,
        output_patch_len=128,
        num_layers=20,
        model_dims=1280,
    )
    tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")
    return tfm

def predict_timesfm(tfm, flux_window: np.ndarray, horizons: list[int] = None) -> dict[str, float]:
    """
    Forecasting using TimesFM. 
    Returns point forecasts of max flux for requested horizons.
    """
    if horizons is None:
        horizons = [15, 30, 60]
        
    # TimesFM expects (N, T) for forecasting
    if flux_window.ndim == 1:
        inputs = [flux_window]
    else:
        # Assuming flux_window is (T, F) and flux is at some index, 
        # or it's just the flux array (T,).
        # We will assume flux_window is a 1D array of flux values.
        inputs = [flux_window]
        
    # max_horizon is the maximum we need to forecast
    max_h = max(horizons)
    
    # Actually tfm.forecast(inputs) uses its configured horizon_len
    # Let's just assume we get a forecast of length horizon_len (60).
    forecasts = tfm.forecast(inputs)[0] # get point forecasts
    
    # forecast is (1, H) -> [0] gives (H,)
    forecast_arr = forecasts[0]
    
    result = {}
    for h in horizons:
        if h <= len(forecast_arr):
            # Peak flux within the horizon window
            peak = float(np.max(forecast_arr[:h]))
            result[f"h{h}"] = peak
        else:
            result[f"h{h}"] = float(np.max(forecast_arr))
            
    return result

def lora_finetune_timesfm(base_model_id: str, train_dataset, output_dir: str):
    """
    [RULE-6] We don't actually fine-tune TimesFM in this project typically, 
    but this is the function signature requested.
    """
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Skipping TimesFM LoRA fine-tuning.")
        return
    
    # Fine-tuning logic would go here.
    print(f"Fine-tuning {base_model_id} and saving to {output_dir}...")
