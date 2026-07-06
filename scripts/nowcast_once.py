"""
Cron Inference Worker (scripts/nowcast_once.py)

Runs every 15 minutes via GitHub Actions.
Loads trained models, fetches recent telemetry, runs inference,
and securely POSTs the result to the Render dashboard API.
"""

import os
import json
import requests
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

def fetch_telemetry():
    """
    Fetch the latest 60-minute window of real telemetry data.
    Returns None if no data is available.
    """
    print("Fetching latest telemetry...")
    # In a real environment, this might call:
    # from src.ingestion.goes_downloader import download_goes_xrs
    # df = download_goes_xrs(start=now - 2 hours, end=now)
    # 
    # Since we can't reliably fetch live GOES data in this CI script without full NOAA API access,
    # we will look for a local recent file or fail gracefully.
    
    # Try to simulate fetching real data
    try:
        from sunpy.net import Fido, attrs as a
        from sunpy.timeseries import TimeSeries
        
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=6)
        start_str = start.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end.strftime("%Y-%m-%d %H:%M:%S")
        
        result = Fido.search(
            a.Time(start_str, end_str),
            a.Instrument.xrs,
            a.Resolution("avg1m"),
            a.goes.SatelliteNumber(16)
        )
        if len(result) > 0:
            downloaded = Fido.fetch(result)
            if downloaded:
                ts = TimeSeries(downloaded, concatenate=True)
                df = ts.to_dataframe()
                df = df.dropna(subset=['xrsa', 'xrsb'], how='all')
                df = df.resample('1min').mean()
                
                # Get the most recent complete 60-minute chunk
                if len(df) >= 60:
                    df = df.tail(60)
                    
                    # Load calibration
                    calib = {"slope": 1.0, "intercept": 0.0}
                    try:
                        import yaml
                        with open("configs/nowcasting.yaml", "r") as f:
                            cfg = yaml.safe_load(f) or {}
                            if "goes_calibration" in cfg:
                                calib = cfg["goes_calibration"]
                    except FileNotFoundError:
                        pass
                        
                    from src.preprocessing.cross_calibration import apply_goes_calibration
                    arr = np.zeros((1, 60, 4))
                    arr[0, :, 0] = df['xrsa'].values # A uses raw as calibrated
                    arr[0, :, 1] = apply_goes_calibration(df['xrsb'].values, calib)
                    arr[0, :, 2] = df['xrsa'].values
                    arr[0, :, 3] = df['xrsb'].values
                    return arr
    except Exception as e:
        print(f"Failed to fetch live telemetry: {e}")
        pass

    # If we couldn't fetch real data, we MUST return None to prevent fabricating alerts.
    return None

def run_inference(model, X_window):
    """
    Run the multi-class nowcaster and compute SHAP values.
    """
    # X_window is (1, 60, 4)
    # TCN feature is untrained and dropped to prevent feeding noise.
    tcn_feats = np.empty((1, 0))
        
    flat_window = X_window.reshape(len(X_window), -1)
    combined = np.concatenate([tcn_feats, flat_window], axis=1)
    
    proba = model.predict_proba(combined)[0]
    
    # Read thresholds from config if available
    import yaml
    thresholds = {"C": 0.38, "M": 0.45, "X": 0.28}
    try:
        with open("configs/nowcasting.yaml") as f:
            cfg = yaml.safe_load(f) or {}
            if "class_thresholds" in cfg:
                thresholds = cfg["class_thresholds"]
    except FileNotFoundError:
        pass
        
    pred_class = "N"
    if proba[3] >= thresholds.get("X", 0.5):
        pred_class = "X"
    elif proba[2] >= thresholds.get("M", 0.5):
        pred_class = "M"
    elif proba[1] >= thresholds.get("C", 0.5):
        pred_class = "C"
        
    # Real SHAP
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(combined)
        # Simplify SHAP for payload
        shap_payload = {
            "flux_recent": float(np.mean(shap_vals[0][:, -4:])),
        }
    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        shap_payload = {}
        
    classes = ["N", "C", "M", "X"]
    
    return {
        "flare_class": pred_class,
        "probabilities": {c: float(p) for c, p in zip(classes, proba)},
        "shap": shap_payload,
        "ts": datetime.now(timezone.utc).isoformat()
    }

def main():
    api_url = os.environ.get("RENDER_API_URL", "http://localhost:8000")
    api_key = os.environ.get("X_NOWCAST_KEY", "")
    
    print("Loading models...")
    try:
        model = joblib.load("models/xgb_multiclass.pkl")
    except Exception as e:
        print(f"Error loading models: {e}. Exiting.")
        return

    telemetry = fetch_telemetry()
    
    if telemetry is None:
        # [CRITICAL REQUIREMENT] If telemetry is empty, do NOT mock a window.
        # Skip the POST entirely. The frontend's timestamp-based staleness tracker
        # will naturally show the STALE banner without fabricating data.
        print("Telemetry is empty. Skipping POST to avoid fabricating alerts.")
        return
        
    print("Running inference...")
    result_payload = run_inference(model, telemetry)
    
    if result_payload is None:
        print("Inference failed. Skipping POST.")
        return
        
    print(f"POSTing result to {api_url}/alert...")
    headers = {
        "Content-Type": "application/json",
        "X-Nowcast-Key": api_key
    }
    
    try:
        response = requests.post(f"{api_url}/alert", json=result_payload, headers=headers)
        response.raise_for_status()
        print("Successfully POSTed alert.")
    except Exception as e:
        print(f"Failed to POST alert: {e}")

if __name__ == "__main__":
    main()
