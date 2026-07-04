"""
Shadow Inference Runner (Canary mode).
Reads same FITS files as prod, runs the new model, and logs predictions
WITHOUT triggering any alerts.
"""

import os
import argparse
import pandas as pd
from pathlib import Path
from src.ingestion.fits_reader import read_solexs

def run_shadow(model_dir: str, output_csv: str):
    # Dummy implementation since we don't have the full model loading here
    # In a real scenario, this would load models from model_dir and predict
    print(f"[SHADOW] Initializing shadow runner with models from {model_dir}")
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    
    # Write a dummy prediction for testing
    df = pd.DataFrame({
        "timestamp": [pd.Timestamp.utcnow()],
        "predicted_class": ["N"],
        "confidence": [0.99]
    })
    df.to_csv(output_csv, index=False)
    print(f"[SHADOW] Wrote predictions to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_shadow(args.model_dir, args.output)
