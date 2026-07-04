import time
import torch
import numpy as np

def verify_chronos():
    try:
        from chronos import BaseChronosPipeline
        
        print("Loading chronos-bolt-small...")
        t0 = time.time()
        pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-small",
            device_map="cpu",
            torch_dtype=torch.float32
        )
        t1 = time.time()
        print(f"Chronos load time: {t1 - t0:.2f}s")
        
        context = torch.rand(1, 60)
        print("Calling predict()...")
        forecast = pipeline.predict(
            context,
            prediction_length=15
        )
        print(f"Output shape: {forecast.shape}")  # (batch, num_quantiles, prediction_length) — already quantiles, not samples

        quantiles, mean = pipeline.predict_quantiles(
            context,
            prediction_length=15,
            quantile_levels=[0.1, 0.5, 0.9]
        )
        print(f"Quantiles shape: {quantiles.shape}, Mean shape: {mean.shape}")
        print("Quantiles retrieved successfully via predict_quantiles.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    verify_chronos()
