import pandas as pd
import numpy as np

def verify_timesfm():
    try:
        import timesfm
        print("timesfm imported successfully.")
        
        try:
            import timesfm.finetuning
            print("timesfm.finetuning exists!")
        except ImportError:
            print("timesfm.finetuning does NOT exist (Expected).")
            
        tfm = timesfm.TimesFm(
            context_len=60,
            horizon_len=15,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=20,
            model_dims=1280,
            backend="cpu"
        )
        
        methods = [m for m in dir(tfm) if not m.startswith("_")]
        print(f"Public methods on TimesFm instance: {methods}")
        
        print("\nLoading weights to test forecast_on_df()...")
        tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")
        
        df = pd.DataFrame({
            "unique_id": ["T1"] * 60,
            "ds": pd.date_range("2024-01-01", periods=60, freq="min"),
            "y": np.random.rand(60)
        })
        
        forecast_df = tfm.forecast_on_df(df, freq="min", value_name="y")
        print("\nforecast_on_df output columns:")
        print(forecast_df.columns.tolist())
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    verify_timesfm()
