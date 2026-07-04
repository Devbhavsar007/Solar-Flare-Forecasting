import numpy as np
from typing import List, Dict, Tuple, Any
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import confusion_matrix
import pandas as pd

def compute_fold_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, float]:
    """
    Computes classification metrics for a single fold.
    Labels should be 0 and 1.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    # TSS calculation with 1e-9 to prevent division by zero
    tss = (tp / (tp + fn + 1e-9)) - (fp / (fp + tn + 1e-9))
    
    # Other metrics
    far = fp / max(fp + tn, 1)
    ppv = tp / max(tp + fp, 1)
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    
    per_class_f1 = (2 * tp) / max(2 * tp + fp + fn, 1)
    
    return {
        "tss": tss,
        "far": far,
        "ppv": ppv,
        "tpr": tpr,
        "tnr": tnr,
        "per_class_f1": per_class_f1
    }

def walk_forward_cv(df: pd.DataFrame, pipeline_fn: callable, n_splits: int = 5, gap_days: int = 7) -> Dict[str, Tuple[float, float]]:
    """
    Performs walk-forward cross-validation.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap_days * 1440)
    
    metrics_list = {
        "tss": [], "far": [], "ppv": [], "tpr": [], "tnr": [], "per_class_f1": []
    }
    
    for train_index, test_index in tscv.split(df):
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]
        
        # Determine actual target column names. Assuming 'target' or passed via pipeline.
        # This will depend on implementation details, passing train/test data to pipeline_fn
        # which returns true labels, predictions, and optionally probabilities.
        y_true, y_pred, y_proba = pipeline_fn(train_df, test_df)
        
        fold_metrics = compute_fold_metrics(y_true, y_pred, y_proba)
        
        for k, v in fold_metrics.items():
            metrics_list[k].append(v)
            
    # Compute mean and std for all metrics
    aggregated = {}
    for k, v in metrics_list.items():
        aggregated[f"mean_{k}"] = np.mean(v)
        aggregated[f"std_{k}"] = np.std(v)
        
    return aggregated

def compute_lead_time(forecast_proba_series: pd.Series, actual_flare_times: pd.Series, threshold: float = 0.50) -> List[float]:
    """
    Computes lead times between first model trigger (prob >= threshold) and actual peak flare time.
    """
    lead_times = []
    
    for _, actual_time in actual_flare_times.items():
        if pd.isna(actual_time):
            continue
            
        # Find all predictions before the actual peak time
        past_forecasts = forecast_proba_series[forecast_proba_series.index <= actual_time]
        
        # Find the first timestep where probability exceeds the threshold
        triggers = past_forecasts[past_forecasts >= threshold]
        
        if not triggers.empty:
            first_trigger_time = triggers.index[0]
            # Calculate lead time in minutes
            lead_time_td = actual_time - first_trigger_time
            lead_times.append(lead_time_td.total_seconds() / 60.0)
        else:
            lead_times.append(np.nan)
            
    return lead_times


def export_eval_metrics(aggregated: Dict[str, Any], lead_times: List[float] = None, output_dir: str = None) -> None:
    """
    Dump walk-forward evaluation metrics to data/processed/eval_metrics.json
    so the /evaluation API endpoint and Analytics dashboard page can consume them.
    """
    import json
    import os

    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "processed"
        )
    os.makedirs(output_dir, exist_ok=True)

    mean_lead = float(np.nanmean(lead_times)) if lead_times and len(lead_times) > 0 else None

    payload = {
        "tss": float(aggregated.get("mean_tss", 0)),
        "far": float(aggregated.get("mean_far", 0)),
        "tpr": float(aggregated.get("mean_tpr", 0)),
        "ppv": float(aggregated.get("mean_ppv", 0)),
        "tnr": float(aggregated.get("mean_tnr", 0)),
        "mean_lead_min": mean_lead,
    }

    out_path = os.path.join(output_dir, "eval_metrics.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
