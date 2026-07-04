"""
Multi-class N/C/M/X labeller and sliding window generator.
"""

import pandas as pd
import numpy as np

CLASS_MAP     = {"N": 0, "B": 0, "C": 1, "M": 2, "X": 3}  # D10: B-class = no operational alert (see docs/decisions/D10.md)
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items() if k != "B"}


def build_multiclass_labels(df: pd.DataFrame,
                             master_catalogue: pd.DataFrame) -> pd.DataFrame:
    """
    Label each 1-min row with N/C/M/X using the master catalogue.
    Also writes convenience binary flags for per-class threshold evaluation.
    """
    df = df.copy()
    df["label"]       = 0
    df["label_name"]  = "N"

    for _, event in master_catalogue.iterrows():
        mask = (df.index >= event["start_time"]) & (df.index <= event["end_time"])
        cls  = str(event.get("flare_class", "N"))[0].upper()
        if cls in CLASS_MAP:
            df.loc[mask, "label"]      = CLASS_MAP[cls]
            df.loc[mask, "label_name"] = cls

    df["is_c_plus"] = (df["label"] >= 1).astype(int)
    df["is_m_plus"] = (df["label"] >= 2).astype(int)
    df["is_x"]      = (df["label"] == 3).astype(int)

    dist = df["label_name"].value_counts()
    print("Label distribution:", dist.to_dict())
    
    n_flare = (df["label"] > 0).sum()
    if n_flare < 50:
        print(
            f"WARNING: Only {n_flare} labeled flare rows. "
            "Consider increasing GOES supplementary weight or lowering detector sigma. "
            "See Decision D2."
        )
    return df


def create_windows(df: pd.DataFrame,
                   feature_cols: list,
                   window_size:  int = 60,
                   horizon:      int = 15,
                   step:         int = 1) -> tuple:
    """Sliding window generator returning (X, y_now, y_fore)."""
    X, y_now, y_fore = [], [], []
    
    # Filter for columns that actually exist
    available_cols = [c for c in feature_cols if c in df.columns]
    
    data   = df[available_cols].values
    labels = df["label"].values

    for i in range(0, len(data) - window_size - horizon + 1, step):
        X.append(data[i:i + window_size])
        y_now.append(labels[i + window_size - 1])
        y_fore.append(labels[i + window_size + horizon - 1])

    return (
        np.array(X,      dtype=np.float32),
        np.array(y_now,  dtype=np.int64),
        np.array(y_fore, dtype=np.int64),
    )
