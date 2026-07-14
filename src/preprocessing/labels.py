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
                   horizon:      int | list[int] = 15,
                   step:         int = 1) -> tuple:
    """Sliding window generator returning (X, y_now, y_fore, dates).

    Downsamples N-class (label=0) windows by ``step`` while keeping
    100 % of C / M / X windows (label >= 1).  This prevents the
    uniform-step approach from discarding the vast majority of rare
    flare events.

    Returns:
        X:      (N, window_size, n_features) float32 array of input windows.
        y_now:  (N,) int64 array of labels at the decision point.
        y_fore: (N,) int64 array (if horizon is int) or dict of arrays (if list).
        dates:  pandas DatetimeIndex of length N — the decision-point timestamp.
    """
    if isinstance(horizon, int):
        horizons = [horizon]
        single_horizon = True
    else:
        horizons = horizon
        single_horizon = False

    X, y_now, date_list = [], [], []
    y_fore_dict = {h: [] for h in horizons}

    available_cols = [c for c in feature_cols if c in df.columns]

    data   = df[available_cols].values
    labels = df["label"].values
    index  = df.index  # keep reference for date extraction

    max_horizon = max(horizons)
    n_total = len(data) - window_size - max_horizon + 1
    
    if n_total <= 0:
        y_fore_ret = np.empty((0,), dtype=np.int64) if single_horizon else {f"h{h}": np.empty((0,), dtype=np.int64) for h in horizons}
        return (
            np.empty((0, window_size, len(available_cols)), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            y_fore_ret,
            pd.DatetimeIndex([], dtype="datetime64[ns, UTC]"),
        )

    for i in range(n_total):
        lbl_now  = labels[i + window_size - 1]
        
        # Check if ANY of the horizons have a flare
        is_flare_window = (lbl_now > 0)
        lbls_fore = {}
        for h in horizons:
            lbl_h = labels[i + window_size + h - 1]
            lbls_fore[h] = lbl_h
            if lbl_h > 0:
                is_flare_window = True

        if is_flare_window or (i % step == 0):
            X.append(data[i:i + window_size])
            y_now.append(lbl_now)
            for h in horizons:
                y_fore_dict[h].append(lbls_fore[h])
            date_list.append(index[i + window_size - 1])

    y_fore_res = {f"h{h}": np.array(y_fore_dict[h], dtype=np.int64) for h in horizons}
    if single_horizon:
        y_fore_res = y_fore_res[f"h{horizons[0]}"]

    return (
        np.array(X,      dtype=np.float32),
        np.array(y_now,  dtype=np.int64),
        y_fore_res,
        pd.DatetimeIndex(date_list),
    )

