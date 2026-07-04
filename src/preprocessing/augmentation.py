"""
Minority class augmentation using tsaug.
"""

import numpy as np

try:
    from tsaug import TimeWarp, Drift, Quantize

    _augmenter = (
        TimeWarp(n_speed_change=3, max_speed_ratio=3.0) * 2
        + Drift(max_drift=0.1, n_drift_points=5)
        + Quantize(n_levels=20)
    )
    TSAUG_AVAILABLE = True
except ImportError:
    print("WARNING: tsaug not installed. Augmentation will be skipped.")
    TSAUG_AVAILABLE = False


def augment_minority(X_train: np.ndarray,
                     y_train: np.ndarray,
                     target_ratio: float = 0.30) -> tuple:
    """
    Augment flare windows (C/M/X) until they reach target_ratio of total dataset.
    Only operates on minority windows — quiet-sun windows are NOT augmented.
    """
    if not TSAUG_AVAILABLE:
        return X_train, y_train
        
    X_flare = X_train[y_train > 0]
    n_needed = int(len(X_train) * target_ratio) - len(X_flare)
    
    if n_needed <= 0 or len(X_flare) == 0:
        return X_train, y_train

    reps        = (n_needed // max(len(X_flare), 1)) + 1
    y_flare     = y_train[y_train > 0]
    
    try:
        X_aug_pool  = np.vstack([_augmenter.augment(X_flare) for _ in range(reps)])[:n_needed]
        y_aug_pool  = np.tile(y_flare, reps + 1)[:n_needed]
        return np.vstack([X_train, X_aug_pool]), np.hstack([y_train, y_aug_pool])
    except Exception as e:
        print(f"Augmentation failed: {e}. Returning unaugmented dataset.")
        return X_train, y_train
