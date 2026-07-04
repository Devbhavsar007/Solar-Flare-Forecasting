"""
Conformal prediction wrapper using MAPIE.

Provides distribution-free 90% prediction sets for multi-class
solar flare classification. The conformal guarantee holds regardless
of the base model, as long as calibration data is exchangeable.
"""
import joblib
import os
import numpy as np

try:
    from mapie.classification import MapieClassifier
except ImportError:
    MapieClassifier = None


def train_mapie(X_cal: np.ndarray, y_cal: np.ndarray, base_estimator) -> "MapieClassifier":
    """
    Fit a MAPIE conformal classifier on a calibration set.

    Args:
        X_cal: Calibration features (n_samples, n_features).
        y_cal: Calibration labels (n_samples,).
        base_estimator: A pre-fitted sklearn-compatible classifier.

    Returns:
        Fitted MapieClassifier.
    """
    if MapieClassifier is None:
        raise ImportError("mapie is not installed. Run: pip install mapie")

    mapie = MapieClassifier(
        estimator=base_estimator,
        method="score",
        cv="prefit",
    )
    mapie.fit(X_cal, y_cal)

    # Save model [RULE-13]
    os.makedirs("models", exist_ok=True)
    joblib.dump(mapie, "models/conformal_mapie.pkl")

    return mapie


def load_mapie() -> "MapieClassifier":
    """
    Load a previously trained MAPIE conformal classifier.

    Returns:
        MapieClassifier loaded from models/conformal_mapie.pkl.
    """
    return joblib.load("models/conformal_mapie.pkl")


def predict_with_sets(
    mapie: "MapieClassifier",
    X: np.ndarray,
    alpha: float = 0.10,
) -> tuple:
    """
    Predict with conformal prediction sets at (1 - alpha) coverage.

    Args:
        mapie: Fitted MapieClassifier.
        X: Input features (n_samples, n_features).
        alpha: Significance level (default 0.10 = 90% coverage).

    Returns:
        Tuple of (y_pred, y_set):
          - y_pred: Point predictions (n_samples,).
          - y_set: Boolean array (n_samples, n_classes) indicating
                   which classes are in the conformal prediction set.
    """
    y_pred, y_set = mapie.predict(
        X, alpha=alpha, include_last_label="randomized"
    )
    return y_pred, y_set
