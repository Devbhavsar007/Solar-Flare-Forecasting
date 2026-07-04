"""
FLAML AutoML tuning for XGBoost nowcasting model (M8).

Implements hyperparameter search optimizing for TSS [RULE-3].
Includes a hard False Alarm Rate (FAR) gate [SLO-5] that rejects any
model with FAR > 0.10 regardless of TSS improvement.
"""
import numpy as np
import random, torch
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

from sklearn.metrics import confusion_matrix
import mlflow

try:
    from flaml import AutoML, tune
except ImportError:
    AutoML = None
    tune = None

from src.monitoring.metrics import FAR_GAUGE


def custom_metric(
    X_val, y_val, estimator, labels, X_train, y_train,
    weight_val=None, weight_train=None, config=None,
    groups_val=None, groups_train=None,
):
    """
    Custom metric for FLAML optimizing True Skill Statistic (TSS).

    Returns a tuple (loss, metric_dict). FLAML minimizes loss, so we
    return 1.0 - TSS. Also returns FAR for the promotion gate.
    """
    # XGBoost multi:softprob output
    proba = estimator.predict_proba(X_val)

    # Binarize: classes 1, 2, 3 (C, M, X) sum to >= 0.5
    binary_pred = (proba[:, 1:].sum(axis=1) >= 0.5).astype(int)
    binary_true = (y_val > 0).astype(int)

    # RULE-8: NEVER use confusion_matrix() without labels=[0, 1].
    # Walk-forward CV folds may have no positive events.
    tn, fp, fn, tp = confusion_matrix(
        binary_true, binary_pred, labels=[0, 1]
    ).ravel()

    # Avoid division by zero
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)

    tss = tpr - fpr
    far = fpr  # False Alarm Rate = FPR

    # FLAML minimizes the primary metric, so return 1.0 - TSS
    loss = 1.0 - tss
    metrics = {"tss": tss, "far": far, "tpr": tpr, "fpr": fpr}

    return loss, metrics


def _extract_best_far(automl, X_val, y_val) -> float:
    """
    Compute FAR for the best FLAML model by re-evaluating on val set.
    """
    model = automl.model.estimator
    proba = model.predict_proba(X_val)
    binary_pred = (proba[:, 1:].sum(axis=1) >= 0.5).astype(int)
    binary_true = (y_val > 0).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        binary_true, binary_pred, labels=[0, 1]   # [RULE-8]
    ).ravel()
    return float(fp / max(fp + tn, 1))

def run_flaml_automl(X_tr, y_tr, X_val, y_val, time_budget=300):
    """
    Run FLAML AutoML for XGBoost, minimizing 1 - TSS.

    Applies the [SLO-5] hard block if FAR > 0.10.

    Returns:
        tuple: (best_estimator or None, best_far, best_tss)
    """
    if AutoML is None:
        raise ImportError("flaml is not installed. Run pip install flaml")

    automl = AutoML()
    
    # Search space for XGBoost
    custom_hp = {
        "xgboost": {
            "n_estimators": {"domain": tune.randint(100, 800), "init_value": 300},
            "max_depth": {"domain": tune.randint(3, 9), "init_value": 6},
            "learning_rate": {"domain": tune.loguniform(1e-3, 0.3)},
            "subsample": {"domain": tune.uniform(0.5, 1.0)},
            "colsample_bytree": {"domain": tune.uniform(0.5, 1.0)},
        }
    }

    automl.fit(
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_val,
        y_val=y_val,
        metric=custom_metric,
        task="classification",
        estimator_list=["xgboost"],
        time_budget=time_budget,
        log_file_name="logs/flaml_automl.log",
        custom_hp=custom_hp,
        verbose=0,
    )

    best_tss = -automl.best_loss if automl.best_loss is not None else 0.0

    best_far = _extract_best_far(automl, X_val, y_val)

    # Update Prometheus metric
    FAR_GAUGE.set(best_far)

    # [SLO-5] HARD BLOCK: FAR > 0.10
    if best_far > 0.10:
        mlflow.log_params({
            "model_rejected": "FAR_exceeds_threshold",
            "model_far": round(best_far, 4),
            "model_far_limit": 0.10,
            "model_tss": round(best_tss, 4),
        })
        print(
            f"[SLO-5] MODEL REJECTED: FAR={best_far:.4f} > 0.10 limit.\n"
            f"  TSS={best_tss:.4f} is irrelevant — FAR gate blocks promotion.\n"
            f"  Action: Tune class thresholds, increase sigma_threshold in\n"
            f"  detectors, or reduce augmentation ratio and re-run M8."
        )
        return None, best_far, best_tss

    # Successful promotion
    mlflow.log_params({
        "model_far": round(best_far, 4),
        "model_tss": round(best_tss, 4),
        "model_rejected": "no",
    })
    
    return automl.model.estimator, best_far, best_tss
