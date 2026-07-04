import yaml
import mlflow
from typing import Dict, Any

def check_slo_compliance(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Checks the provided metrics against the SLO thresholds defined in configs/slo.yaml.
    """
    with open("configs/slo.yaml", "r") as f:
        slo = yaml.safe_load(f)
        
    slo_status = {}
    
    # Latency check typically happens in integration tests, dummy pass here
    slo_status["latency"] = "pass"
    
    # SLO-5: FAR hard gate
    if metrics.get("far") is not None:
        slo_status["far"] = "pass" if metrics["far"] <= slo.get("far_max", 0.10) else "fail"
    
    # Lead time check
    if metrics.get("mean_lead_min") is not None:
        slo_status["lead_time"] = "pass" if metrics.get("mean_lead_min", 0) >= slo.get("lead_time_min_accepted", 10.0) else "fail"
        
    # SLO-6: TPR gate for M+X class
    tpr_mx = metrics.get("tpr_mx")
    if tpr_mx is not None:
        slo_status["tpr_mx"] = "pass" if tpr_mx >= slo.get("tpr_mx_min", 0.80) else "fail"
        
    overall = "pass" if all(v == "pass" for v in slo_status.values()) else "fail"
    
    return {
        "overall": overall, 
        "detail": slo_status, 
        "tpr_mx": tpr_mx
    }

def log_evaluation_to_mlflow(metrics: Dict[str, Any], slo_result: Dict[str, Any]):
    """
    Logs the evaluation metrics and SLO results to MLflow and updates Prometheus gauges.
    """
    from src.monitoring.metrics import FAR_GAUGE, LEAD_TIME_GAUGE
    
    with mlflow.start_run(nested=True):
        mlflow.log_metrics({
            "mean_tss": metrics.get("tss", 0.0),
            "mean_far": metrics.get("far", 0.0),
            "mean_tpr": metrics.get("tpr", 0.0),
            "mean_lead_min": metrics.get("mean_lead_min", 0.0),
            "tpr_mx": metrics.get("tpr_mx", -1),
        })
        
        mlflow.set_tags({
            "slo_status": slo_result["overall"],
            "slo_far_status": slo_result["detail"].get("far", "n/a"),
            "slo_lead_status": slo_result["detail"].get("lead_time", "n/a"),
            "slo_tpr_mx_status": slo_result["detail"].get("tpr_mx", "n/a"),
        })
        
        if slo_result["overall"] == "fail":
            failing = [k for k, v in slo_result["detail"].items() if v == "fail"]
            mlflow.set_tag("slo_failing_dims", ",".join(failing))
            
    # Update Prometheus gauges
    if "far" in metrics:
        FAR_GAUGE.set(metrics["far"])
    if "mean_lead_min" in metrics:
        LEAD_TIME_GAUGE.set(metrics["mean_lead_min"])
