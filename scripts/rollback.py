import argparse
import hashlib
import shutil
import yaml
import os
import mlflow
from pathlib import Path

MODEL_FILES = {
    "xgb_multiclass.json": "models/xgb_multiclass.json",
    "xgb_multiclass.pkl":  "models/xgb_multiclass.pkl",
    "causal_lstm.pt":      "models/causal_lstm.pt",
    "conformal_mapie.pkl": "models/conformal_mapie.pkl",
    "tcn_encoder.onnx":    "models/tcn_encoder.onnx",
}

def rollback_to_run(run_id: str, mlflow_tracking_uri: str) -> None:
    '''
    Revert canonical model files to the artifacts of a given MLflow run.
    Updates configs/model_hashes.yaml and configs/version.yaml.
    WHY: If FAR spikes in production after a promotion, the on-call
    engineer runs: python scripts/rollback.py --run-id <prev_run_id>
    This is the only safe rollback path — never manually copy model files.
    '''
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    print(f"Rolling back to run {run_id} "
          f"(TSS={run.data.metrics.get('mean_tss','?')}, "
          f"FAR={run.data.metrics.get('mean_far','?')})")

    # Download artifacts from the target run
    artifact_dir = Path(f"/tmp/rollback_{run_id[:8]}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for name, dest in MODEL_FILES.items():
        try:
            local = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=name,
                dst_path=str(artifact_dir))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(local, dest)
            print(f"  Restored {name}")
        except Exception as exc:
            print(f"  [WARN] Could not restore {name}: {exc}")

    # Recompute and write hashes [T-2]
    hashes = {}
    for name, path in MODEL_FILES.items():
        if Path(path).exists():
            hashes[name] = hashlib.sha256(
                Path(path).read_bytes()).hexdigest()
    
    os.makedirs("configs", exist_ok=True)
    with open("configs/model_hashes.yaml","w") as f:
        yaml.dump(hashes, f)

    # Rewrite version.yaml
    version = f"rollback-{run_id[:8]}"
    with open("configs/version.yaml","w") as f:
        yaml.dump({"model_version": version}, f)
    print(f"Rollback complete. model_version={version}")
    print("Action: restart the api and worker containers to pick up new models.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-id",  required=True)
    p.add_argument("--tracking-uri",
                   default=os.environ.get("MLFLOW_TRACKING_URI",
                                          "http://localhost:5000"))
    args = p.parse_args()
    rollback_to_run(args.run_id, args.tracking_uri)
