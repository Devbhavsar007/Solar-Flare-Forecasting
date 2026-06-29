import os

files = {
    "requirements.txt": """torch==2.3.0
xgboost==2.0.3
timesfm==1.0.1
momentfm==0.0.1
amazon-chronos-forecasting==0.0.1
dspy-ai==2.4.9
graphrag==0.1.1
flaml==2.1.2
mapie==0.8.3
shap==0.45.0
langgraph==0.0.39
peft==0.10.0
transformers==4.40.1
pandera==0.19.3
prometheus-client==0.20.0
PyYAML==6.0.1
pandas==2.2.2
numpy==1.26.4
astropy==6.0.1
sunpy==5.1.1
scipy==1.13.0
joblib==1.4.2
""",
    ".gitignore": """.env
data/raw/
models/*.pt
models/*.onnx
models/*.pkl
models/*.json
data/graphrag/output/
__pycache__/
logs/
""",
    ".env.example": """PRADAN_USERNAME=your_username
PRADAN_PASSWORD=your_password
GH_PAT=your_github_pat
MLFLOW_TRACKING_URI=http://localhost:5000
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/XXX/YYY/ZZZ
ENV=dev
""",
    ".pre-commit-config.yaml": """repos:
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.2
    hooks:
      - id: gitleaks
""",
    "src/__init__.py": "",
    "src/api/__init__.py": "",
    "src/ingestion/__init__.py": "",
    "src/preprocessing/__init__.py": "",
    "src/intelligence/__init__.py": "",
    "src/monitoring/__init__.py": "",
    "configs/nowcasting.yaml": """phase_detector:
  peak_frac_pre: 0.30
  peak_frac_gradual: 0.80
  background_sigma: 0.10
class_thresholds:
  C: 0.38
  M: 0.45
  X: 0.28
  binary: 0.40
dead_time_us: 2.5
""",
    "configs/slo.yaml": """latency_p99_seconds: 90
throughput_cadence_s: 60
availability_pct: 99.0
lead_time_min_accepted: 10
lead_time_min_target: 30
far_max: 0.10
""",
    "configs/version.yaml": """model_version: "0.0.0-dev"
""",
    "configs/model_hashes.yaml": """# SHA-256 hashes of canonical model files
# Populated automatically by promote_if_better() in M16
# Used by verify_model_hash() for integrity checks [T-2]
xgb_multiclass.json: ""
xgb_multiclass.pkl: ""
causal_lstm.pt: ""
conformal_mapie.pkl: ""
tcn_encoder.onnx: ""
""",
    "configs/fits_columns.yaml": """# STOP — do not edit values below until you have run:
# python scripts/verify_fits_columns.py path/to/solexs.fits path/to/hel1os.fits
# Every pipeline failure traces back to wrong column names here.
""",
    "configs/pradan_auth.yaml": """# PRADAN Auth Mechanism - Fill after CHECK 6
login_url: ""
payload_fields: []
""",
    "configs/forecasting.yaml": """# Minimal stub for M0
""",
    "configs/llm.yaml": """# Minimal stub for M0
""",
    "src/monitoring/metrics.py": """from prometheus_client import Counter, Histogram, Gauge

INFERENCE_LATENCY = Histogram(
    "solar_inference_duration_seconds",
    "End-to-end pipeline latency per cycle",
    buckets=[1,5,10,20,30,60,90,120]
)

ALERT_COUNTER = Counter(
    "solar_alert_total",
    "Total alerts triggered",
    ["flare_class", "source"]
)

DATA_FRESHNESS = Gauge(
    "solar_data_freshness_seconds",
    "Age of most recently processed FITS file in seconds"
)

NOWCAST_CONFIDENCE = Gauge(
    "solar_nowcast_confidence",
    "Confidence of most recent nowcast prediction"
)

FAR_GAUGE = Gauge(
    "solar_far_latest",
    "False Alarm Rate from most recent walk-forward evaluation"
)

LEAD_TIME_GAUGE = Gauge(
    "solar_lead_time_minutes",
    "Mean lead time from most recent evaluation"
)
""",
    "src/ingestion/schemas.py": """import pandera as pa
from pandera import Column, DataFrameSchema

SOLEXS_SCHEMA = DataFrameSchema({
    "flux_high": Column(float, pa.Check.greater_than_or_equal_to(0)),
    "flux_low": Column(float, pa.Check.greater_than_or_equal_to(0)),
    # ADD remaining columns after CHECK 1 reveals real names
}, index=pa.Index(pa.DateTime), coerce=False)

HEL1OS_SCHEMA = DataFrameSchema({
    "counts_low": Column(int, pa.Check.greater_than_or_equal_to(0)),
    "counts_high": Column(int, pa.Check.greater_than_or_equal_to(0)),
    # ADD remaining columns after CHECK 1 reveals real names
}, index=pa.Index(pa.DateTime), coerce=False)
"""
}

for path, content in files.items():
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
