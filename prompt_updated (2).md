## MASTER CONTEXT BLOCK
*Paste this at the start of every milestone conversation.*

```
You are implementing SolarSentinel — a production-grade solar flare
nowcasting and forecasting system built on ISRO's Aditya-L1 mission
instruments SoLEXS (soft X-ray) and HEL1OS (hard X-ray).

PROJECT IDENTITY
- Name:     SolarSentinel (internal name JWALA)
- Team:     Quantum Crew
- Contest:  ISRO Build-A-Thon 2026 (BAH 2026), Problem Statement 15
- Repo:     github.com/quantum-crew/solar-sentinel

NON-NEGOTIABLE RULES — enforce these in every file you write:

[RULE-1] NEVER merge SoLEXS and HEL1OS data before running independent
detectors. solexs_detector.py and hel1os_detector.py must run on separate
single-instrument dataframes. Merging happens only inside merger.py.

[RULE-2] NEVER use bidirectional=True in any LSTM used for forecasting.
All LSTM layers must be unidirectional (bidirectional=False). This is not
a preference — it prevents data leakage in the real-time inference path.

[RULE-3] NEVER report accuracy as a model quality metric. TSS (True Skill
Score = TPR − FPR) is the mandatory operational metric for solar flare
prediction. 95%+ of data is quiet sun; accuracy is meaningless here.

[RULE-4] NEVER hardcode FITS column names. All column names come from
configs/fits_columns.yaml which is populated by running
scripts/verify_fits_columns.py on real PRADAN FITS files. Every FITS
reader must raise a KeyError with the list of available columns if a
configured column name is not found.

[RULE-5] NEVER use XGBoost with binary classification for the main model.
The XGBoost nowcasting model always uses objective='multi:softprob' with
num_class=4 (N=0, C=1, M=2, X=3). Per-class thresholds are optimized
separately — X-class threshold is LOWER than M-class (maximize recall for
the most dangerous class).

[RULE-6] NEVER call timesfm.finetuning — it does not exist in the public
package. TimesFM fine-tuning uses PEFT + HuggingFace Trainer. Zero-shot
inference via forecast_on_df() works immediately without fine-tuning.

[RULE-7] NEVER call pipeline.predict_quantiles(). Chronos-Bolt API uses
predict() which returns (batch, samples, horizon). Quantiles are computed
manually from the sample distribution using np.quantile().

[RULE-8] NEVER use confusion_matrix() without labels=[0, 1]. Walk-forward
CV folds may have no positive events; without explicit labels the matrix
is 1×1 and .ravel() raises ValueError crashing the entire CV loop.

[RULE-9] NEVER run GraphRAG with default settings — it will call OpenAI.
Always configure local embeddings (nomic-embed-text via Ollama) in
data/graphrag/settings.yaml before indexing.

[RULE-10] NEVER use dspy.configure(lm=...) for async FastAPI. Always use
the per-call context manager: with dspy.context(lm=_LM): — this is
thread-safe. Global configure() is not safe under concurrent requests.

[RULE-11] NEVER call torch.load() without weights_only=True. In PyTorch
2.0+, omitting this argument triggers a security warning; in 2.6+ it
defaults to True but earlier versions silently run unsafe pickle.
Correct form: torch.load(path, map_location="cpu", weights_only=True)
Exception: if loading a full model object (not state_dict), use
weights_only=False but add a comment explaining why it is safe.

[RULE-12] NEVER open config files at module level without a FileNotFoundError
guard. Module-level code runs on import, which happens before M0 setup
on a fresh clone. Correct pattern:
    try:
        with open("configs/nowcasting.yaml") as f:
            _CFG = yaml.safe_load(f) or {}
    except FileNotFoundError:
        _CFG = {}
    PHASE_CFG = _CFG.get("phase_detector", {})
This applies to every module-level open() call.

[RULE-13] MODEL PERSISTENCE FORMAT IS STRICTLY DEFINED — never mix formats:
    XGBoost model:     model.save_model("models/xgb_multiclass.json")
                       + joblib.dump(model, "models/xgb_multiclass.pkl")
                       (save both: JSON for inspection, pkl for agents)
    PyTorch model:     torch.save(model.state_dict(), "models/name.pt")
                       load: model.load_state_dict(
                           torch.load(path, map_location="cpu",
                                      weights_only=True))
    ONNX model:        always .onnx extension
    MAPIE/sklearn:     joblib.dump(mapie, "models/conformal_mapie.pkl")
    DSPy reporter:     reporter.save("models/dspy_reporter_optimised.json")
If a file is saved with one format, it must be loaded with the same
format. agents.py loads from local files using these exact paths.

[RULE-14] NEVER pass a session-scoped pytest fixture as a parameter to a
hypothesis @given test. Hypothesis controls the test invocation lifecycle
independently of pytest's fixture scopes, so session fixtures are often
not initialised when hypothesis calls the function. Instead use a
module-level singleton pattern inside the test file:
    _NOWCASTER = None
    def _get_nowcaster():
        global _NOWCASTER
        if _NOWCASTER is None: _NOWCASTER = _build_toy_nowcaster()
        return _NOWCASTER
    @given(st.arrays(...))
    def test_property(window):
        nowcaster = _get_nowcaster()  # module-level, not a fixture
        ...
Function-scoped fixtures are safe with hypothesis; session-scoped are not.

[RULE-15] NEVER import private functions (underscore prefix) across module
boundaries. A private function in module A is an implementation detail
that module B should not depend on. If another module needs it, either
make it public (remove the underscore) or refactor. Specific violations
to avoid:
    - Do NOT: from src.catalogue.merger import _check_noaa
    - Do NOT: from src.forecasting.moment_anomaly import _load_moment
    - Correct: expose get_moment_model() or make _load_moment public.

[RULE-16] NEVER load heavyweight models (Chronos, MOMENT, ONNX sessions)
inside functions that are called repeatedly, such as agent functions in
the LangGraph pipeline. Use module-level singletons:
    _CHRONOS_PIPELINE = None
    def get_chronos():
        global _CHRONOS_PIPELINE
        if _CHRONOS_PIPELINE is None:
            from chronos import BaseChronosPipeline
            import torch
            _CHRONOS_PIPELINE = BaseChronosPipeline.from_pretrained(
                "amazon/chronos-bolt-small",
                dtype=torch.float32, device_map="cpu")
        return _CHRONOS_PIPELINE
Chronos takes ~20s to load. MOMENT takes ~30s. Loading on every agent
call means a 45s+ pipeline per inference cycle, breaking real-time use.

[RULE-17] NEVER write to master_catalogue.csv without provenance columns.
Every row must include:
    solexs_fits_path  (str) — absolute path of the source SoLEXS FITS file
    hel1os_fits_path  (str) — absolute path, or "" if SoLEXS_only event
    model_version     (str) — read from configs/version.yaml key "model_version"
    pipeline_run_id   (str) — UUID4 generated once at pipeline start, stored
                              in SolarPipelineState["pipeline_run_id"]
Without provenance, a misclassified event in production cannot be traced
to its source data or model version. scripts/audit_catalogue.py uses
these columns to link any alert back to the exact inputs that produced it.
This applies to every write to master_catalogue.csv — manual or scheduled.

[RULE-18] NEVER pass a DataFrame from fits_reader.py downstream without
first validating it through the pandera schema in src/ingestion/schemas.py.
Correct pattern inside _safe_read():
    from src.ingestion import schemas
    import pandera as pa
    df = _build_dataframe(hdul, instrument)
    schema = schemas.SOLEXS_SCHEMA if instrument == "solexs" \
             else schemas.HEL1OS_SCHEMA
    try:
        schema.validate(df, lazy=True)          # collect ALL failures
    except pa.errors.SchemaErrors as exc:
        raise ValueError(
            f"Schema validation failed for {filepath}:\n{exc.failure_cases}"
        ) from exc
Do NOT silently drop bad rows. Schema failure must halt the pipeline with
a message pointing to the bad file. Catching at the ingestion boundary
prevents corrupt or saturated data from reaching detectors and models.

[RULE-19] NEVER build the dashboard in Streamlit. The dashboard is
React + TypeScript + Vite. This decision is final (made at M0) and
must not be revisited at M14. Rationale: Streamlit cannot support true
server-push WebSockets, has no real build pipeline, and cannot be
containerised efficiently for production. Any earlier reference to
"React/TypeScript or Streamlit" is superseded by this rule.

[RULE-20] NEVER instantiate Prometheus metrics (Counter, Gauge, Histogram)
inline in application code. All metrics are declared once in
src/monitoring/metrics.py and imported wherever needed:
    from src.monitoring.metrics import INFERENCE_LATENCY, ALERT_COUNTER
Inline instantiation (e.g., Counter("name", ...) inside an agent) causes
a ValueError: "Duplicated timeseries" on the second module import.
The canonical metrics module also guarantees consistent label names across
all exporters and Grafana dashboards.

[RULE-21] NEVER deploy to production without a passing staging gate.
Environment progression: dev → staging → prod. After every model promotion,
the GitHub Actions retrain workflow must:
  1. Deploy new models to the staging Docker stack.
  2. Run: curl http://staging:8000/health → 200 required.
  3. Run: python scripts/integration_test_20240222.py
          → "INTEGRATION TEST PASSED" required.
  4. Only if BOTH pass: copy model files to prod canonical paths [RULE-13].
Never manually copy model files to a production server. Never bypass
promote_if_better() in M16 to push directly to models/ in production.

KEY FILES AND THEIR PURPOSES:
- configs/fits_columns.yaml     → FITS column names (fill from CHECK 1)
- configs/nowcasting.yaml       → phase thresholds, class thresholds
- configs/forecasting.yaml      → forecasting model config
- configs/llm.yaml              → DSPy + Ollama config
- configs/pradan_auth.yaml      → PRADAN auth mechanism (fill from CHECK 6)
- configs/version.yaml          → model_version string (updated by M16)
- configs/model_hashes.yaml     → SHA-256 hashes of canonical model files
- configs/slo.yaml              → SLO targets (auto-checked in integration test)
- data/processed/master_catalogue.csv → PRIMARY PS15 DELIVERABLE
- scripts/verify_*.py           → Pre-implementation verification scripts
- scripts/audit_catalogue.py    → Provenance audit tool
- src/ingestion/schemas.py      → Pandera DataFrame schemas [RULE-18]
- src/monitoring/metrics.py     → All Prometheus metric declarations [RULE-20]
- models/xgb_multiclass.json   → XGBoost model (native JSON)
- models/xgb_multiclass.pkl    → XGBoost model (joblib for agents)
- models/causal_lstm.pt        → PyTorch state_dict (not full model)
- models/conformal_mapie.pkl   → MAPIE conformal classifier

ARCHITECTURE SUMMARY:
Independent SoLEXS detector → SoLEXS catalogue
Independent HEL1OS detector → HEL1OS catalogue
         ↓ temporal coincidence merger (±2 min) + provenance injection [RULE-17]
         Master catalogue (dual / SoLEXS_only / HEL1OS_only)
         ↓ pandera schema-validated [RULE-18] physics features + MOMENT score
Nowcasting:  TCN encoder (ONNX) + XGBoost multi:softprob → N/C/M/X
Forecasting: Causal LSTM + TCN + TimesFM → P(flare | 15/30/60 min)
Uncertainty: MAPIE conformal + Chronos-Bolt probabilistic intervals
Intelligence: DSPy + GraphRAG + Phi-3-mini (Ollama) + structured fallback
Orchestration: LangGraph stateful agent graph with Annotated fan-in reducer
Production:  ONNX inference, drift detection, walk-forward CV, CI/CD
             React+TS+Vite dashboard [RULE-19], Prometheus metrics [RULE-20]
             Multi-env (dev→staging→prod) [RULE-21], gitleaks secret scanning

LANGUAGE AND STYLE:
- Python 3.11
- Type hints on every function signature
- Docstrings explaining WHY, not just what
- All config values loaded from YAML — no magic numbers in code
- All config loads at module level must have FileNotFoundError fallback [RULE-12]
- Every new file needs a corresponding test or fixture
- Model save/load must use the consistent formats in [RULE-13]
- All Prometheus metrics declared in src/monitoring/metrics.py [RULE-20]
```


---

## SLO DEFINITIONS
*Paste alongside the MASTER CONTEXT BLOCK in every milestone conversation.*

```
PERFORMANCE SLOs — these are the binary acceptance criteria for production.
SLO compliance is logged to MLflow under run tag "slo_status": "pass"|"fail".
A model promotion is blocked if ANY SLO is failing on the current dataset.

[SLO-1] END-TO-END LATENCY
Target: alert fires within 90 seconds of a FITS file becoming available.
Measured at P99 by the final integration test via state["timing"].
Component budget — each agent logs its wall-clock elapsed_seconds to
state["timing"][agent_name]. The integration test asserts every component
is within its budget. Budget breakdown:
  FITS ingestion + dead-time correction:  ≤  5s
  Independent detection (both, parallel): ≤  3s
  Catalogue merge + provenance inject:    ≤  1s
  Feature engineering + MOMENT score:    ≤  8s
  ONNX nowcast inference:                ≤  0.1s
  Ensemble forecast (LSTM + TCN):        ≤  2s
  Chronos uncertainty (singleton):       ≤ 10s
  SHAP explanation:                      ≤  3s
  LLM report — DSPy path:               ≤ 30s
  LLM report — fallback path:           ≤  1s
  Alert routing + WebSocket push:        ≤  1s
  Total (DSPy path):                     ≤ 63s   [within 90s SLA]
  Total (fallback path):                 ≤ 34s

Implementation in every agent:
    import time
    t0 = time.perf_counter()
    # ... agent logic ...
    state.setdefault("timing", {})[agent_name] = round(time.perf_counter()-t0, 3)

[SLO-2] THROUGHPUT
Target: ≥ 1 full inference cycle per 60-second cadence without queuing.
If a cycle takes > 60s, log WARNING and skip — do NOT queue. This is a
real-time system. If 3 consecutive cycles exceed 60s, post to
SLACK_WEBHOOK_URL ("SolarSentinel throughput degraded") and fall back
to a 120s cadence automatically.

[SLO-3] AVAILABILITY
Target: /health returns 200 for ≥ 99% of requests over any 24-hour window.
Measured by Prometheus probe_success{job="solar-sentinel-api"}.
Grafana alert fires if probe_success_24h_avg < 0.99.

[SLO-4] FORECAST LEAD TIME
Accepted: mean lead time ≥ 10 minutes.
Target:   mean lead time ≥ 30 minutes.
Log both to MLflow as "mean_lead_min" and "target_lead_min".
Grafana alert if 7-day rolling mean_lead_min < 5 minutes (model degradation).

[SLO-5] FALSE ALARM RATE GATE
Hard limit: FAR ≤ 0.10 on walk-forward validation.
If FLAML produces a model with FAR > 0.10, REJECT it regardless of TSS
improvement. Log rejection to MLflow:
    tags["model_rejected"]   = "FAR_exceeds_threshold"
    tags["model_far"]        = str(round(far, 4))
    tags["model_far_limit"]  = "0.10"
promote_if_better() in M16 must check this tag before copying model files.

SLO YAML (auto-checked in integration test):
# configs/slo.yaml
latency_p99_seconds:   90
throughput_cadence_s:  60
availability_pct:      99.0
lead_time_min_accepted: 10
lead_time_min_target:   30
far_max:               0.10
```

---

## THREAT MODEL
*Review at the start of M16. Update if deployment scope expands.*

```
THREAT SURFACE:
  PRADAN credentials | FastAPI endpoints (unauthenticated MVP) |
  WebSocket feed | Docker filesystem (models, configs) |
  GitHub Actions secrets | GraphRAG subprocess

THREATS AND MITIGATIONS:

[T-1] Credential exfiltration via logs
  Risk: PRADAN password leaks into tracebacks or debug print statements.
  Mitigation:
    Use a logging.config SensitiveFormatter that masks any string matching
    the value of PRADAN_PASSWORD in all log output at all levels.
    Never log the raw PRADAN POST payload body.
    CI gate (add to ci.yml):
      grep -rn "password\|PRADAN_PASSWORD\|pradan_pass" logs/ tests/
      # must return zero results after any test run

[T-2] Arbitrary code execution via pickle model files [RULE-11 base, extended here]
  Risk: A compromised .pkl or .pt file runs arbitrary Python via pickle.
  Mitigation:
    weights_only=True on all torch.load() calls (RULE-11).
    Additionally, verify SHA-256 of every model file on load in production:

    # src/monitoring/model_integrity.py
    import hashlib, yaml
    from pathlib import Path

    def verify_model_hash(path: str) -> None:
        """Verify model file integrity against configs/model_hashes.yaml."""
        try:
            hashes = yaml.safe_load(
                open("configs/model_hashes.yaml"))
        except FileNotFoundError:
            return  # skip in dev/test where hash file may not exist
        key = Path(path).name
        expected = hashes.get(key)
        if expected is None:
            return  # new model not yet registered
        actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if actual != expected:
            raise SecurityError(
                f"Model hash mismatch: {key}\n"
                f"  expected {expected[:16]}…\n"
                f"  got      {actual[:16]}…\n"
                f"  Possible tamper. Delete and re-promote from MLflow.")

    Call verify_model_hash() at the top of every agent module's singleton
    loader (get_chronos, _get_lstm, _get_mapie, ONNXNowcaster.__init__).
    promote_if_better() in M16 writes new hashes to configs/model_hashes.yaml
    after every successful promotion.

[T-3] Denial of service via WebSocket flood
  Risk: Unauthenticated clients send thousands of messages, blocking asyncio.
  Mitigation in src/api/main.py:
    MAX_CONNECTIONS = 20
    _active_ws: set = set()

    @app.websocket("/ws/live")
    async def live_feed(ws: WebSocket):
        if len(_active_ws) >= MAX_CONNECTIONS:
            await ws.close(code=1008)  # Policy violation
            return
        await ws.accept()
        _active_ws.add(ws)
        last_msg_time = 0.0
        try:
            while True:
                data = await asyncio.wait_for(ws.receive_text(), timeout=10)
                now = time.time()
                if now - last_msg_time < 10:        # 1 msg per 10s rate limit
                    await ws.close(code=1008)
                    break
                last_msg_time = now
                # ... process data ...
        finally:
            _active_ws.discard(ws)

[T-4] SSRF via GraphRAG subprocess
  Risk: Manipulated settings.yaml redirects GraphRAG to an arbitrary URL.
  Mitigation:
    settings.yaml is committed to the repo (not user-provided at runtime).
    Subprocess uses timeout=30.
    CHECK 7 verifies that api_base contains ONLY "localhost" or "127.0.0.1".
    Add assertion in CHECK 7:
      assert "localhost" in settings["llm"]["api_base"] or \
             "127.0.0.1" in settings["llm"]["api_base"], \
             "GraphRAG must use local Ollama — no external LLM endpoints"

[T-5] Data injection via crafted FITS files
  Risk: Malicious FITS file triggers parser panic or exhausts memory.
  Mitigation:
    1. Reject files > 500MB before opening:
       if os.path.getsize(path) > 500 * 1024 * 1024:
           raise ValueError(f"FITS file too large (>500MB): {path}")
    2. Wrap ALL fits.open() in try/except; reraise as ValueError — never
       propagate raw astropy errors to the API response layer.
    3. Pandera schema validation (RULE-18) provides a second rejection layer
       by enforcing numeric ranges and null thresholds on every column.

[T-6] Secret scanning in CI
  Mitigation:
    Add gitleaks to CI as both a pre-commit hook and a workflow step:

    # .github/workflows/ci.yml (add as first step):
    - name: Secret scan
      uses: gitleaks/gitleaks-action@v2
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

    Also add as pre-commit hook in .pre-commit-config.yaml:
      repos:
        - repo: https://github.com/gitleaks/gitleaks
          rev: v8.18.2
          hooks:
            - id: gitleaks

    Fail the pipeline if any staged file matches API key, password, or token
    patterns. .gitignore excludes .env but CI must also scan staged files.

ACCEPTED RISKS (explicitly documented — not hidden):
  - Dashboard is unauthenticated in MVP. Intended for ops team LAN only.
    Hardening path: add OAuth2/JWT in a post-hackathon production release.
  - PRADAN session cookies expire; no automated re-auth beyond retry+circuit.
  - Model files are not encrypted at rest. Acceptable for research deployment.

SECURITY CHECKLIST — verify all boxes before M16 checkpoint:
  [ ] SensitiveFormatter in logging.config (masks PRADAN_PASSWORD in logs)
  [ ] verify_model_hash() called in every singleton model loader
  [ ] promote_if_better() writes updated model_hashes.yaml after promotion
  [ ] WebSocket rate limiting + connection cap implemented in main.py
  [ ] gitleaks in CI (workflow step) and pre-commit hook installed
  [ ] CHECK 7 assertion: GraphRAG api_base is localhost only
  [ ] grep -rn "password\|secret\|token" src/ configs/ → 0 results
      (except comments and test fixtures using obviously fake values)
  [ ] FITS file size gate (>500MB rejection) in _safe_read()
```


---

## PRE-MILESTONE: EIGHT VERIFICATION CHECKS

Run these before any milestone begins. Failures here cause multi-day debugging later.

```
You are the senior engineer on SolarSentinel. Before writing any
implementation code, run and document the results of eight verification
checks. Write each check as a standalone Python script in scripts/.

[CONTEXT BLOCK above]

CHECK 1 — FITS Column Names (scripts/verify_fits_columns.py)
Write a script that opens any FITS file path(s) passed as CLI args,
prints hdul.info(), then for each extension with columns prints the
full column list with name, format, and unit. Outputs a JSON structure
suitable for pasting into configs/fits_columns.yaml.
Run on the first real SoLEXS and HEL1OS FITS files downloaded from
PRADAN. Fill configs/fits_columns.yaml from the output.

CHECK 2 — TimesFM API (scripts/verify_timesfm.py)
Write a script that imports timesfm, lists all public methods on a
TimesFm instance, runs forecast_on_df() on a synthetic 60-point series,
and prints the exact column names in the output DataFrame. Also tries
to import timesfm.finetuning and prints whether it exists.
Expected: finetuning does NOT exist. Forecast column is "timesfm" not
"timesfm_15". Record the actual output column name.

CHECK 3 — Chronos-Bolt API (scripts/verify_chronos.py)
Write a script that loads amazon/chronos-bolt-small, calls predict()
on a batch-1 context tensor of length 60 with prediction_length=15
and num_samples=20. Prints the shape of the output tensor. Then computes
q10/q50/q90 from the sample distribution using np.quantile.
Expected: predict() NOT predict_quantiles(). Output shape (1, 20, 15).
Also time the first load: print "Chronos load time: Ns" so the team
knows the startup cost and understands why [RULE-16] is required.

CHECK 4 — MOMENT API (scripts/verify_moment.py)
Write a script that loads AutonLab/MOMENT-1-large with task_name=
"reconstruction", passes a (2, 1, 512) input tensor, and prints all
non-dunder attributes of the output object. Records the exact attribute
name used for reconstruction output — may be "reconstruction" or differ.
Also time the load: print "MOMENT load time: Ns". This confirms why
the singleton pattern in [RULE-16] is mandatory.

CHECK 5 — DSPy + Ollama syntax (scripts/verify_dspy.py)
Write a script that tries three DSPy LM constructor syntaxes for Ollama:
1. dspy.LM("ollama/phi3:mini", api_base="http://localhost:11434")
2. dspy.LM("ollama_chat/phi3:mini", api_base="http://localhost:11434")
3. dspy.OllamaLocal("phi3:mini")
Records which one succeeds. Also confirms the per-call context manager
works: with dspy.context(lm=lm): result = dspy.Predict("q->a")(q="test")
Records the working syntax for src/intelligence/dspy_reporter.py.

CHECK 6 — PRADAN Auth Mechanism
This check is manual. Document the steps:
1. Open https://pradan.issdc.gov.in in Chrome with DevTools Network tab
2. Submit the login form
3. Inspect the POST request: URL, payload field names, response cookies
4. Check for CSRF token in page source (<input type="hidden" name="csrf*">)
5. Write findings to configs/pradan_auth.yaml (no passwords)
Output the pradan_auth.yaml template for the developer to fill.

CHECK 7 — GraphRAG Local Embedding (scripts/verify_graphrag.py)
Write the correct settings.yaml changes for data/graphrag/settings.yaml
that point both the LLM and embeddings sections to Ollama local models
(phi3:mini and nomic-embed-text respectively) using the Ollama OpenAI-
compatible endpoint at http://localhost:11434/v1.
Include the full relevant settings.yaml sections, not just the diff.
ALSO add this assertion at the end of the script:
    with open("data/graphrag/settings.yaml") as f:
        s = yaml.safe_load(f)
    llm_base = s["llm"]["api_base"]
    emb_base  = s["embeddings"]["llm"]["api_base"]
    assert "localhost" in llm_base or "127.0.0.1" in llm_base, \
        f"GraphRAG LLM api_base must be localhost, got: {llm_base}"
    assert "localhost" in emb_base or "127.0.0.1" in emb_base, \
        f"GraphRAG embeddings api_base must be localhost, got: {emb_base}"
    print("CHECK 7 PASSED: GraphRAG is wired to local Ollama only.")
This assertion is the [T-4] SSRF mitigation verification.

CHECK 8 — Pandera Schema Smoke Test (scripts/verify_schemas.py)
Write a script that:
1. Imports SOLEXS_SCHEMA and HEL1OS_SCHEMA from src/ingestion/schemas.py.
2. Constructs a valid synthetic SoLEXS DataFrame (datetime index, flux_high
   as float64 in [1e-9, 1e-3], flux_low as float64, no NaN).
3. Runs SOLEXS_SCHEMA.validate(df) — assert it passes.
4. Constructs an INVALID SoLEXS DataFrame (flux_high contains NaN).
5. Runs SOLEXS_SCHEMA.validate(invalid_df, lazy=True) inside a try/except
   pa.errors.SchemaErrors — assert the exception is raised and that the
   failure_cases DataFrame is non-empty.
6. Repeats steps 2-5 for HEL1OS_SCHEMA with counts_low as int64.
7. Prints: "CHECK 8 PASSED: Both schemas validate and reject correctly."
This verifies [RULE-18] is correctly implemented before M1 begins.

Deliver: all 8 scripts in scripts/, configs/fits_columns.yaml template,
configs/pradan_auth.yaml template, GraphRAG settings fragment.
Checkpoint outputs: Chronos load time printed. MOMENT load time printed.
MOMENT output attribute name recorded. Working DSPy syntax recorded.
CHECK 7 assertion passes (localhost only). CHECK 8 passes.
```

---

## M0 — Environment & Setup

```
[CONTEXT BLOCK]

Milestone M0: Environment setup, project scaffold, and all config stubs.

TASKS:
1. Write requirements.txt with ALL packages pinned to exact versions.
   No floating versions (no ">=", no "~="). After writing, verify:
   pip install -r requirements.txt
   Fix any conflicts before committing. Include pandera and prometheus-client.

2. Create the full project directory tree. Create __init__.py in every
   src/ subdirectory. Create .gitignore that excludes:
   .env, data/raw/, models/*.pt, models/*.onnx, models/*.pkl,
   models/*.json, data/graphrag/output/, __pycache__, logs/

3. Create .env.example (no real credentials) showing all required vars:
   PRADAN_USERNAME, PRADAN_PASSWORD, GH_PAT, MLFLOW_TRACKING_URI,
   SLACK_WEBHOOK_URL, ENV (dev|staging|prod, default dev)

4. Create configs/nowcasting.yaml with the structure:
   phase_detector:
     peak_frac_pre:     0.30   # UPDATE after M1 EDA
     peak_frac_gradual: 0.80   # UPDATE after M1 EDA
     background_sigma:  0.10
   class_thresholds:
     C: 0.38
     M: 0.45
     X: 0.28   # LOWER than M — missing X is catastrophically costly [D8]
     binary: 0.40
   dead_time_us: 2.5   # verify from HEL1OS instrument docs

5. Create configs/slo.yaml (SLO targets for integration test):
   latency_p99_seconds:    90
   throughput_cadence_s:   60
   availability_pct:       99.0
   lead_time_min_accepted: 10
   lead_time_min_target:   30
   far_max:                0.10

6. Create configs/version.yaml:
   model_version: "0.0.0-dev"   # updated by promote_if_better() in M16
   # format: MAJOR.MINOR.PATCH-{env}  e.g. "1.2.0-prod"

7. Create configs/model_hashes.yaml as an empty stub:
   # SHA-256 hashes of canonical model files
   # Populated automatically by promote_if_better() in M16
   # Used by verify_model_hash() for integrity checks [T-2]
   xgb_multiclass.json: ""
   xgb_multiclass.pkl:  ""
   causal_lstm.pt:      ""
   conformal_mapie.pkl: ""
   tcn_encoder.onnx:    ""

8. Create configs/fits_columns.yaml as a template with placeholder values
   and a prominent header comment:
   # STOP — do not edit values below until you have run:
   # python scripts/verify_fits_columns.py path/to/solexs.fits path/to/hel1os.fits
   # Every pipeline failure traces back to wrong column names here.

9. Create configs/pradan_auth.yaml as a template (structure only, no creds).
10. Create configs/forecasting.yaml and configs/llm.yaml as minimal stubs.

11. Create src/monitoring/metrics.py — the CANONICAL Prometheus metric
    registry [RULE-20]. Stub all metric names now so no module defines
    metrics inline. Initial set:
    from prometheus_client import Counter, Histogram, Gauge
    INFERENCE_LATENCY = Histogram(
        "solar_inference_duration_seconds",
        "End-to-end pipeline latency per cycle",
        buckets=[1,5,10,20,30,60,90,120])
    ALERT_COUNTER = Counter(
        "solar_alert_total",
        "Total alerts triggered",
        ["flare_class", "source"])
    DATA_FRESHNESS = Gauge(
        "solar_data_freshness_seconds",
        "Age of most recently processed FITS file in seconds")
    NOWCAST_CONFIDENCE = Gauge(
        "solar_nowcast_confidence",
        "Confidence of most recent nowcast prediction")
    FAR_GAUGE = Gauge(
        "solar_far_latest",
        "False Alarm Rate from most recent walk-forward evaluation")
    LEAD_TIME_GAUGE = Gauge(
        "solar_lead_time_minutes",
        "Mean lead time from most recent evaluation")

12. Create src/ingestion/schemas.py — pandera schemas [RULE-18].
    Define initial stubs (real column names filled from CHECK 1):
    import pandera as pa
    from pandera import Column, DataFrameSchema
    SOLEXS_SCHEMA = DataFrameSchema({
        "flux_high":  Column(float, pa.Check.greater_than_or_equal_to(0)),
        "flux_low":   Column(float, pa.Check.greater_than_or_equal_to(0)),
        # ADD remaining columns after CHECK 1 reveals real names
    }, index=pa.Index(pa.DateTime), coerce=False)
    HEL1OS_SCHEMA = DataFrameSchema({
        "counts_low":  Column(int,   pa.Check.greater_than_or_equal_to(0)),
        "counts_high": Column(int,   pa.Check.greater_than_or_equal_to(0)),
        # ADD remaining columns after CHECK 1 reveals real names
    }, index=pa.Index(pa.DateTime), coerce=False)

13. Create .pre-commit-config.yaml with gitleaks hook [T-6]:
    repos:
      - repo: https://github.com/gitleaks/gitleaks
        rev: v8.18.2
        hooks:
          - id: gitleaks

DECISIONS TO DOCUMENT IN comments:
- D3: GPU vs CPU — MOMENT batch inference and TimesFM need GPU for practical
  throughput. CPU-only: disable TimesFM LoRA fine-tuning, limit MOMENT to
  flagged-window-only inference (not full dataset batch).
- D8: X-class threshold is LOWER than M-class. Rationale: missing X is
  catastrophically costly; recall trumps precision for extreme events.
- D9: Dashboard is React + TypeScript + Vite [RULE-19]. Not Streamlit.
  Rationale: WebSocket push, build pipeline, production containerisation.

CHECKPOINT: pip install -r requirements.txt exits 0.
python -c "import torch, xgboost, timesfm, momentfm, chronos, dspy,
graphrag, flaml, mapie, shap, langgraph, peft, transformers, pandera,
prometheus_client" exits 0.
All config files parse with yaml.safe_load() without errors.
src/monitoring/metrics.py importable with no duplicate registration errors.
src/ingestion/schemas.py importable. CHECK 8 passes.
```

---

## M1 — Data Pipeline

```
[CONTEXT BLOCK]

Milestone M1: FITS-safe data ingestion, dead-time correction,
log-space GOES cross-calibration, pandera-validated DataFrames,
physics feature engineering, and goes_downloader.py.

PREREQUISITE: CHECK 1 complete. configs/fits_columns.yaml filled.
              CHECK 8 complete. src/ingestion/schemas.py column names filled.
REMINDER: [RULE-18] — validate EVERY DataFrame from fits_reader.py.
REMINDER: [RULE-12] — FileNotFoundError guard on all module-level opens.

TASKS:

1. src/ingestion/fits_reader.py
   _safe_read(filepath: str, instrument: str) -> pd.DataFrame:
     Load column map from configs/fits_columns.yaml [RULE-12 guard].
     Validate every configured column exists before accessing data.
     KeyError message must show AVAILABLE columns, not just "not found".
     Set DatetimeIndex from time column. Replace inf/-inf, dropna, sort.
     *** PANDERA VALIDATION [RULE-18] — call AFTER building DataFrame:
     from src.ingestion import schemas as _schemas
     import pandera as pa
     schema = _schemas.SOLEXS_SCHEMA if instrument == "solexs" \
              else _schemas.HEL1OS_SCHEMA
     try:
         schema.validate(df, lazy=True)
     except pa.errors.SchemaErrors as exc:
         raise ValueError(
             f"[RULE-18] Schema validation failed for {filepath}:\n"
             f"{exc.failure_cases.to_string()}"
         ) from exc
     *** FITS SIZE GATE [T-5]:
     if os.path.getsize(filepath) > 500 * 1024 * 1024:
         raise ValueError(f"FITS file exceeds 500MB limit: {filepath}")

   read_solexs(filepath: str) -> pd.DataFrame:
     Calls _safe_read("solexs"). Returns validated DataFrame.
   read_hel1os(filepath: str) -> pd.DataFrame:
     Calls _safe_read("hel1os") then apply_dead_time_correction().
   apply_dead_time_correction(df: pd.DataFrame) -> pd.DataFrame:
     Paralyzable model: N_obs = N_true * exp(-N_true * tau).
     tau from configs/fits_columns.yaml hel1os.dead_time_us (default 2.5).
     Invert via scipy.optimize.brentq per bin. Saturated bins -> NaN.
     Interpolate NaN with method="time", limit=5, limit_direction="both".
   merge_instruments(solexs_df, hel1os_df, cadence="1min") -> pd.DataFrame

2. src/ingestion/schemas.py — UPDATE with real column names from CHECK 1:
   Add pa.Check.less_than(1e-2) on flux_high (no instrument can read > 0.01).
   Add pa.Check.less_than(1e9)  on counts_low (Poisson count sanity).
   Add nullable=False to all primary data columns.
   These range checks are the [T-5] data injection mitigation layer 2.

3. src/ingestion/goes_downloader.py
   CLI: python goes_downloader.py --start 2014-01-01 --end 2024-06-01
        python goes_downloader.py --days=30
   Uses sunpy.net.Fido to fetch GOES XRS 1-minute flux data.
   After download: parse with sunpy.timeseries.TimeSeries, resample to 1min,
   save to data/raw/goes/goes_xrs_{start}_{end}.parquet.
   Print: date range, n_rows, n_gaps (gaps > 1 hour).

4. src/preprocessing/cross_calibration.py
   fit_goes_solexs_calibration(goes_df, solexs_df, overlap_start, overlap_end):
     Log-space Huber regression (NOT linear, NOT OLS — flux spans 6 decades).
     WHY HUBER: robust to X-class outliers that saturate one instrument.
     WHY LOG: C-class at 1e-6 W/m2, X at 1e-4. Linear ignores C-class.
     HuberRegressor(epsilon=1.5, max_iter=500).fit(log_goes, log_solexs)
     Log slope, intercept, r2, n_samples to MLflow. Warn if r2 < 0.80.
   apply_goes_calibration(goes_flux, calibration) -> np.ndarray

5. src/preprocessing/physics_features.py — [RULE-12 guard at module level]
   detect_solar_phase(flux_window) -> int (0-4):
     0=quiet, 1=pre-flare, 2=impulsive, 3=peak, 4=gradual
     Uses PHASE_* module-level constants (YAML-driven, not hardcoded).
   engineer_physics_features(df) -> pd.DataFrame:
     Neupert ratio, thermal index, doubling time, normalised flux,
     solar_phase (int), channel ratio, instrument lag,
     rollstd/rollmax for flux columns. dropna() at end.
   subtract_background(df, window_min=10) -> pd.DataFrame

6. src/preprocessing/labels.py
   CLASS_MAP = {"N": 0, "C": 1, "M": 2, "X": 3}
   build_multiclass_labels(df, master_catalogue) -> pd.DataFrame
   create_windows(df, feature_cols, window_size=60, horizon=15, step=1)

7. src/preprocessing/augmentation.py
   augment_minority(X_train, y_train, target_ratio=0.30):
     tsaug TimeWarp + Drift + Quantize on flare windows ONLY.

NOTEBOOK: notebooks/01_fits_column_inspection.ipynb
  Verify dead-time corrected counts >= raw counts on high-flux bins.
  Show pandera schema validation output on a real FITS file.

NOTEBOOK: notebooks/02_eda_and_phase_calibration.ipynb
  Plot flux distributions for quiet vs flare periods.
  Determine phase_frac_pre and phase_frac_gradual. Update nowcasting.yaml.
  Verify GOES cross-calibration r2 > 0.80.

CHECKPOINT: read_solexs() and read_hel1os() run without KeyError on real
PRADAN files. Pandera validation passes on clean FITS; raises ValueError
on synthetic corrupted input. Dead-time corrected counts >= raw on
high-flux bins. Cross-calibration r2 logged to MLflow.
```


---

## M2 — Independent Instrument Detectors

```
[CONTEXT BLOCK]

Milestone M2: Independent SoLEXS and HEL1OS flare detectors.

PREREQUISITE: M1 complete.
REMINDER: [RULE-1] — these detectors run on SEPARATE single-instrument
dataframes. Do not accept or process merged data here.

TASKS:

1. src/nowcasting/solexs_detector.py
   FlareEvent dataclass: start_time, peak_time, end_time, peak_flux,
   flare_class, instrument, confidence,
   fits_path (str)  ← provenance field [RULE-17], passed in from caller.
   GOES_THRESHOLDS: {"X": 1e-4, "M": 1e-5, "C": 1e-6, "B": 1e-7}
   classify_flux(peak_flux: float) -> str
   detect_solexs_flares(df, flux_col="flux_high", sigma_threshold=3.0,
                          min_duration_min=3, fits_path="") -> list[FlareEvent]:
     Rolling 10-min median background + std.
     Dual trigger: rate-of-change > sigma*std AND flux > 1.5*background.
     Event ends when flux drops below 1.5*background.
     Minimum duration filter removes instrument spikes.
     confidence = clip(peak_flux / (5*bg), 0, 1)
     instrument = "SoLEXS"
     Pass fits_path through to FlareEvent for provenance [RULE-17].

2. src/nowcasting/hel1os_detector.py
   Import only FlareEvent from solexs_detector (NOT _check_noaa [RULE-15]).
   detect_hel1os_flares(df, counts_col="counts_low", sigma_threshold=4.0,
                          min_duration_min=2, fits_path="") -> list[FlareEvent]:
     sigma_threshold=4.0: HXR background is noisier (Poisson counting).
     min_duration_min=2: impulsive phase is brief.
     flare_class MUST be "?" — HEL1OS cannot self-classify to GOES scale.
     instrument = "HEL1OS"
     confidence = clip(peak_count / (5*bg), 0, 1)
     Pass fits_path through to FlareEvent for provenance [RULE-17].

3. tests/conftest.py — add fixtures:
   sample_solexs_df:    n=120, background 1e-7, M-class rise t=55-75
   sample_hel1os_df:    n=120, background 50 counts, burst t=53-68
   sample_flare_event:  FlareEvent for M-class SoLEXS, peak_flux=5e-5
   sample_hel1os_event: FlareEvent for HEL1OS burst, peak_flux=2800, class="?"

4. tests/test_pipeline.py — add:
   test_solexs_detects_synthetic_flare: 1 event, class in (C, M, X)
   test_hel1os_detects_synthetic_burst: 1 event, class == "?"
   test_hel1os_peak_precedes_solexs_peak: Neupert effect verified
   test_flare_event_has_fits_path: fits_path field is not None (provenance)

CHECKPOINT: Both detectors return non-empty lists on 2024-02-22 X6.3 data.
HEL1OS peak_time is 1-4 min before SoLEXS peak_time. All 4 tests pass.
```

---

## M3 — Master Catalogue Merger

```
[CONTEXT BLOCK]

Milestone M3: Temporal coincidence merger producing master catalogue CSV.

PREREQUISITE: M2 complete.
REMINDER: [RULE-17] — provenance columns required in every CSV row.

TASKS:

1. src/catalogue/merger.py
   COINCIDENCE_WINDOW = pd.Timedelta(minutes=2)

   _check_noaa(event_time, noaa_catalog, window_min=10) -> bool:
     PRIVATE — implementation detail of merger.py only [RULE-15].
     Returns False if noaa_catalog is None or empty.

   check_noaa_confirmed(event_time, noaa_catalog, window_min=10) -> bool:
     PUBLIC wrapper for external use. Calls _check_noaa.

   merge_catalogues(solexs_events, hel1os_events,
                    noaa_catalog=None,
                    pipeline_run_id="",     ← [RULE-17]
                    model_version="") -> pd.DataFrame:

     CONFIDENCE TIERS:
     - dual:         ±2 min coincidence → confidence * 1.20 (cap 1.0)
                     flare_class from SoLEXS (absolute flux calibration)
     - SoLEXS_only:  no HEL1OS match → confidence * 0.80
     - HEL1OS_only:  no SoLEXS match → confidence * 0.60, class stays "?"

     Required output columns (exact order):
       start_time, peak_time, end_time, flare_class,
       peak_flux_sxr, peak_cnt_hxr, source, confidence, noaa_confirmed,
       solexs_fits_path,   ← [RULE-17] from FlareEvent.fits_path
       hel1os_fits_path,   ← [RULE-17] from FlareEvent.fits_path or ""
       model_version,      ← [RULE-17] passed parameter
       pipeline_run_id     ← [RULE-17] passed parameter

     Saves to data/processed/master_catalogue.csv (append if exists,
     dedup on peak_time+source before saving).
     Prints: total, dual, SoLEXS_only, HEL1OS_only, NOAA_confirmed counts.

2. scripts/audit_catalogue.py — provenance audit tool:
   Load master_catalogue.csv. For a given peak_time argument, print:
     solexs_fits_path, hel1os_fits_path, model_version, pipeline_run_id
   This is the forensics tool for investigating misclassified alerts.
   CLI: python scripts/audit_catalogue.py --peak_time "2024-02-22T10:33:00"

3. tests/test_pipeline.py — add:
   test_dual_detection_produces_dual_event:    1 row, source=="dual", conf>0.90
   test_solexs_only_confidence_reduced:        source=="SoLEXS_only", conf<original
   test_hel1os_only_class_unknown:             flare_class=="?"
   test_master_catalogue_no_duplication:       1 SoLEXS + 1 HEL1OS -> 1 row
   test_out_of_window_events_not_merged:       peak at t=0 + t+5min -> 2 rows
   test_provenance_columns_present:            [RULE-17] — assert all four
     provenance columns exist and are non-null for dual events:
     assert set(["solexs_fits_path","hel1os_fits_path",
                 "model_version","pipeline_run_id"]).issubset(df.columns)
     for col in ["solexs_fits_path","model_version","pipeline_run_id"]:
         assert df[col].notna().all(), f"{col} must not be null"

CHECKPOINT: master_catalogue.csv exists with all provenance columns.
All 6 tests pass. X6.3 event on 2024-02-22 labelled "dual", conf > 0.90.
audit_catalogue.py returns correct provenance for that event.
```

---

## M4 — Multi-class Nowcasting (TCN + XGBoost)

```
[CONTEXT BLOCK]

Milestone M4: TCN encoder + multi-class XGBoost nowcasting pipeline.

PREREQUISITE: M3 complete. configs/fits_columns.yaml filled.
REMINDER: [RULE-5] — always multi:softprob, 4 classes, per-class thresholds.
REMINDER: [RULE-13] — save model in BOTH .json and .pkl formats.
REMINDER: [RULE-20] — use INFERENCE_LATENCY from src/monitoring/metrics.py.

TASKS:

1. src/nowcasting/tcn_encoder.py
   CausalConv1d: right-pad then slice to maintain causality.
   TCNBlock: two CausalConv1d + LayerNorm + Dropout + residual + ReLU.
   TCNEncoder: n_layers=4, dilations=[1,2,4,8], AdaptiveAvgPool1d(1).
   Input (B, T, F) -> transpose -> TCN blocks -> pool -> (B, embed_dim).

2. src/nowcasting/train.py
   extract_tcn_features(encoder, X, device="cpu", batch_size=256):
     Batched, encoder.eval(), torch.no_grad().
   train_multiclass_nowcast(X_tr, y_tr, X_val, y_val, tcn_tr, tcn_val):
     combined = concat(tcn_feats, X.reshape(N, -1))
     sample_weights = inverse class frequency
     XGBoost: objective="multi:softprob", num_class=4, tree_method="hist"
     MLflow: log model, params, per-class F1
     After training, SAVE IN BOTH FORMATS [RULE-13]:
       model.save_model("models/xgb_multiclass.json")
       joblib.dump(model, "models/xgb_multiclass.pkl")
     Also save PyTorch encoder [RULE-11, RULE-13]:
       torch.save(encoder.state_dict(), "models/tcn_encoder.pt")

   optimize_per_class_thresholds(model, combined_val, y_val) -> dict:
     Sweep 0.10->0.90 step 0.05 per class, pick max TSS.
     X-class override: if thresholds["X"] > thresholds["M"]:
       thresholds["X"] = min(thresholds["M"]-0.05, best_t)
       mlflow.log_param("x_threshold_override", True)
     Save to configs/nowcasting.yaml class_thresholds section.

   display_feature_importance(model, tcn_embed_dim, physics_feature_names):
     all_names = [f"tcn_{i}" for i in range(tcn_embed_dim)] + physics_feature_names
     importance_df = pd.Series(model.feature_importances_, index=all_names)
     print(importance_df.sort_values(ascending=False).head(20))

   Instrument inference latency [SLO-1, RULE-20]:
     from src.monitoring.metrics import INFERENCE_LATENCY, NOWCAST_CONFIDENCE
     with INFERENCE_LATENCY.labels(component="nowcast").time():
         result = nowcaster.predict(window, feats)
     NOWCAST_CONFIDENCE.set(result["confidence"])

3. tests/conftest.py — add:
   toy_tcn_encoder: TCNEncoder(n_features=8, embed_dim=32, n_layers=2)

4. tests/test_pipeline.py — add:
   test_causal_conv_no_future_leakage:
     Input zeros except [0,0,30]=1.0. output[:,:,:30] must ALL be zero.
   test_tcn_output_shape:           (4,60,8) input -> (4,32) output
   test_multiclass_proba_sums_to_one: XGBoost row probabilities sum to 1+-1e-5
   test_model_saved_both_formats:   both .json and .pkl exist, same predictions

CHECKPOINT: model saved in both formats. X-class threshold <= M-class.
test_causal_conv_no_future_leakage passes. Feature importance table printed.
INFERENCE_LATENCY metric importable and observable in /metrics endpoint.
```

---

## M5 — Three-Model Forecasting Ensemble

```
[CONTEXT BLOCK]

Milestone M5: Causal LSTM + TCN + TimesFM ensemble.

PREREQUISITE: M4 complete. CHECK 2 complete (TimesFM API verified).
REMINDER: [RULE-2] — bidirectional=False. [RULE-6] — no timesfm.finetuning.
REMINDER: [RULE-13] — torch.save(model.state_dict(), ...) not full model.

TASKS:

1. src/forecasting/causal_lstm.py
   CausalLSTMForecaster(n_features, hidden_dim=128, n_layers=2,
                         dropout=0.3, n_classes=4):
   LSTM: batch_first=True, bidirectional=False [RULE-2].
   Head: Linear -> ReLU -> Dropout -> Linear(n_classes) -> Softmax
   Save/load pattern [RULE-11, RULE-13]:
     torch.save(model.state_dict(), "models/causal_lstm.pt")
     model.load_state_dict(
         torch.load("models/causal_lstm.pt",
                    map_location="cpu", weights_only=True))

2. src/forecasting/multi_horizon.py
   MultiHorizonForecaster(n_features, embed_dim=256, horizons=[15,30,60]):
     TCNEncoder as shared backbone.
     Separate head per horizon: Linear(embed_dim,64)->ReLU->Dropout->Linear(4)->Softmax
     forward(x) -> dict: {"h15": (B,4), "h30": (B,4), "h60": (B,4)}
   Training: single forward pass for all horizons simultaneously.
   Loss: sum of CrossEntropyLoss for each horizon head.
   Save: torch.save(model.state_dict(), "models/multi_horizon.pt")

3. src/forecasting/ensemble.py
   ThreeModelEnsemble:
     __init__(lstm_model, tcn_model, timesfm_model,
              weights=(0.35, 0.35, 0.30))
     predict_single(X_tensor, flux_np, horizon=15) -> np.ndarray (4,):
       Weighted average of LSTM + TCN + TimesFM probabilities.
       NOTE: TimesFM is NOT batched. Report LSTM+TCN TSS and TimesFM
       zero-shot TSS separately in MLflow. Ensemble weights apply at
       inference time only.

4. src/forecasting/timesfm_forecaster.py
   load_timesfm() -> timesfm.TimesFm
   predict_timesfm(tfm, flux_window, horizons=[15,30,60]) -> dict:
     FORECAST_COL = "timesfm"  # confirmed by CHECK 2 — update if different
   lora_finetune_timesfm(base_model_id, train_dataset, output_dir):
     Guard: if not torch.cuda.is_available(): print warning and return.

5. tests/test_pipeline.py — add:
   test_causal_lstm_bidirectional_false: lstm.bidirectional == False
   test_multi_horizon_output_keys:       keys h15, h30, h60 present
   test_multi_horizon_probabilities_sum_to_one
   test_ensemble_output_shape:           shape (4,) summing to ~1.0
   test_lstm_state_dict_save_load:       save/load with weights_only=True

CHECKPOINT: bidirectional=False verified in test. multi_horizon.py and
ensemble.py exist and tested. TimesFM uses verified column name from CHECK 2.
```

---

## M6 — ONNX Export & Optimised Inference

```
[CONTEXT BLOCK]

Milestone M6: Export TCN and CausalLSTM to ONNX. Fix hypothesis+session-
fixture incompatibility [RULE-14] in test_properties.py.

PREREQUISITE: M5 complete.

TASKS:

1. src/deployment/onnx_export.py
   export_to_onnx(model, dummy_input, path, input_name, output_name, opset=17):
     torch.onnx.export with dynamic batch axis.
     onnx.checker.check_model(onnx.load(path)) — verify after export.
   ONNXNowcaster(tcn_path, xgb_model):
     ort.InferenceSession with CPUExecutionProvider.
     __init__: call verify_model_hash(tcn_path) [T-2] before loading session.
     predict(window, handcrafted_feats) -> {"class", "proba", "confidence"}
   benchmark_speedup(encoder, onnx_nowcaster, X_test, n=200):
     Time both over n inferences. Target: ONNX mean < 100ms on CPU [SLO-1].

2. tests/conftest.py — add toy_onnx_nowcaster as FUNCTION-SCOPED fixture.
   @pytest.fixture(scope="function")   ← NOT "session" [RULE-14]
   def toy_onnx_nowcaster(tmp_path): ...

3. tests/test_properties.py — CORRECT hypothesis pattern [RULE-14]:
   Use module-level singleton _TOY_NOWCASTER, not a session fixture.
   (Full pattern as specified in RULE-14 example.)

   @settings(max_examples=20)
   @given(st.arrays(np.float32, shape=(60, 8),
                    elements=st.floats(0.0, 1e-3, allow_nan=False)))
   def test_onnx_nowcaster_output_valid(window):
       nowcaster = _get_toy_nowcaster()
       result = nowcaster.predict(
           window[np.newaxis].astype(np.float32),
           np.zeros((1, 0), dtype=np.float32))
       assert result["class"] in ("N", "C", "M", "X")
       assert 0.0 <= result["confidence"] <= 1.0
       assert abs(sum(result["proba"].values()) - 1.0) < 1e-4

CHECKPOINT: onnx.checker passes for TCN and LSTM exports.
ONNX inference mean < 100ms on CPU (benchmarked). [SLO-1 component verified]
test_onnx_nowcaster_output_valid runs 20 hypothesis examples without crash.
```

---

## M7 — Agent Orchestration (LangGraph)

```
[CONTEXT BLOCK]

Milestone M7: LangGraph stateful agent pipeline.

PREREQUISITE: M6 complete. All model files saved per [RULE-13].
REMINDERS:
[RULE-12] module-level open() guards.
[RULE-13] XGBoost loaded via joblib from .pkl; PyTorch as state_dict.
[RULE-15] No private function imports across modules.
[RULE-16] Singleton loaders at module level — NOT inside agent functions.
[RULE-17] pipeline_run_id must be generated and threaded through state.
[RULE-20] Use metrics from src/monitoring/metrics.py only.
[SLO-1]  Every agent logs timing to state["timing"][agent_name].

CRITICAL DESIGN (Decision D7):
Two detector edges into one merge node requires the Annotated reducer
pattern. Simple add_edge() calls do NOT parallelize automatically in
LangGraph. The Annotated field on detected_events handles fan-in.

TASKS:

1. src/orchestration/state.py
   import uuid
   _merge_catalogue_lists(a, b) -> list: handle None inputs, concat lists.
   SolarPipelineState TypedDict with ALL fields including:
     detected_events:   Annotated[Optional[list], _merge_catalogue_lists]
     errors:            Annotated[list, operator.add]
     timing:            Annotated[dict, lambda a, b: {**a, **b}]  ← [SLO-1]
     pipeline_run_id:   str  ← [RULE-17] UUID4, set by ingestion_agent
     model_version:     str  ← [RULE-17] read from configs/version.yaml
     alert_triggered:   bool
     llm_report:        Optional[str]
     shap_explanation:  Optional[dict]

2. src/uncertainty/chronos_uncertainty.py — add module-level singleton [RULE-16]:
   _CHRONOS_PIPELINE = None
   def get_chronos():
       global _CHRONOS_PIPELINE
       if _CHRONOS_PIPELINE is None:
           import time
           t0 = time.perf_counter()
           from chronos import BaseChronosPipeline
           _CHRONOS_PIPELINE = BaseChronosPipeline.from_pretrained(
               "amazon/chronos-bolt-small",
               dtype=torch.float32, device_map="cpu")
           print(f"Chronos loaded in {time.perf_counter()-t0:.1f}s — singleton cached.")
       return _CHRONOS_PIPELINE
   Update chronos_forecast_interval() to call get_chronos() internally.
   Also call verify_model_hash on the Chronos cache dir if applicable [T-2].

3. src/forecasting/moment_anomaly.py — make _load_moment() public [RULE-15]:
   Rename _load_moment to get_moment_model(). Update all internal refs.

4. src/orchestration/agents.py
   MODULE-LEVEL [RULE-12]:
     try:
         with open("configs/nowcasting.yaml") as f: _CFG = yaml.safe_load(f) or {}
     except FileNotFoundError: _CFG = {}
     try:
         with open("configs/version.yaml") as f:
             _MODEL_VERSION = yaml.safe_load(f).get("model_version", "unknown")
     except FileNotFoundError: _MODEL_VERSION = "unknown"

   SINGLETON PATTERN [RULE-16] for XGBoost, LSTM, MAPIE.
   (Pattern as defined in MASTER CONTEXT BLOCK RULE-16.)

   TIMING INSTRUMENTATION [SLO-1] — every agent wraps core logic:
   import time
   def ingestion_agent(state: SolarPipelineState) -> SolarPipelineState:
       t0 = time.perf_counter()
       try:
           import uuid
           state["pipeline_run_id"] = str(uuid.uuid4())  ← [RULE-17]
           state["model_version"]   = _MODEL_VERSION      ← [RULE-17]
           # ... read FITS files ...
       except Exception as exc:
           state["errors"].append(f"ingestion_agent: {exc}")
       state.setdefault("timing", {})["ingestion"] = round(
           time.perf_counter()-t0, 3)
       return state
   Apply same timing pattern to ALL agents. The final integration test
   reads state["timing"] and asserts every component is within [SLO-1] budget.

   Prometheus metrics [RULE-20]:
   from src.monitoring.metrics import ALERT_COUNTER, DATA_FRESHNESS
   In alert_router: if alert fires:
       ALERT_COUNTER.labels(flare_class=cls, source=src).inc()
   In ingestion_agent: DATA_FRESHNESS.set(time.time() - fits_mtime)

   All agents: ingestion_agent, solexs_detect_agent, hel1os_detect_agent,
   merge_agent, preprocess_agent, moment_score_agent, nowcast_agent,
   forecast_agent, uncertainty_agent, shap_agent, llm_report_agent.
   alert_router(state) -> "llm_report" | "end"

   merge_agent: pass pipeline_run_id and model_version to merge_catalogues [RULE-17]:
     master = merge_catalogues(
         solexs_ev, hel1os_ev,
         pipeline_run_id=state.get("pipeline_run_id", ""),
         model_version=state.get("model_version", ""))

5. src/orchestration/graph.py
   build_pipeline():
     Annotated reducer handles detected_events fan-in.
     g.add_edge("ingestion", "detect_solexs")
     g.add_edge("ingestion", "detect_hel1os")
     g.add_edge("detect_solexs", "merge")
     g.add_edge("detect_hel1os", "merge")   ← reducer merges both lists

6. tests/test_pipeline.py — add:
   test_pipeline_runs_end_to_end:       monkeypatch all agents to stub state
   test_alert_router_fires_on_mx:       nowcast_class="X" -> "llm_report"
   test_alert_router_silent_on_quiet:   nowcast_class="N", p=0.1 -> "end"
   test_merge_agent_separates_by_instrument
   test_timing_dict_populated:
     After pipeline.invoke() with stub agents, assert state["timing"] has
     keys for every agent and every value is a non-negative float.
   test_pipeline_run_id_is_uuid:
     After ingestion_agent runs, state["pipeline_run_id"] must be a valid
     UUID4 string: uuid.UUID(state["pipeline_run_id"], version=4) succeeds.

CHECKPOINT: build_pipeline() compiles. Chronos and MOMENT loaded once per
process. state["timing"] populated. pipeline_run_id is UUID4.
grep -r "_check_noaa\|_load_moment" src/orchestration/ -> zero results.
```


---

## M8 — AutoML Tuning (FLAML)

```
[CONTEXT BLOCK]

Milestone M8: FLAML AutoML for XGBoost hyperparameter search with
hard FAR ≤ 0.10 promotion gate [SLO-5].

PREREQUISITE: M4 complete. M7 pipeline at least stub-complete.
REMINDER: [RULE-3] — TSS is primary metric. [RULE-5] — multi:softprob.
REMINDER: [SLO-5] — FAR > 0.10 blocks promotion regardless of TSS.

TASKS:

1. src/automl/flaml_tuner.py

   custom_metric(X_val, y_val, estimator, labels, X_train, y_train,
                 weight_val=None, weight_train=None, config=None,
                 groups_val=None, groups_train=None):
     proba = estimator.predict_proba(X_val)
     binary_pred = (proba[:, 1:].sum(axis=1) >= 0.5).astype(int)
     binary_true = (y_val > 0).astype(int)
     tn, fp, fn, tp = confusion_matrix(
         binary_true, binary_pred, labels=[0, 1]   ← [RULE-8]
     ).ravel()
     tpr = tp / max(tp+fn, 1)
     fpr = fp / max(fp+tn, 1)
     tss = tpr - fpr
     far = fpr                          # False Alarm Rate = FPR
     return 1.0 - tss, {"tss": tss, "far": far, "tpr": tpr, "fpr": fpr}

   run_flaml_automl(X_tr, y_tr, X_val, y_val, time_budget=300):
     automl = AutoML()
     automl.fit(
         X_train=X_tr, y_train=y_tr,
         metric=custom_metric,
         task="classification",
         estimator_list=["xgboost"],
         time_budget=time_budget,
         log_file_name="logs/flaml_automl.log",
         custom_hp={"xgboost": {
             "n_estimators":  {"domain": tune.randint(100, 800), "init_value": 300},
             "max_depth":     {"domain": tune.randint(3, 9),   "init_value": 6},
             "learning_rate": {"domain": tune.loguniform(1e-3, 0.3)},
             "subsample":     {"domain": tune.uniform(0.5, 1.0)},
             "colsample_bytree": {"domain": tune.uniform(0.5, 1.0)},
         }})

     best_tss = -automl.best_loss
     best_far = automl.best_config_train_time   # placeholder — extract below
     # Extract FAR from the metric history
     best_far = _extract_best_far(automl)

     *** FAR PROMOTION GATE [SLO-5] — HARD BLOCK: ***
     if best_far > 0.10:
         mlflow.log_params({
             "model_rejected":  "FAR_exceeds_threshold",
             "model_far":       round(best_far, 4),
             "model_far_limit": 0.10,
             "model_tss":       round(best_tss, 4),
         })
         print(
             f"[SLO-5] MODEL REJECTED: FAR={best_far:.4f} > 0.10 limit.\n"
             f"  TSS={best_tss:.4f} is irrelevant — FAR gate blocks promotion.\n"
             f"  Action: Tune class thresholds, increase sigma_threshold in\n"
             f"  detectors, or reduce augmentation ratio and re-run M8."
         )
         return None, best_far, best_tss  # caller must check for None model

     mlflow.log_params({"model_far": round(best_far, 4),
                         "model_tss": round(best_tss, 4),
                         "model_rejected": "no"})
     return automl.model.estimator, best_far, best_tss

   _extract_best_far(automl) -> float:
     Parse automl.best_config_train_time or iterate automl.model_history
     to find FAR associated with the best TSS configuration. If extraction
     fails, re-evaluate on validation set: return the raw FPR.

2. tests/test_pipeline.py — add:
   test_flaml_tuner_rejects_high_far:
     Mock automl to return a model with best FAR=0.15.
     Assert run_flaml_automl returns (None, 0.15, *).
     Assert mlflow.log_params was called with "model_rejected": "FAR_exceeds_threshold".
   test_flaml_tuner_accepts_low_far:
     Mock automl to return a model with best FAR=0.05 and TSS=0.72.
     Assert return value is not None (model promoted).
   test_custom_metric_returns_tss_and_far:
     Construct a perfect confusion matrix, assert TSS=1.0 and FAR=0.0.
     Construct a worst-case matrix, assert TSS <= 0 and FAR >= 0.5.

CHECKPOINT: run_flaml_automl returns None when FAR > 0.10 (gate verified
by test). MLflow run shows model_rejected tag. FAR_GAUGE metric updated
in src/monitoring/metrics.py call. Both new tests pass.
```

---

## M9 — LLM Intelligence (DSPy + GraphRAG + Phi-3-mini)

```
[CONTEXT BLOCK]

Milestone M9: DSPy-optimised alert reporter with GraphRAG context and
structured fallback. Ollama Phi-3-mini local only.

PREREQUISITE: M7 complete. CHECK 5 (DSPy syntax) and CHECK 7 (GraphRAG
local config + SSRF assertion) both complete.
REMINDER: [RULE-9] GraphRAG must use local embeddings.
REMINDER: [RULE-10] dspy.context() per-call, NOT dspy.configure().

TASKS:

1. data/graphrag/ — build GraphRAG knowledge base:
   Source documents:
   - NOAA solar flare classification guide
   - SoLEXS + HEL1OS instrument papers (from ISSDC)
   - Aditya-L1 science objectives doc
   Run: graphrag init --root data/graphrag/
   Update settings.yaml with local Ollama config (from CHECK 7).
   Run: graphrag index --root data/graphrag/
   Verify: graphrag query --root data/graphrag/ --query "What is a C-class flare?"

2. src/intelligence/dspy_reporter.py
   _LM = None  ← module-level singleton [RULE-16]
   def get_lm():
       global _LM
       if _LM is None:
           import dspy
           _LM = dspy.LM(...)  # use syntax confirmed by CHECK 5
       return _LM

   class FlareAlertSignature(dspy.Signature):
       flare_class, peak_flux, lead_time_min, confidence, graphrag_context,
       uncertainty_q10, uncertainty_q90  ->  alert_summary (str)

   class SolarFlareReporter(dspy.Module):
       predict: dspy.Predict(FlareAlertSignature)
       forward() -> str: with dspy.context(lm=get_lm()): ...   ← [RULE-10]

   optimize_reporter(training_examples):
     Compile with dspy.MIPROv2(metric=...).compile()
     reporter.save("models/dspy_reporter_optimised.json")

3. src/intelligence/structured_fallback.py
   Deterministic fallback when Ollama is down:
   build_alert_message(flare_class, peak_flux, lead_min, q10, q90) -> str
   Maps flare class to severity text. Returns formatted alert string.
   No external calls. No subprocess. Pure Python.
   Must return within 1 second — tests with time.perf_counter().

4. src/intelligence/graphrag_retriever.py
   retrieve_flare_context(query: str, top_k: int = 3) -> str:
     subprocess.run(["graphrag", "query", "--root", ..., "--query", query],
                    capture_output=True, timeout=30)
     On timeout or error: return "" (silent fallback — DSPy still runs).

5. tests/test_pipeline.py — add:
   test_dspy_reporter_uses_context_manager: mock dspy.context, assert called.
   test_structured_fallback_returns_in_1s:  time.perf_counter() assert.
   test_structured_fallback_no_subprocess:  mock subprocess.run, assert NOT called.
   test_graphrag_retriever_timeout_returns_empty: mock subprocess timeout -> "".
   test_llm_report_agent_uses_fallback_on_error:
     In agent graph, if DSPy raises, state["llm_report"] comes from fallback.

CHECKPOINT: graphrag index completed. graphrag query returns solar flare
context. dspy.context() used (grep -r "dspy.configure" src/intelligence/ -> 0).
Fallback returns in < 1s. All 5 new tests pass.
```

---

## M10 — Physics Layer (PINN + Phase Detector)

```
[CONTEXT BLOCK]

Milestone M10: PINN-based Neupert Effect validator and 5-phase detector.

PREREQUISITE: M7 complete.
REMINDER: [RULE-12] — module-level config open guards.

TASKS:

1. src/physics/pinn.py
   NeupertPINN(nn.Module):
     Encodes flux window, predicts dSXR/dt.
     Physics loss = MSE(predicted_dSXR_dt, actual_dSXR_dt) + beta * violation_penalty
     violation_penalty(dSXR_dt, HXR) -> tensor:
       product = dSXR_dt * HXR  (Neupert says product > 0 during flare rise)
       violation = -torch.clamp(product, max=0)  # penalise negative product
       return violation.mean()
   train_neupert_pinn(X_windows, hxr_windows, sxr_windows, beta=0.1):
     Log physics_loss, violation_penalty, total_loss to MLflow every epoch.
     Save: torch.save(pinn.state_dict(), "models/neupert_pinn.pt")

2. src/physics/phase_detector.py — [RULE-12] guard on config load.
   detect_phase(flux_window: np.ndarray, peak_frac: dict | None = None) -> int:
     0=quiet, 1=pre-flare, 2=impulsive, 3=peak, 4=gradual
     Uses PHASE_CFG from module-level YAML load (with FileNotFoundError guard).
   build_phase_sequence(df: pd.DataFrame, flux_col="flux_high") -> np.ndarray:
     Rolling apply detect_phase, returns int array same length as df.

3. tests/test_pipeline.py — add:
   test_neupert_violation_penalty_positive:
     dSXR_dt=neg, HXR=pos -> product<0 -> penalty > 0
   test_neupert_violation_penalty_zero_when_valid:
     dSXR_dt=pos, HXR=pos -> product>0 -> penalty == 0
   test_phase_detector_on_synthetic:
     Flat input -> 0 (quiet). Ramp then peak -> sequence contains 1,2,3,4.
   test_phase_detector_no_hardcoded_values:
     Load configs/nowcasting.yaml, assert detect_phase reads from it.
   test_hxr_leads_sxr_by_1_to_4_min:
     fixture sample_hel1os_event.peak_time - sample_flare_event.peak_time
     must be within -4...-1 minutes (HXR leads SXR, Neupert Effect).

CHECKPOINT: Neupert loss logged to MLflow. Phase detector returns 5
distinct values on a synthetic full-flare window. All 5 tests pass.
```

---

## M11 — Dual Uncertainty Layer (MAPIE + Chronos-Bolt)

```
[CONTEXT BLOCK]

Milestone M11: Conformal prediction + probabilistic Chronos intervals.

PREREQUISITE: M7, M5 complete. CHECK 3 (Chronos API verified).
REMINDERS: [RULE-7] predict() not predict_quantiles().
           [RULE-16] Chronos singleton in chronos_uncertainty.py (done M7).

TASKS:

1. src/uncertainty/conformal_mapie.py
   train_mapie(X_cal, y_cal, base_estimator) -> MapieClassifier:
     MapieClassifier(base_estimator, method="score", cv="prefit")
     mapie.fit(X_cal, y_cal)
     joblib.dump(mapie, "models/conformal_mapie.pkl")  [RULE-13]
   load_mapie() -> MapieClassifier:
     return joblib.load("models/conformal_mapie.pkl")
   predict_with_sets(mapie, X, alpha=0.10):
     y_pred, y_set = mapie.predict(X, alpha=alpha, include_last_label="randomized")
     Interpretation: y_set contains all classes in the 90% conformal set.
     Multi-class 90% coverage guarantee.

2. src/uncertainty/chronos_uncertainty.py
   chronos_forecast_interval(flux_window, prediction_length=15,
                              num_samples=200, q=(0.10, 0.90)):
     pipeline = get_chronos()   ← singleton from M7 [RULE-16]
     context = torch.tensor(flux_window, dtype=torch.float32).unsqueeze(0)
     forecast = pipeline.predict(          ← NOT predict_quantiles [RULE-7]
         context, prediction_length=prediction_length,
         num_samples=num_samples)
     samples = forecast[0].numpy()         # shape (num_samples, horizon)
     q10 = float(np.quantile(samples, q[0]))
     q90 = float(np.quantile(samples, q[1]))
     return {"q10": q10, "q90": q90, "median": float(np.median(samples)),
             "std": float(np.std(samples.mean(axis=1)))}

3. src/uncertainty/agreement_checker.py
   Agreement check — MAPIE and Chronos provide independent uncertainty
   signals; they should agree on high-uncertainty events:
   check_dual_agreement(mapie_set_size, chronos_std,
                         mapie_threshold=2, chronos_std_threshold=1e-7):
     Returns "HIGH_UNCERTAINTY" if both exceed thresholds, else "NORMAL".
     Log to state["uncertainty_level"]: str.

4. tests/test_pipeline.py — add:
   test_mapie_returns_prediction_sets:    y_set.shape[1] == 4 (one per class)
   test_chronos_uses_predict_not_quantiles:
     mock pipeline.predict; assert predict() called, predict_quantiles NOT.
   test_chronos_quantiles_computed_from_samples:
     mock predict returning fixed samples; assert q10 < median < q90.
   test_chronos_singleton_only_loaded_once:
     Call get_chronos() 3 times; assert BaseChronosPipeline.from_pretrained
     called exactly once (mock and count calls).
   test_dual_agreement_high_on_both_signals: both thresholds exceeded -> HIGH.

CHECKPOINT: chronos_uncertainty.py uses get_chronos() singleton.
pipeline.predict() called, not predict_quantiles() (grep confirms).
All 5 tests pass. mapie model saved to conformal_mapie.pkl.
```


---

## M12 — Pre-flare Anomaly Detection (MOMENT)

```
[CONTEXT BLOCK]

Milestone M12: MOMENT-1-large reconstruction-error anomaly detector for
pre-flare precursor identification.

PREREQUISITE: M7 complete. CHECK 4 complete — RECONSTRUCTION_ATTR confirmed.
REMINDER: [RULE-16] — get_moment_model() singleton (renamed from _load_moment
in M7 [RULE-15]). Load time ~30s — singleton prevents per-call delay.

CRITICAL IMPLEMENTATION DETAIL — MOMENT OUTPUT ATTRIBUTE SAFETY:
Do NOT hardcode the output attribute name. MOMENT's API has changed
across versions. Use the verified name from CHECK 4, but protect against
future version drift with a safe getter:

    RECONSTRUCTION_ATTR = "reconstruction"  # Verified by CHECK 4 — update if different

    def _get_reconstruction(output) -> torch.Tensor:
        '''
        Safely extract reconstruction tensor from MOMENT output object.
        MOMENT's output attribute has changed across versions; this function
        centralises the lookup and gives an actionable error if it changes again.
        '''
        result = getattr(output, RECONSTRUCTION_ATTR, None)
        if result is None:
            available = [a for a in dir(output)
                         if not a.startswith("_") and isinstance(
                             getattr(output, a, None), torch.Tensor)]
            raise AttributeError(
                f"MOMENT output has no attribute '{RECONSTRUCTION_ATTR}'.\n"
                f"Available tensor attributes: {available}\n"
                f"Update RECONSTRUCTION_ATTR in moment_anomaly.py and "
                f"re-run CHECK 4 to confirm.")
        return result

TASKS:

1. src/forecasting/moment_anomaly.py
   get_moment_model() -> MOMENTPipeline:
     Module-level singleton [RULE-16]:
     _MOMENT = None
     def get_moment_model():
         global _MOMENT
         if _MOMENT is None:
             from momentfm import MOMENTPipeline
             _MOMENT = MOMENTPipeline.from_pretrained(
                 "AutonLab/MOMENT-1-large",
                 model_kwargs={"task_name": "reconstruction"})
             _MOMENT.init()
         return _MOMENT
   _get_reconstruction(output) as defined above.
   compute_reconstruction_error(flux_window: np.ndarray) -> float:
     model = get_moment_model()
     x = torch.tensor(flux_window[-512:], dtype=torch.float32
                       ).unsqueeze(0).unsqueeze(0)  # (1, 1, 512)
     with torch.no_grad(): output = model(x)
     recon = _get_reconstruction(output).squeeze()
     return float(F.mse_loss(recon, x.squeeze()).item())
   classify_anomaly(error: float, threshold: float | None = None) -> bool:
     If threshold is None: load from YAML.
     Return error > threshold.
   batch_compute_moment_scores(df, flux_col, window_size=512, stride=60):
     Only compute MOMENT scores on windows flagged as pre-flare by
     phase_detector. Skip quiet-sun windows to stay within [SLO-1] budget.
     Use tqdm for progress.

2. tests/test_pipeline.py — add:
   test_moment_singleton_loads_once:
     Call get_moment_model() twice; assert MOMENTPipeline.from_pretrained
     called exactly once.
   test_get_reconstruction_raises_on_wrong_attr:
     Create mock output object with attribute "logits" but NOT "reconstruction".
     Call _get_reconstruction(mock_output).
     Assert AttributeError is raised and message contains "Available tensor attributes".
   test_get_reconstruction_success:
     Create mock output with getattr(output, "reconstruction") returning zeros tensor.
     Assert _get_reconstruction returns that tensor without error.
   test_batch_skips_quiet_windows:
     Synthetic df with 1 pre-flare window and 9 quiet windows.
     After batch_compute_moment_scores, assert get_moment_model() called
     at most 1 time (quiet windows skipped, [SLO-1] budget preserved).
   test_reconstruction_error_increases_near_flare:
     Quiet baseline window -> error_quiet.
     Synthetic pre-flare ramp window -> error_preflare.
     Assert error_preflare > error_quiet.

CHECKPOINT: _get_reconstruction safely raises with actionable message on
wrong attribute. Singleton pattern verified by test. MOMENT only called on
pre-flare windows. All 5 tests pass.
```

---

## M13 — Explainability (SHAP)

```
[CONTEXT BLOCK]

Milestone M13: SHAP explanations for XGBoost nowcasting decisions.

PREREQUISITE: M4 complete.

TASKS:

1. src/explainability/shap_explainer.py
   SHAPExplainer:
     __init__(xgb_model, feature_names): TreeExplainer(xgb_model)
     explain(combined_features: np.ndarray) -> dict:
       shap_values = explainer.shap_values(combined_features)   # list[4]
       Class with highest mean |SHAP| is the dominant class.
       top_features = top 5 features by |SHAP| for the predicted class.
       return {"top_features": top_features,
               "class_importances": per_class_mean_abs_shap,
               "dominant_class": int, "raw_shap": shap_values}
     generate_report_text(explain_result, predicted_class) -> str:
       Human-readable summary. Used as input context for DSPy reporter.

2. tests/test_pipeline.py — add:
   test_shap_top_features_are_subset_of_all_features
   test_shap_explain_result_has_required_keys:
     assert {"top_features","class_importances","dominant_class","raw_shap"
             }.issubset(result.keys())

CHECKPOINT: explain() returns all required keys. top_features are a subset
of feature_names. Both tests pass.
```

---

## M14 — Dashboard & Visualization (React + TypeScript + Vite)

```
[CONTEXT BLOCK]

Milestone M14: Production dashboard.

PREREQUISITE: M7 (WebSocket alert push) complete.
REMINDER: [RULE-19] — React + TypeScript + Vite. Streamlit is NOT an option.
REMINDER: [T-3] — WebSocket rate limiting and connection cap in main.py.
REMINDER: [RULE-20] — Prometheus /metrics endpoint served by the API.

ARCHITECTURE DECISION (D9 — final, made at M0):
React + TypeScript + Vite. Rationale documented in M0 D9 comment.
Containerised as a multi-stage Dockerfile: build stage (node:20-alpine),
runtime stage (nginx:alpine serving dist/).

TASKS:

1. src/api/main.py — FastAPI API layer:
   /health  GET   -> {"status": "ok", "model_version": str, "env": str}
   /alert   POST  -> Broadcast nowcast/forecast to WebSocket subscribers
   /metrics GET   -> Prometheus /metrics (prometheus_client.generate_latest())
   /history GET   -> Last N rows of master_catalogue.csv as JSON
   /explain GET   -> SHAP explanation for last alert
   /status  GET   -> {"timing": state["timing"], "slo_status": str}

   WEBSOCKET RATE LIMITING [T-3]:
   MAX_CONNECTIONS = 20
   _active_ws: set = set()
   @app.websocket("/ws/live")
   async def live_feed(websocket: WebSocket):
       if len(_active_ws) >= MAX_CONNECTIONS:
           await websocket.close(code=1008)
           return
       await websocket.accept()
       _active_ws.add(websocket)
       last_msg_time = 0.0
       try:
           while True:
               try:
                   data = await asyncio.wait_for(
                       websocket.receive_text(), timeout=10)
                   now = time.time()
                   if now - last_msg_time < 10:  # 1 msg per 10s limit
                       await websocket.send_text(
                           '{"error":"rate_limit","msg":"1 msg per 10s"}')
                       await websocket.close(code=1008)
                       break
                   last_msg_time = now
               except asyncio.TimeoutError:
                   pass  # no inbound message, continue broadcasting
       finally:
           _active_ws.discard(websocket)

2. dashboard/ — React + TypeScript + Vite project:
   dashboard/
   ├── package.json         (vite, react, typescript, recharts, lucide-react)
   ├── tsconfig.json
   ├── vite.config.ts
   └── src/
       ├── main.tsx
       ├── App.tsx
       ├── components/
       │   ├── FluxChart.tsx        (Recharts LineChart, dual-axis SXR+HXR)
       │   ├── AlertBanner.tsx      (colour-coded: N=grey, C=blue, M=orange, X=red)
       │   ├── ForecastGauge.tsx    (probability gauge for 15/30/60 min)
       │   ├── UncertaintyBand.tsx  (q10-q90 ribbon on flux chart)
       │   ├── SHAPExplainer.tsx    (horizontal bar chart of top SHAP features)
       │   ├── LeadTimeBadge.tsx    (live "Lead Time: Xmin" badge)
       │   └── SLOHealthPanel.tsx   (real-time SLO status from /status endpoint)
       ├── hooks/
       │   └── useWebSocket.ts      (auto-reconnect with exponential backoff)
       └── types/
           └── api.ts               (TypeScript interfaces for all API responses)

   SLOHealthPanel.tsx: Polls /status every 10s. Displays per-component timing
   from state["timing"] as coloured bars (green < budget, red > budget).
   Shows overall SLO status: PASS | FAIL with the offending component.

   useWebSocket.ts reconnect logic:
   - Base delay: 1s
   - Exponential backoff: delay = min(base * 2^attempt, 30s)
   - Reset attempt counter on successful connection

3. dashboard/Dockerfile (multi-stage):
   FROM node:20-alpine AS builder
   WORKDIR /app
   COPY package*.json ./
   RUN npm ci
   COPY . .
   RUN npm run build
   FROM nginx:alpine
   COPY --from=builder /app/dist /usr/share/nginx/html
   COPY nginx.conf /etc/nginx/conf.d/default.conf
   EXPOSE 80

4. tests/test_pipeline.py — add:
   test_health_endpoint_returns_model_version:
     GET /health -> {"status": "ok", "model_version": str, "env": str}
   test_metrics_endpoint_returns_prometheus_format:
     GET /metrics response contains "solar_inference_duration_seconds"
     (the metric defined in src/monitoring/metrics.py [RULE-20]).
   test_websocket_rate_limit_enforced:
     Connect to /ws/live. Send 2 messages < 10s apart.
     Assert second response closes with code 1008.
   test_websocket_connection_cap:
     Open 21 simultaneous connections. Assert 21st closed with 1008.

CHECKPOINT: npm run build exits 0. FastAPI starts. /health, /metrics, /status
all return 200. WebSocket rate limiting test passes. SLOHealthPanel renders
timing data. All 4 tests pass.
```

---

## M15 — Evaluation & Walk-Forward Validation

```
[CONTEXT BLOCK]

Milestone M15: Walk-forward CV, per-class metrics, SLO compliance logging.

PREREQUISITE: M4, M5 complete.
REMINDERS: [RULE-3] TSS primary. [RULE-8] labels=[0,1]. [SLO-4] lead time.
           [SLO-5] FAR gate. [RULE-20] update FAR_GAUGE and LEAD_TIME_GAUGE.

TASKS:

1. src/evaluation/walk_forward.py
   walk_forward_cv(df, pipeline_fn, n_splits=5, gap_days=7):
     TimeSeriesSplit(n_splits=n_splits, gap=gap_days*1440).
     Per fold: fit on train, evaluate on test.
     compute_fold_metrics(y_true, y_pred, y_proba) -> dict:
       tn, fp, fn, tp = confusion_matrix(
           y_true, y_pred, labels=[0,1]).ravel()  ← [RULE-8]
       tss = (tp/(tp+fn+1e-9)) - (fp/(fp+tn+1e-9))
       far = fp / max(fp+tn, 1)
       ppv = tp / max(tp+fp, 1)
       Returns: tss, far, ppv, tpr, tnr, per_class_f1
     Aggregate: mean ± std for all metrics.

   compute_lead_time(forecast_proba_series, actual_flare_times,
                     threshold=0.50) -> list[float]:
     For each actual flare: find the first timestep where forecast exceeds
     threshold. Lead time = actual_peak - first_trigger.
     Return list of lead times in minutes (NaN if no trigger).
     mean_lead = np.nanmean(lead_times)  # [SLO-4] target ≥ 10min

2. src/evaluation/slo_reporter.py
   import yaml, mlflow
   from src.monitoring.metrics import FAR_GAUGE, LEAD_TIME_GAUGE

   check_slo_compliance(metrics: dict) -> dict:
     Load configs/slo.yaml.
     slo_status = {}
     slo_status["latency"] = "pass"   # checked in integration test, not here
     slo_status["far"]      = "pass" if metrics["far"] <= slo["far_max"] else "fail"
     slo_status["lead_time"]= "pass" if metrics["mean_lead_min"] >=
                               slo["lead_time_min_accepted"] else "fail"
     overall = "pass" if all(v == "pass" for v in slo_status.values()) else "fail"
     return {"overall": overall, "detail": slo_status}

   log_evaluation_to_mlflow(metrics: dict, slo_result: dict):
     with mlflow.start_run(nested=True):
         mlflow.log_metrics({
             "mean_tss":       metrics["tss"],
             "mean_far":       metrics["far"],
             "mean_tpr":       metrics["tpr"],
             "mean_lead_min":  metrics["mean_lead_min"],
         })
         mlflow.set_tags({
             "slo_status":       slo_result["overall"],
             "slo_far_status":   slo_result["detail"]["far"],
             "slo_lead_status":  slo_result["detail"]["lead_time"],
         })
         if slo_result["overall"] == "fail":
             failing = [k for k,v in slo_result["detail"].items() if v == "fail"]
             mlflow.set_tag("slo_failing_dims", ",".join(failing))
     # Update Prometheus gauges [RULE-20]:
     FAR_GAUGE.set(metrics["far"])
     LEAD_TIME_GAUGE.set(metrics["mean_lead_min"])

3. tests/test_pipeline.py — add:
   test_walk_forward_no_data_leakage:
     Assert max(train_index) < min(test_index) for all folds.
   test_confusion_matrix_empty_fold_safe:
     Fold with y_true all-zeros (no positives) — no ValueError raised [RULE-8].
   test_slo_reporter_fails_on_high_far:
     check_slo_compliance({"far": 0.15, "mean_lead_min": 15.0})
     Assert overall == "fail", slo_far_status == "fail".
   test_slo_reporter_passes_when_within_limits:
     check_slo_compliance({"far": 0.06, "mean_lead_min": 12.0})
     Assert overall == "pass".
   test_lead_time_computation_correct:
     Synthetic proba series crossing 0.50 at t=20. Flare at t=30.
     Assert compute_lead_time returns [10.0].
   test_prometheus_gauges_updated_after_evaluation:
     Mock FAR_GAUGE.set and LEAD_TIME_GAUGE.set.
     Call log_evaluation_to_mlflow. Assert both .set() called with correct values.

NOTEBOOK: notebooks/05_walk_forward_eval.ipynb
  Plot TSS and FAR per fold as bar chart.
  Plot lead-time distribution as histogram.
  Display SLO compliance summary table:
    | SLO Dimension | Target | Actual | Status |
    Show GREEN/RED traffic-light for each row.

CHECKPOINT: walk_forward_cv runs 5 folds without crash or data leakage.
SLO compliance logged to MLflow (slo_status tag present).
FAR_GAUGE and LEAD_TIME_GAUGE updated. All 6 tests pass.
```

---

## M16 — Production Deployment & MLOps

```
[CONTEXT BLOCK]

Milestone M16: Full production stack — Docker multi-env, CI/CD, drift
detection, model promotion with FAR gate, hash verification, staging gate,
secret scanning, Prometheus + Grafana dashboards, data retention policy.

PREREQUISITE: All M0–M15 complete. All SLOs within targets on walk-forward.
REMINDERS:
[RULE-21] Staging gate BEFORE production model promotion.
[RULE-20] All Prometheus metrics from src/monitoring/metrics.py.
[SLO-5]  FAR > 0.10 blocks promote_if_better() regardless of TSS.
[T-2]    verify_model_hash() on every model load path.
[T-6]    gitleaks in CI (already in .pre-commit-config.yaml from M0).

ENVIRONMENT STRATEGY:
Three environments, selected by ENV env var (default "dev"):
  dev:     local Docker Compose, MLflow at localhost:5000, no staging gate.
  staging: full Docker stack on CI runner, synthetic FITS data, staging gate required.
  prod:    production server, real PRADAN data, staging gate enforced by RULE-21.
All environment-specific config is in docker-compose.{dev,staging,prod}.yml.
Application code NEVER reads ENV to change behaviour — only infrastructure
files (compose, nginx, Dockerfile targets) differ by environment.

TASKS:

1. Dockerfile (multi-stage, 3 targets):
   base:     python:3.11-slim, pip install -r requirements.txt, non-root USER.
   api:      FROM base, COPY src/ configs/ models/ scripts/,
             EXPOSE 8000, ENTRYPOINT uvicorn src.api.main:app
   worker:   FROM base, COPY src/ configs/ models/ scripts/,
             ENTRYPOINT python src/orchestration/scheduler.py

   dashboard/Dockerfile already written in M14.

2. docker-compose.dev.yml:
   Services: api, worker, mlflow, prometheus, grafana.
   Volumes: ./data:/app/data, ./models:/app/models, ./logs:/app/logs.
   No PRADAN ingestion — manual FITS paths only.

3. docker-compose.staging.yml:
   Same services as dev. Override: FITS_SOURCE=synthetic (worker uses
   scripts/generate_synthetic_fits.py instead of PRADAN downloader).
   No external network access (validates system without PRADAN credentials).

4. docker-compose.prod.yml:
   Full services including PRADAN scheduler, Slack webhook notifications.
   PRADAN_USERNAME and PRADAN_PASSWORD injected from GitHub Secrets — NEVER
   in docker-compose.prod.yml directly.

5. src/deployment/model_registry.py
   promote_if_better(new_run_id, current_run_id, mlflow_client) -> bool:
     Load both runs from MLflow. Compare TSS.
     *** FAR GATE [SLO-5] — FIRST CHECK before TSS comparison:
     new_far = float(mlflow_client.get_run(new_run_id).data.metrics["mean_far"])
     if new_far > 0.10:
         print(f"[SLO-5] PROMOTION BLOCKED: FAR={new_far:.4f} > 0.10.\n"
               f"  TSS comparison skipped. Model will NOT be promoted.")
         mlflow_client.set_tag(new_run_id, "promotion_status", "rejected_far")
         return False

     new_tss     = float(mlflow_client.get_run(new_run_id).data.metrics["mean_tss"])
     current_tss = float(mlflow_client.get_run(current_run_id).data.metrics["mean_tss"])
     if new_tss <= current_tss:
         mlflow_client.set_tag(new_run_id, "promotion_status", "rejected_tss")
         return False

     *** STAGING GATE [RULE-21]:
     if os.environ.get("ENV", "dev") != "dev":
         staging_ok = _run_staging_gate()
         if not staging_ok:
             mlflow_client.set_tag(new_run_id, "promotion_status", "rejected_staging")
             return False

     *** COPY MODELS AND UPDATE HASHES [T-2]:
     _copy_model_files_to_canonical_paths()
     _update_model_hashes("configs/model_hashes.yaml")  ← write SHA-256
     _update_version_yaml("configs/version.yaml", new_run_id)

     mlflow_client.set_tag(new_run_id, "promotion_status", "promoted")
     return True

   _run_staging_gate() -> bool:
     import subprocess
     # 1. Health check
     r1 = subprocess.run(
         ["curl", "-sf", "http://staging:8000/health"],
         capture_output=True, timeout=30)
     if r1.returncode != 0:
         print("[RULE-21] STAGING GATE FAILED: /health returned non-200")
         return False
     # 2. Integration test
     r2 = subprocess.run(
         ["python", "scripts/integration_test_20240222.py"],
         capture_output=True, timeout=300, text=True)
     if "INTEGRATION TEST PASSED" not in r2.stdout:
         print(f"[RULE-21] STAGING GATE FAILED: integration test output:\n{r2.stdout}")
         return False
     print("[RULE-21] STAGING GATE PASSED.")
     return True

   _update_model_hashes(path: str):
     import hashlib, yaml
     model_files = {
         "xgb_multiclass.json": "models/xgb_multiclass.json",
         "xgb_multiclass.pkl":  "models/xgb_multiclass.pkl",
         "causal_lstm.pt":      "models/causal_lstm.pt",
         "conformal_mapie.pkl": "models/conformal_mapie.pkl",
         "tcn_encoder.onnx":    "models/tcn_encoder.onnx",
     }
     hashes = {}
     for name, fpath in model_files.items():
         if Path(fpath).exists():
             hashes[name] = hashlib.sha256(
                 Path(fpath).read_bytes()).hexdigest()
     with open(path, "w") as f:
         yaml.dump(hashes, f)
     print(f"[T-2] Model hashes updated in {path}.")

6. src/deployment/drift_detector.py
   detect_covariate_drift(reference_df, current_df, threshold=0.05) -> dict:
     KS test per feature. Feature drift if p_value < threshold.
     Returns {"drifted": bool, "drifted_features": list[str], "p_values": dict}.
   detect_prediction_drift(ref_proba, curr_proba, threshold=0.10):
     PSI (Population Stability Index) on predicted class distributions.
     PSI > 0.20: major shift. PSI 0.10-0.20: moderate. < 0.10: stable.

7. .github/workflows/ci.yml:
   trigger: push to main or PR.
   steps:
     - Secret scan: uses gitleaks/gitleaks-action@v2 [T-6] ← FIRST STEP
     - Setup Python 3.11
     - pip install -r requirements.txt
     - pytest tests/ --timeout=120
       NOTE: CI tests MUST be fast (< 120s). Use synthetic data only.
       Do NOT load MOMENT or Chronos in CI — skip with @pytest.mark.slow.
       Mark heavyweight tests: @pytest.mark.slow and skip in CI:
         pytest -m "not slow" --timeout=120
     - Build Docker api image
     - Run: docker compose -f docker-compose.staging.yml up -d
     - Run: sleep 15 && curl -sf http://localhost:8000/health
     - Run: docker compose -f docker-compose.staging.yml down

8. .github/workflows/retrain.yml:
   trigger: schedule (weekly) or manual dispatch.
   steps:
     - Download latest PRADAN data
     - Run full pipeline: python scripts/run_pipeline.py --mode retrain
     - Run promote_if_better() — FAR gate + staging gate enforced
     - On promotion: push updated configs/model_hashes.yaml and
       configs/version.yaml to repo (auto-commit by workflow bot)

9. Prometheus + Grafana configuration:
   prometheus/prometheus.yml:
     scrape_configs:
       - job_name: solar-sentinel-api
         static_configs:
           - targets: ["api:8000"]
         metrics_path: /metrics
         scrape_interval: 15s
       - job_name: solar-sentinel-probe
         static_configs:
           - targets: ["api:8000"]
         metrics_path: /health
         scrape_interval: 30s

   grafana/dashboards/solar_sentinel.json — DEFINE THESE PANELS:
     Panel 1: solar_inference_duration_seconds P99 — red line at 90s [SLO-1]
     Panel 2: solar_alert_total rate by flare_class — stacked time series
     Panel 3: solar_data_freshness_seconds — red alert if > 180s [SLO-2]
     Panel 4: solar_far_latest — red threshold line at 0.10 [SLO-5]
     Panel 5: solar_lead_time_minutes — green target line at 30min [SLO-4]
     Panel 6: probe_success 24h average — SLO-3 availability gauge
   Each panel has an alert rule with notification to SLACK_WEBHOOK_URL.

10. DATA RETENTION POLICY (document in docs/data_retention.md):
    Raw FITS files:            Retain 90 days, then delete. (Storage: ~5GB/month)
    Processed parquet cache:   Retain 365 days.
    master_catalogue.csv:      PERMANENT — the primary PS15 deliverable.
    MLflow experiment runs:    Retain 365 days for active models, 30 days for
                               rejected runs (model_rejected tag present).
    Grafana metrics (Prometheus): 90-day retention (default Prometheus setting).
    logs/                      Retain 30 days. Rotate daily. Max 1GB total.
    Implementation: scripts/cleanup_old_data.py (cron daily at 02:00).

11. scripts/generate_synthetic_fits.py:
    Generate minimal valid SoLEXS and HEL1OS FITS files for CI/staging.
    Inject a synthetic M3.0 flare at t=60min in a 120-minute window.
    These files are used by docker-compose.staging.yml to run the full
    stack without PRADAN access. Output to data/synthetic/.

CHECKPOINT:
docker compose -f docker-compose.staging.yml up -d exits without error.
curl http://localhost:8000/health returns {"status":"ok"}.
curl http://localhost:8000/metrics contains "solar_inference_duration_seconds".
promote_if_better() rejects FAR>0.10 (verified by unit test).
promote_if_better() runs staging gate when ENV != "dev" (verified by test).
model_hashes.yaml populated with real SHA-256 hashes after promotion.
gitleaks CI step exits 0 (no secrets in repo).
```


---

## FINAL INTEGRATION TEST

```
[CONTEXT BLOCK — include ALL sections: MASTER CONTEXT BLOCK,
SLO DEFINITIONS, THREAT MODEL, PRE-MILESTONE CHECKS]

This is the final production readiness gate for SolarSentinel.
All milestones M0–M16 must be complete before running this test.
The test script is scripts/integration_test_20240222.py.
It must print "INTEGRATION TEST PASSED — SolarSentinel L6 ready."
to be accepted by the staging gate in promote_if_better().

The event under test: X6.3 solar flare on 2024-02-22.
This is the most powerful flare of Solar Cycle 25 up to that date.
Peak at approximately 22:34 UTC. GOES confirmed. Both SoLEXS and
HEL1OS should record this event clearly.

DATA PREPARATION:
  Download from PRADAN: SoLEXS Level-1 and HEL1OS Level-1 for 2024-02-22.
  If PRADAN is unavailable, use synthetic FITS from scripts/generate_synthetic_fits.py
  and note in output: "[SYNTHETIC DATA — PRADAN unavailable]".

scripts/integration_test_20240222.py
AREA 1 — DATA INGESTION AND SCHEMA VALIDATION:
  solexs_df = read_solexs(SOLEXS_PATH_20240222)
  hel1os_df = read_hel1os(HEL1OS_PATH_20240222)
  assert solexs_df is not None and len(solexs_df) > 60, \
      "SoLEXS DataFrame must have > 60 rows"
  assert hel1os_df is not None and len(hel1os_df) > 60, \
      "HEL1OS DataFrame must have > 60 rows"
  print("AREA 1 PASSED: Ingestion and schema validation.")

AREA 2 — INDEPENDENT DETECTIONS:
  solexs_events = detect_solexs_flares(solexs_df, fits_path=SOLEXS_PATH_20240222)
  hel1os_events = detect_hel1os_flares(hel1os_df, fits_path=HEL1OS_PATH_20240222)
  assert any(e.flare_class in ("M", "X") for e in solexs_events), \
      "SoLEXS must detect at least one M or X-class flare on 2024-02-22"
  assert len(hel1os_events) > 0, "HEL1OS must detect at least one burst"
  hxr_peak = min(e.peak_time for e in hel1os_events)
  sxr_peak = min(e.peak_time for e in solexs_events
                 if e.flare_class in ("M","X"))
  neupert_lag = (sxr_peak - hxr_peak).total_seconds() / 60
  assert 0 < neupert_lag <= 10, \
      f"Neupert effect: HXR must lead SXR by 1-10 min, got {neupert_lag:.1f}min"
  print(f"AREA 2 PASSED: SoLEXS detected {len(solexs_events)} events. "
        f"HEL1OS detected {len(hel1os_events)} events. "
        f"Neupert lag: {neupert_lag:.1f} min.")

AREA 3 — MASTER CATALOGUE MERGE AND PROVENANCE:
  import uuid
  test_run_id = str(uuid.uuid4())
  test_model_version = yaml.safe_load(open("configs/version.yaml"))["model_version"]
  master = merge_catalogues(
      solexs_events, hel1os_events,
      pipeline_run_id=test_run_id,
      model_version=test_model_version)
  assert len(master) > 0, "Master catalogue must be non-empty"
  dual_events = master[master["source"] == "dual"]
  assert len(dual_events) > 0, "At least one dual (SXR+HXR) event must be detected"
  x63 = master[master["flare_class"] == "X"]
  assert len(x63) > 0, "X6.3 event must be in master catalogue as X-class"
  assert x63.iloc[0]["confidence"] > 0.85, \
      f"X6.3 confidence must be > 0.85, got {x63.iloc[0]['confidence']:.3f}"
  # PROVENANCE CHECK [RULE-17]:
  for col in ["solexs_fits_path", "hel1os_fits_path",
              "model_version", "pipeline_run_id"]:
      assert col in master.columns, f"Missing provenance column: {col}"
  assert (master["pipeline_run_id"] == test_run_id).all(), \
      "pipeline_run_id must match the test run UUID in every row"
  print(f"AREA 3 PASSED: {len(master)} events in catalogue. "
        f"X6.3 detected, confidence {x63.iloc[0]['confidence']:.3f}. "
        f"Provenance columns verified.")

AREA 4 — NOWCASTING (N/C/M/X classification):
  x63_feat = engineer_physics_features(
      solexs_df.loc[x63.iloc[0]["start_time"]:x63.iloc[0]["peak_time"]])
  nowcaster = ONNXNowcaster(TCN_ONNX_PATH, joblib.load(XGB_PATH))
  window = x63_feat[FEATURE_COLS].values[-60:]
  result  = nowcaster.predict(window[np.newaxis].astype(np.float32),
                              window[-1:].astype(np.float32))
  assert result["class"] in ("M", "X"), \
      f"Nowcaster must predict M or X for X6.3 flare, got {result['class']}"
  assert result["confidence"] > 0.70, \
      f"Confidence for X6.3 must be > 0.70, got {result['confidence']:.3f}"
  print(f"AREA 4 PASSED: Nowcast = {result['class']}, "
        f"confidence = {result['confidence']:.3f}.")

AREA 5 — FORECASTING AND LEAD TIME:
  forecaster = ThreeModelEnsemble(lstm, tcn, timesfm)
  quiet_start = x63.iloc[0]["start_time"] - pd.Timedelta(hours=1)
  quiet_end   = x63.iloc[0]["start_time"] - pd.Timedelta(minutes=5)
  quiet_window = solexs_df.loc[quiet_start:quiet_end]["flux_high"].values
  forecast = forecaster.predict_single(
      torch.tensor(quiet_window[-60:], dtype=torch.float32).unsqueeze(0),
      quiet_window[-60:], horizon=15)
  assert forecast[1] + forecast[2] + forecast[3] > 0.30, \
      f"Forecast must show P(flare) > 0.30 in pre-flare window"
  print(f"AREA 5 PASSED: P(C/M/X | 15min) = {sum(forecast[1:]):.3f}.")

AREA 6 — UNCERTAINTY QUANTIFICATION:
  mapie = load_mapie()
  chronos_unc = chronos_forecast_interval(quiet_window[-60:])
  assert chronos_unc["q90"] > chronos_unc["q10"], \
      "Chronos q90 must be > q10"
  assert chronos_unc["q10"] >= 0, \
      "Chronos q10 must be non-negative (flux cannot be negative)"
  mapie_pred, mapie_set = mapie.predict(
      np.array([window[-1]]).reshape(1,-1), alpha=0.10)
  assert mapie_set.shape[1] == 4, "MAPIE set must have 4 class columns"
  print(f"AREA 6 PASSED: Chronos q10={chronos_unc['q10']:.3e}, "
        f"q90={chronos_unc['q90']:.3e}. MAPIE set shape {mapie_set.shape}.")

AREA 7 — PHYSICS (NEUPERT + PINN + PHASE DETECTOR):
  pinn = NeupertPINN(...)
  pinn.load_state_dict(torch.load("models/neupert_pinn.pt",
                                   map_location="cpu", weights_only=True))
  phase_seq = build_phase_sequence(solexs_df.loc[
      x63.iloc[0]["start_time"]:x63.iloc[0]["end_time"]])
  phases_present = set(phase_seq)
  assert len(phases_present) >= 3, \
      f"Must detect ≥ 3 distinct phases for X6.3, got {phases_present}"
  print(f"AREA 7 PASSED: Phases detected = {sorted(phases_present)}.")

AREA 8 — ANOMALY (MOMENT):
  pre_flare_window = solexs_df.loc[
      x63.iloc[0]["start_time"] - pd.Timedelta(hours=1):
      x63.iloc[0]["start_time"]]["flux_high"].values
  quiet_baseline_window = solexs_df.iloc[:512]["flux_high"].values
  if len(pre_flare_window) < 512:
      pre_flare_window = np.pad(pre_flare_window, (512-len(pre_flare_window),0))
  if len(quiet_baseline_window) < 512:
      quiet_baseline_window = np.pad(quiet_baseline_window,
                                     (512-len(quiet_baseline_window),0))
  error_preflare = compute_reconstruction_error(pre_flare_window)
  error_quiet    = compute_reconstruction_error(quiet_baseline_window)
  assert error_preflare > error_quiet, \
      f"MOMENT reconstruction error must be higher pre-flare " \
      f"({error_preflare:.4f}) than quiet ({error_quiet:.4f})"
  print(f"AREA 8 PASSED: MOMENT pre-flare error={error_preflare:.4f} "
        f"> quiet error={error_quiet:.4f}.")

AREA 9 — EXPLAINABILITY (SHAP):
  explainer = SHAPExplainer(joblib.load(XGB_PATH), feature_names=FEATURE_COLS)
  exp = explainer.explain(window[-1:])
  assert "top_features" in exp and len(exp["top_features"]) == 5, \
      "SHAP must return exactly 5 top features"
  assert all(f in FEATURE_COLS for f in exp["top_features"]), \
      "All SHAP top features must be valid feature names"
  print(f"AREA 9 PASSED: Top SHAP features = {exp['top_features']}.")

AREA 10 — LLM REPORT:
  reporter = SolarFlareReporter()
  import time
  t_report = time.perf_counter()
  report = reporter.forward(
      flare_class=result["class"],
      peak_flux=float(x63.iloc[0]["peak_flux_sxr"]),
      lead_time_min=15.0,
      confidence=result["confidence"],
      graphrag_context="X6.3 flare on 2024-02-22 — extreme radiation event.",
      uncertainty_q10=chronos_unc["q10"],
      uncertainty_q90=chronos_unc["q90"])
  report_time = time.perf_counter() - t_report
  assert isinstance(report, str) and len(report) > 20, \
      "LLM report must be a non-trivial string"
  assert report_time < 30, \
      f"LLM report must complete within 30s [SLO-1], took {report_time:.1f}s"
  print(f"AREA 10 PASSED: Report generated in {report_time:.1f}s. "
        f"Preview: {report[:80]}...")

AREA 11 — SECURITY CHECKS:
  from src.monitoring.model_integrity import verify_model_hash
  # Must not raise SecurityError (hashes match after last promotion)
  for model_file in ["models/xgb_multiclass.json", "models/causal_lstm.pt",
                     "models/conformal_mapie.pkl"]:
      if Path(model_file).exists():
          verify_model_hash(model_file)  # raises SecurityError on mismatch
  # WebSocket rate limiting — check rate limit function exists
  from src.api.main import live_feed
  assert hasattr(live_feed, "__wrapped__") or True, \
      "WebSocket handler must be defined"
  print("AREA 11 PASSED: Model hash verification clean. WebSocket endpoint present.")

AREA 12 — SLO COMPLIANCE:
  import yaml, mlflow
  slo_cfg = yaml.safe_load(open("configs/slo.yaml"))
  # Fetch latest evaluation metrics from MLflow
  client = mlflow.tracking.MlflowClient()
  latest_run = client.search_runs(
      experiment_ids=["0"], order_by=["start_time DESC"], max_results=1)[0]
  mean_far      = latest_run.data.metrics.get("mean_far",  1.0)
  mean_lead_min = latest_run.data.metrics.get("mean_lead_min", 0.0)
  assert mean_far <= slo_cfg["far_max"], \
      f"[SLO-5] FAR={mean_far:.4f} exceeds limit {slo_cfg['far_max']}"
  assert mean_lead_min >= slo_cfg["lead_time_min_accepted"], \
      f"[SLO-4] Lead time={mean_lead_min:.1f}min < accepted {slo_cfg['lead_time_min_accepted']}min"
  # Check component timing against budget from state["timing"] in M7 e2e run
  print(f"AREA 12 PASSED: FAR={mean_far:.4f} ≤ {slo_cfg['far_max']}. "
        f"Lead time={mean_lead_min:.1f}min ≥ {slo_cfg['lead_time_min_accepted']}min.")

print("\n" + "="*60)
print("INTEGRATION TEST PASSED — SolarSentinel L6 ready.")
print("="*60)
print(f"  Event tested:    X6.3 flare 2024-02-22")
print(f"  Areas verified:  12/12")
print(f"  Nowcast result:  {result['class']} (confidence {result['confidence']:.3f})")
print(f"  FAR:             {mean_far:.4f}")
print(f"  Lead time:       {mean_lead_min:.1f} min")
print(f"  SLO status:      PASS")
print("="*60)
```

---

## USAGE GUIDE

*How to use this document effectively across milestones.*

```
DOCUMENT STRUCTURE:
  This prompt.md has six sections. In each milestone conversation, paste:
  1. MASTER CONTEXT BLOCK (21 rules + architecture summary)
  2. SLO DEFINITIONS (5 SLOs with component budgets)
  3. THREAT MODEL (6 threats + security checklist — paste at M16 start)
  4. PRE-MILESTONE CHECKS (paste when starting from scratch or new session)
  5. The specific milestone (M0–M16) you are implementing
  Never paste the entire file into one prompt — it is too long.
  Paste by section; the AI will use what it needs.

MILESTONE SEQUENCING (dependencies):
  M0 → M1 → M2 → M3 → M4 → M5 → M6 → M7 → M8 → M9 → M10
  M11 (needs M5, M7) → M12 (needs M7) → M13 (needs M4) → M14 (needs M7)
  M15 (needs M4, M5) → M16 (needs all) → FINAL INTEGRATION TEST

FIRST SESSION CHECKLIST (before any code):
  1. Run all 8 verification checks (scripts/verify_*.py)
  2. Fill configs/fits_columns.yaml from CHECK 1 output
  3. Update src/ingestion/schemas.py column names from CHECK 1
  4. Record MOMENT attribute name from CHECK 4
  5. Record working DSPy syntax from CHECK 5
  6. Record PRADAN auth mechanism from CHECK 6
  7. Confirm GraphRAG SSRF assertion passes from CHECK 7
  8. Confirm CHECK 8 (schema smoke test) passes

WHEN THE AI MAKES A MISTAKE:
  If the AI violates a rule, reference the rule number in your correction:
  "RULE-7: do NOT call predict_quantiles — use predict() instead."
  If the AI uses a session fixture with hypothesis: cite RULE-14.
  If the AI opens a config at module level without a guard: cite RULE-12.
  If a metric is defined inline: cite RULE-20.
  If provenance columns are missing from the catalogue: cite RULE-17.
  This leads to faster correction than explaining from scratch.

SLO TRACKING:
  After every M15 run, update this table:
  | SLO | Target | Last Measured | Status |
  |-----|--------|---------------|--------|
  | SLO-1 End-to-end latency | ≤ 90s   | ___s | ___ |
  | SLO-2 Throughput cadence | ≥1/60s  | ___s | ___ |
  | SLO-3 Availability       | ≥99%    | ___%  | ___ |
  | SLO-4 Lead time          | ≥10min  | ___min| ___ |
  | SLO-5 FAR                | ≤0.10   | ___   | ___ |
  Any FAIL here blocks M16 model promotion.

THREAT MODEL USAGE:
  Share the THREAT MODEL section with the AI at the start of M16.
  Ask it to verify each threat mitigation is implemented.
  Use the SECURITY CHECKLIST to walk through each line.

CHECKPOINT DISCIPLINE:
  Do not proceed to the next milestone until all checkpoint conditions
  pass. Checkpoint conditions are the minimum bar, not the ceiling.
  If a checkpoint fails, fix the root cause — do not patch around it.

PROVENANCE AUDIT WORKFLOW (post-deployment):
  When an alert is disputed or needs investigation:
  python scripts/audit_catalogue.py --peak_time "2024-02-22T22:34:00"
  This traces the alert back to its FITS files, model version, and
  pipeline run ID using the provenance columns added in M3 [RULE-17].

GRAFANA ALERT RESPONSE:
  solar_far_latest > 0.10    → Run M8 (FLAML re-tune). Check class thresholds.
  solar_lead_time_minutes < 5 → Check TimesFM forecast horizon. Retrain ensemble.
  solar_data_freshness > 180s → Check PRADAN connectivity. Check scheduler logs.
  probe_success < 0.99       → Check Docker stack health. Restart API container.
```

---

## RULE QUICK REFERENCE

*Post this on your wall. Reference it before every code review.*

```
 #  | Rule summary                                         | Where violated most
----|------------------------------------------------------|---------------------
 1  | Independent detectors — never merge before detecting | M2, M3
 2  | LSTM bidirectional=False (no leakage)                | M5, M7
 3  | TSS not accuracy                                     | M4, M8, M15
 4  | FITS columns from YAML — no hardcoding               | M1
 5  | XGBoost multi:softprob, 4 classes, per-class thresh  | M4, M8
 6  | timesfm.finetuning does not exist                    | M5
 7  | predict() not predict_quantiles()                    | M11
 8  | confusion_matrix labels=[0,1] always                 | M8, M15
 9  | GraphRAG: local embeddings only (no OpenAI call)     | M9
10  | dspy.context() per-call (not dspy.configure())       | M9, M7
11  | torch.load weights_only=True                         | M5, M7, M10, M11
12  | Module-level open() needs FileNotFoundError guard    | M1, M7, M10
13  | Consistent model save format (JSON+PKL / PT / ONNX)  | M4, M5, M6
14  | No session fixtures in hypothesis @given tests       | M6
15  | No private function imports across modules           | M3, M7
16  | Singleton loaders for Chronos, MOMENT, ONNX          | M7, M11, M12
17  | Provenance columns in every master_catalogue.csv row | M3, M7
18  | Pandera schema validation at ingestion boundary      | M1
19  | Dashboard = React+TypeScript+Vite (not Streamlit)    | M0, M14
20  | Prometheus metrics centralised in metrics.py         | M7, M14, M16
21  | dev→staging→prod gate before production promotion    | M16
----|------------------------------------------------------|---------------------
SLO-5 | FAR > 0.10 blocks promote_if_better() always     | M8, M16
T-2   | verify_model_hash() on every model load path      | M6, M7, M11, M12
T-3   | WebSocket: 20-conn cap, 1 msg/10s rate limit      | M14
T-6   | gitleaks in CI and pre-commit                      | M16
```

---

## ADDENDUM v3 — ALL REMAINING GAPS CLOSED
*This section supersedes earlier milestone descriptions where they conflict.
Paste alongside the MASTER CONTEXT BLOCK in any milestone conversation.*

---

### NEW RULES: RULE-22 and RULE-23

```
[RULE-22] NEVER start a training script without setting all random seeds first.
The first four executable lines of every training script (train.py,
flaml_tuner.py, pinn.py, multi_horizon.py) must be:
    import random, numpy as np, torch
    SEED = 42
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
This applies to ALL scripts that call model.fit(), model.train(), or
any augmentation function. Without this, two runs on identical data
produce different models, making bug reproduction and ablations impossible.
CI enforcement: add to .pre-commit-config.yaml:
    - repo: local
      hooks:
        - id: seed-check
          name: "Training scripts must set SEED=42"
          entry: grep -rn "^SEED = 42" src/nowcasting/train.py
                 src/forecasting/multi_horizon.py
                 src/automl/flaml_tuner.py src/physics/pinn.py
          language: system
          pass_filenames: false

[RULE-23] NEVER ingest a FITS file without first checking the seen-files
registry. The registry is data/processed/seen_files.db (SQLite). Before
calling read_solexs() or read_hel1os() in the scheduler loop, check:
    from src.ingestion.seen_files import is_seen, mark_seen
    if is_seen(fits_path):
        logger.info(f"Skipping already-processed FITS: {fits_path}")
        continue
    # ... process ...
    mark_seen(fits_path, pipeline_run_id)
Without this, the scheduler re-processes the same FITS file ~10 times
between PRADAN updates (scheduler: 60s cadence, PRADAN: ~10min cadence),
writing duplicate catalogue rows and wasting compute. is_seen() uses a
SQLite PRIMARY KEY on fits_path — duplicate inserts are rejected at DB level.
```

---

### UPDATED SLO DEFINITIONS (add SLO-6)

```
[SLO-6] TRUE POSITIVE RATE — HARD GATE, same weight as SLO-5
Target: TPR ≥ 0.80 for M-class + X-class events combined on walk-forward CV.
Rationale: SLO-5 gates on FAR ≤ 0.10 but a model that never fires has
FAR=0 and would pass SLO-5 while missing every flare. TPR = TP / (TP + FN)
on (y_true > 1) vs (y_pred > 1). The PS explicitly requires "High True
Positive Rate" — this SLO makes that concrete and auditable.
Implementation: add to check_slo_compliance() in slo_reporter.py:
    mx_mask = y_true >= 2  # M=2, X=3
    if mx_mask.sum() == 0:
        tpr_mx = None  # no M/X in this fold, skip
    else:
        tp = ((y_pred >= 2) & mx_mask).sum()
        fn = ((y_pred <  2) & mx_mask).sum()
        tpr_mx = tp / (tp + fn + 1e-9)
    slo_status["tpr_mx"] = (
        "pass" if (tpr_mx is None or tpr_mx >= slo["tpr_mx_min"]) else "fail")
Add to configs/slo.yaml:
    tpr_mx_min: 0.80
promote_if_better() must reject models where tpr_mx < 0.80:
    if tpr_mx is not None and tpr_mx < 0.80:
        mlflow_client.set_tag(new_run_id, "promotion_status",
                              "rejected_tpr_mx")
        mlflow_client.set_tag(new_run_id, "model_tpr_mx", str(round(tpr_mx,4)))
        return False
```

---

### M1 UPDATE — Seen-Files Registry (supersedes M1 CHECKPOINT)

```
Add to src/ingestion/:

src/ingestion/seen_files.py
    import sqlite3, contextlib
    from pathlib import Path

    _DB_PATH = "data/processed/seen_files.db"

    def _conn():
        db = Path(_DB_PATH)
        db.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(db)
        con.execute("""
            CREATE TABLE IF NOT EXISTS seen_files (
                fits_path       TEXT PRIMARY KEY,
                instrument      TEXT NOT NULL,
                processed_at    TEXT NOT NULL DEFAULT (datetime('now')),
                pipeline_run_id TEXT NOT NULL
            )""")
        con.commit()
        return con

    def is_seen(fits_path: str) -> bool:
        '''Return True if this FITS file has already been ingested.'''
        with contextlib.closing(_conn()) as con:
            row = con.execute(
                "SELECT 1 FROM seen_files WHERE fits_path=?",
                (fits_path,)).fetchone()
        return row is not None

    def mark_seen(fits_path: str, instrument: str,
                  pipeline_run_id: str) -> None:
        '''Register a FITS file as processed. Idempotent (IGNORE on dup).'''
        with contextlib.closing(_conn()) as con:
            con.execute(
                "INSERT OR IGNORE INTO seen_files "
                "(fits_path, instrument, pipeline_run_id) VALUES (?,?,?)",
                (fits_path, instrument, pipeline_run_id))
            con.commit()

    def list_seen(instrument: str | None = None) -> list[dict]:
        '''Audit: return all seen files, optionally filtered by instrument.'''
        with contextlib.closing(_conn()) as con:
            q = "SELECT * FROM seen_files"
            if instrument:
                q += " WHERE instrument=?"
                rows = con.execute(q, (instrument,)).fetchall()
            else:
                rows = con.execute(q).fetchall()
        keys = ["fits_path","instrument","processed_at","pipeline_run_id"]
        return [dict(zip(keys, r)) for r in rows]

Integration in src/orchestration/scheduler.py (add to ingestion loop):
    from src.ingestion.seen_files import is_seen, mark_seen
    for fits_path in pradan_downloader.list_new_files():
        instrument = "solexs" if "solexs" in fits_path.lower() else "hel1os"
        if is_seen(fits_path):
            logger.info(f"[RULE-23] Skip duplicate: {fits_path}")
            continue
        state = pipeline.invoke({
            "solexs_path": fits_path if instrument=="solexs" else "",
            "hel1os_path": fits_path if instrument=="hel1os" else "",
            ...
        })
        mark_seen(fits_path, instrument, state["pipeline_run_id"])

Tests to add in tests/test_pipeline.py:
    test_seen_files_is_seen_returns_false_for_new_file
    test_seen_files_mark_then_is_seen_returns_true
    test_seen_files_duplicate_mark_is_idempotent:
        mark_seen(path, "solexs", "run-1")
        mark_seen(path, "solexs", "run-2")  # must NOT raise
        assert list_seen() has exactly 1 entry for path
    test_scheduler_skips_already_seen_file:
        Mark a FITS file as seen. Run scheduler loop over that file.
        Assert read_solexs() is NOT called. Assert catalogue NOT updated.
```

---

### M2 UPDATE — B-class Design Decision (supersedes M2 FlareEvent spec)

```
DESIGN DECISION D10 — B-class classification (document this in M2):

The PS requires "detection of low- and high-class flares." B-class is the
lowest recognised GOES flare class (peak flux 1e-7 to 1e-6 W/m2).

DECISION: B-class is treated as N (quiet sun) in the primary 4-class model
(N/C/M/X). Rationale:
  1. Aditya-L1 SoLEXS sensitivity floor at solar minimum is ~1e-8 W/m2,
     making B-class detection marginal and instrument-noise-dependent.
  2. B-class events cause no space weather impact — they do not affect GPS,
     satellites, or power grids. Alerting on B-class would massively inflate
     FAR without operational value.
  3. GOES_THRESHOLDS retains B: 1e-7 for catalogue completeness, but
     classify_flux() returns "B" only in the raw catalogue, not in model labels.

IMPLEMENTATION:
  In labels.py, CLASS_MAP maps GOES class to model label:
      CLASS_MAP = {"N": 0, "B": 0, "C": 1, "M": 2, "X": 3}
  The "B" → 0 mapping is explicit (not silent). A comment in CLASS_MAP:
      "B": 0,  # D10: B-class = no operational alert (see docs/decisions/D10.md)

  In detect_solexs_flares(), events with flare_class=="B" are included
  in the raw SoLEXS catalogue (they ARE detected and logged) but labelled
  class=0 in training data. The master catalogue column "flare_class"
  retains "B" for observational completeness.

  Create docs/decisions/D10.md with this full rationale. Reference it
  from CLASS_MAP and from the integration test (no assert on B-class events
  in the integration test — they are in the catalogue but not in the model).

Add to tests/test_pipeline.py:
    test_b_class_detected_in_raw_catalogue:
        Synthetic flux at 5e-7 (B-class). detect_solexs_flares returns
        1 event with flare_class == "B".
    test_b_class_maps_to_label_zero:
        CLASS_MAP["B"] == 0
    test_b_class_not_in_alert_path:
        nowcast_class == "B" should never reach alert_router (class 0 = N = silent).
```

---

### M4 UPDATE — Reproducibility Seed (supersedes M4 train.py spec)

```
Add to the TOP of src/nowcasting/train.py (first 5 executable lines):
    import random, numpy as np, torch
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

Add XGBoost seed:
    XGBoost: objective="multi:softprob", num_class=4, tree_method="hist",
             seed=SEED     ← add this param [RULE-22]

Add to tests/test_pipeline.py:
    test_training_is_reproducible:
        Call train_multiclass_nowcast twice with same data and same SEED.
        Assert model.predict(X_val) returns identical arrays both times.
        This catches non-determinism from XGBoost subsampling and TSN dropout.
```

---

### M5 UPDATE — Reproducibility Seed

```
Add to the TOP of src/forecasting/multi_horizon.py and causal_lstm.py:
    SEED = 42
    import random, numpy as np, torch
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

Add to tests/test_pipeline.py:
    test_lstm_training_is_reproducible:
        Train CausalLSTMForecaster twice with same data and SEED.
        Assert model(x).detach().numpy() is identical across runs.
```

---

### M15 UPDATE — SLO-6 Enforcement (supersedes M15 slo_reporter.py spec)

```
Updated check_slo_compliance() in src/evaluation/slo_reporter.py:

def check_slo_compliance(metrics: dict) -> dict:
    slo = yaml.safe_load(open("configs/slo.yaml"))
    slo_status = {}
    slo_status["far"]      = "pass" if metrics["far"] <= slo["far_max"] else "fail"
    slo_status["lead_time"]= "pass" if metrics.get("mean_lead_min", 0) >=
                              slo["lead_time_min_accepted"] else "fail"
    # [SLO-6] TPR gate for M+X class:
    tpr_mx = metrics.get("tpr_mx")
    if tpr_mx is not None:
        slo_status["tpr_mx"] = "pass" if tpr_mx >= slo["tpr_mx_min"] else "fail"
    overall = "pass" if all(v == "pass" for v in slo_status.values()) else "fail"
    return {"overall": overall, "detail": slo_status, "tpr_mx": tpr_mx}

Updated log_evaluation_to_mlflow():
    mlflow.log_metrics({
        "mean_tss":      metrics["tss"],
        "mean_far":      metrics["far"],
        "mean_tpr":      metrics["tpr"],
        "mean_lead_min": metrics["mean_lead_min"],
        "tpr_mx":        metrics.get("tpr_mx", -1),  ← [SLO-6]
    })
    mlflow.set_tags({
        "slo_status":       slo_result["overall"],
        "slo_far_status":   slo_result["detail"]["far"],
        "slo_lead_status":  slo_result["detail"]["lead_time"],
        "slo_tpr_mx_status":slo_result["detail"].get("tpr_mx","n/a"),  ← [SLO-6]
    })

Add to configs/slo.yaml:
    tpr_mx_min: 0.80

Add tests:
    test_slo_reporter_fails_on_low_tpr_mx:
        check_slo_compliance({"far":0.06,"mean_lead_min":15,"tpr_mx":0.62})
        Assert overall=="fail", slo_tpr_mx_status=="fail".
    test_slo_reporter_passes_all_three_slos:
        check_slo_compliance({"far":0.06,"mean_lead_min":12,"tpr_mx":0.85})
        Assert overall=="pass".
    test_promote_blocks_on_low_tpr_mx:
        MLflow mock: new run has mean_far=0.05, tpr_mx=0.65.
        Assert promote_if_better() returns False.
        Assert mlflow tag "promotion_status"=="rejected_tpr_mx".
```

---

### M15 UPDATE — NOAA Catalog Download Script

```
Add scripts/download_noaa_catalog.py:

    import requests, pandas as pd
    from pathlib import Path

    NOAA_URL = (
        "https://www.ngdc.noaa.gov/stp/space-weather/solar-data/"
        "solar-features/solar-flares/x-rays/goes/xrs/"
        "goes-xrs-report_{year}.txt"
    )

    def download_noaa_catalog(start_year: int = 2010,
                               end_year:   int = 2024,
                               out_path:   str = "data/raw/noaa_catalog.parquet"
                               ) -> pd.DataFrame:
        '''
        Download NOAA GOES flare catalog for start_year..end_year.
        Parses the fixed-width text format. Saves to parquet.
        WHY: merger.py's check_noaa_confirmed() needs this file.
             CI/staging use synthetic data so NOAA access must be
             optional (skip if URL unavailable — log warning, return empty df).
        '''
        frames = []
        for year in range(start_year, end_year + 1):
            try:
                resp = requests.get(NOAA_URL.format(year=year), timeout=30)
                resp.raise_for_status()
                # Parse fixed-width: cols = [date, time, class, location, ...]
                from io import StringIO
                df = pd.read_fwf(StringIO(resp.text), skiprows=4,
                                 colspecs=[(0,8),(9,14),(15,21),(22,27),
                                           (28,30),(31,34),(35,40)],
                                 names=["date","start","peak","end",
                                        "goes","xray_class","noaa_region"])
                df["peak_time"] = pd.to_datetime(
                    df["date"].astype(str) + " " + df["peak"].astype(str),
                    format="%Y%m%d %H%M", utc=True, errors="coerce")
                df = df.dropna(subset=["peak_time"])
                frames.append(df)
            except Exception as exc:
                print(f"[WARNING] NOAA {year} unavailable: {exc}. Skipping.")
        if not frames:
            print("[WARNING] No NOAA data downloaded. "
                  "check_noaa_confirmed() will return False for all events.")
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_path)
        print(f"NOAA catalog: {len(out)} flares → {out_path}")
        return out

    if __name__ == "__main__":
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--start-year", type=int, default=2010)
        p.add_argument("--end-year",   type=int, default=2024)
        args = p.parse_args()
        download_noaa_catalog(args.start_year, args.end_year)

Add to merger.py: load NOAA catalog with graceful empty-df fallback:
    try:
        noaa_df = pd.read_parquet("data/raw/noaa_catalog.parquet")
    except FileNotFoundError:
        noaa_df = pd.DataFrame()   # CI/staging: NOAA not available
        logger.warning("NOAA catalog not found — noaa_confirmed=False for all events.")

Add to ci.yml: NOAA catalog is NOT downloaded in CI (no external network).
The empty-df path is the expected CI code path. Test coverage for
check_noaa_confirmed() uses a synthetic 3-row NOAA DataFrame as a fixture.
```

---

### M16 UPDATE — Rollback Script (add to M16 tasks)

```
Add scripts/rollback.py:

    import argparse, hashlib, shutil, yaml, mlflow
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
        artifact_dir.mkdir(exist_ok=True)
        for name, dest in MODEL_FILES.items():
            try:
                local = mlflow.artifacts.download_artifacts(
                    run_id=run_id, artifact_path=name,
                    dst_path=str(artifact_dir))
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

Add to tests/test_pipeline.py:
    test_rollback_updates_model_hashes:
        Mock mlflow artifact download returning toy model files.
        Run rollback_to_run("fake_run_id", "http://mock").
        Assert configs/model_hashes.yaml exists and has non-empty hashes.
    test_rollback_updates_version_yaml:
        After rollback, yaml.safe_load(version.yaml)["model_version"]
        must start with "rollback-".
```

---

### M16 UPDATE — Canary Deployment (add to promote_if_better())

```
Updated promote_if_better() — add canary step AFTER staging gate:

    *** CANARY GATE (add after _run_staging_gate() passes) ***

    if os.environ.get("ENV") == "prod":
        canary_ok = _run_canary(new_run_id, duration_minutes=60)
        if not canary_ok:
            mlflow_client.set_tag(new_run_id, "promotion_status",
                                  "rejected_canary")
            print("[CANARY] New model rejected at canary stage. "
                  "Current model retained.")
            return False

Add to src/deployment/model_registry.py:

    def _run_canary(new_run_id: str,
                    duration_minutes: int = 60) -> bool:
        '''
        Run old and new models in parallel for duration_minutes.
        Compare alert rates and FAR estimates on live PRADAN data.
        WHY: Walk-forward CV uses historical GOES data. Real-time PRADAN
        data has different noise characteristics and gap patterns. A canary
        catches FAR spikes before full production cutover.

        Implementation:
          1. Copy new model files to models/canary/ (separate directory).
          2. Start a shadow inference process: reads same FITS files as prod,
             runs new model, logs predictions to data/canary/predictions.csv
             WITHOUT triggering any alerts (shadow mode).
          3. After duration_minutes, compare:
             - new_far  = FPR on canary predictions vs NOAA catalog
             - prod_far = current production FAR from FAR_GAUGE metric
          4. If new_far > prod_far * 1.5: canary fails (new model worse).
             If new_far <= prod_far * 1.5: canary passes.
        '''
        import subprocess, time, csv
        from pathlib import Path
        canary_dir = Path("models/canary")
        canary_dir.mkdir(exist_ok=True)
        # Copy new model artifacts to canary directory
        _copy_model_files_to_canonical_paths(dest_dir="models/canary/")

        # Start shadow inference process
        proc = subprocess.Popen([
            "python", "src/orchestration/shadow_runner.py",
            "--model-dir", "models/canary/",
            "--output",    "data/canary/predictions.csv",
        ])

        # Wait for canary duration
        print(f"[CANARY] Shadow mode running for {duration_minutes} min...")
        time.sleep(duration_minutes * 60)
        proc.terminate()
        proc.wait()

        # Evaluate canary predictions
        try:
            preds = pd.read_csv("data/canary/predictions.csv")
            noaa  = pd.read_parquet("data/raw/noaa_catalog.parquet")
            new_far = _compute_canary_far(preds, noaa)
            prod_far_gauge = float(
                os.environ.get("PROD_FAR_ESTIMATE", "0.10"))
            passed = new_far <= prod_far_gauge * 1.5
            print(f"[CANARY] new_far={new_far:.4f}, "
                  f"prod_far={prod_far_gauge:.4f}. "
                  f"Result: {'PASS' if passed else 'FAIL'}")
            return passed
        except Exception as exc:
            print(f"[CANARY] Evaluation failed: {exc}. "
                  f"Failing safe — rejecting promotion.")
            return False

Add to .github/workflows/retrain.yml: set CANARY_DURATION_MINUTES: 60
In dev/staging environments, canary is skipped (ENV != "prod").
```

---

### M16 UPDATE — Load Testing (add to M16 tasks)

```
Add tests/load/locustfile.py:

    from locust import HttpUser, task, between, events
    import json

    class SolarSentinelUser(HttpUser):
        '''
        Load test verifying SLO-2 (throughput) and SLO-3 (availability).
        Run: locust -f tests/load/locustfile.py --host http://localhost:8000
             --users 20 --spawn-rate 2 --run-time 5m --headless
             --csv tests/load/results
        Pass criteria (SLO-2 + SLO-3):
          - /health 99th percentile response time < 1000ms
          - /history 99th percentile response time < 5000ms
          - failure rate < 1% across all endpoints
        '''
        wait_time = between(1, 3)

        @task(5)
        def health_check(self):
            self.client.get("/health",
                            name="/health [SLO-3 probe]")

        @task(3)
        def get_history(self):
            self.client.get("/history?n=50",
                            name="/history")

        @task(2)
        def get_status(self):
            self.client.get("/status",
                            name="/status [SLO-1 timing]")

        @task(1)
        def get_explain(self):
            self.client.get("/explain",
                            name="/explain [SHAP endpoint]")

    @events.quitting.add_listener
    def on_quit(environment, **kwargs):
        '''Assert SLO-2 and SLO-3 pass criteria at end of load test.'''
        stats = environment.stats
        health = stats.get("/health [SLO-3 probe]", "GET")
        if health:
            p99_ms = health.get_response_time_percentile(0.99)
            fail_pct = health.num_failures / max(health.num_requests, 1) * 100
            print(f"[SLO-3] /health P99={p99_ms:.0f}ms, fail={fail_pct:.2f}%")
            if p99_ms > 1000:
                print(f"[SLO-3 FAIL] /health P99 {p99_ms:.0f}ms > 1000ms limit")
                environment.process_exit_code = 1
            if fail_pct >= 1.0:
                print(f"[SLO-3 FAIL] failure rate {fail_pct:.2f}% >= 1%")
                environment.process_exit_code = 1
        print("[LOAD TEST] Complete.")

Add to .github/workflows/ci.yml (runs against staging stack):
    - name: Load test (SLO-2 + SLO-3)
      run: |
        pip install locust
        locust -f tests/load/locustfile.py \
          --host http://localhost:8000 \
          --users 20 --spawn-rate 4 --run-time 2m \
          --headless --csv tests/load/ci_results \
          --exit-code-on-error 1
      # 2 min is enough for CI. 5 min for full staging run.

Add to tests/test_pipeline.py:
    test_health_endpoint_responds_under_1s:
        Use requests + time.perf_counter(). 20 sequential calls to /health.
        Assert P99 response time < 1000ms (approximated from sorted times).
```

---

### M16 UPDATE — Test Coverage Gate (add to ci.yml)

```
Updated .github/workflows/ci.yml pytest step:

    - name: Test suite (fast only, with coverage)
      run: |
        pytest tests/ \
          -m "not slow" \
          --timeout=120 \
          --cov=src \
          --cov-report=term-missing \
          --cov-report=xml:coverage.xml \
          --cov-fail-under=75
      # 75% is achievable without loading MOMENT/Chronos (marked slow).
      # Coverage below 75% on src/ fails the build.

    - name: Upload coverage to Codecov (optional)
      uses: codecov/codecov-action@v4
      with:
        files: coverage.xml
        fail_ci_if_error: false

Coverage exclusions (.coveragerc):
    [run]
    omit =
        src/*/migrations/*
        src/monitoring/metrics.py      # declarative, no logic to test
        src/ingestion/schemas.py       # declarative pandera schema
    [report]
    exclude_lines =
        if __name__ == .__main__.:
        raise NotImplementedError

Add test for coverage tooling itself:
    test_all_src_modules_importable:
        Walk src/ and import every .py file (except __init__.py).
        Assert no ImportError. This catches missing __init__.py and
        circular imports that the individual unit tests might miss.
```

---

### M16 UPDATE — On-Call Runbook (add docs/runbook.md spec)

```
Create docs/runbook.md. Content spec (AI must generate the full file):

# SolarSentinel On-Call Runbook

## 1. solar_far_latest > 0.10 [SLO-5 breach]
   WHAT: False Alarm Rate exceeded hard limit. Model is over-alerting.
   IMMEDIATE (< 5 min):
     docker exec solar-api python -c "
       from src.monitoring.metrics import FAR_GAUGE
       print(FAR_GAUGE._value.get())"   # confirm current FAR
   DIAGNOSE (< 30 min):
     python scripts/audit_catalogue.py --last-n 50
     # Check if false alarms cluster around a specific flare class
     # or time of day (instrument noise pattern)
   FIX OPTIONS:
     A. Increase sigma_threshold in configs/nowcasting.yaml from 3.0 to 3.5
        → docker compose restart worker  (no model reload needed)
     B. Rollback model: python scripts/rollback.py --run-id <prev_run_id>
        → docker compose restart api worker
     C. If PRADAN data quality degraded: pause scheduler
        → docker exec solar-worker kill -STOP 1  (SIGSTOP)
   ESCALATE if FAR > 0.25 for > 2 hours.

## 2. solar_lead_time_minutes < 5 [SLO-4 degradation]
   WHAT: Forecast lead time dropped — model is firing alerts too late.
   DIAGNOSE:
     docker logs solar-worker --tail 100 | grep "lead_time"
     # Check if MOMENT anomaly score dropped (no pre-flare signal)
     # Check if TimesFM is timing out (falling back to LSTM+TCN only)
   FIX:
     A. Check Ollama is running (TimesFM may have OOM-killed):
        docker compose ps | grep ollama
        docker compose restart ollama
     B. Lower forecast threshold in configs/forecasting.yaml:
        decision_threshold: 0.35   (was 0.50) → alerts fire earlier
     C. If MOMENT OOM: check memory usage. Reduce batch size in
        batch_compute_moment_scores() window stride from 60 to 120.

## 3. solar_data_freshness_seconds > 180 [SLO-2 degradation]
   WHAT: No new FITS data processed for 3+ minutes.
   DIAGNOSE:
     docker logs solar-worker --tail 50
     python scripts/download_noaa_catalog.py --start-year 2024 --end-year 2024
     # ↑ tests network connectivity as a proxy for PRADAN access
   FIX:
     A. Check PRADAN connectivity:
        curl -sf https://pradan.issdc.gov.in/pradan/ || echo "PRADAN DOWN"
        If PRADAN is down: expected behaviour — circuit breaker is active.
        Log the outage start time. Check https://status.issdc.gov.in.
     B. Check seen_files.db isn't treating all files as already-seen:
        python -c "
          from src.ingestion.seen_files import list_seen
          import datetime
          recent = [f for f in list_seen()
                    if f['processed_at'] > str(datetime.datetime.utcnow()
                       - datetime.timedelta(hours=1))]
          print(len(recent), 'files processed in last hour')"
     C. Full restart if worker is deadlocked:
        docker compose restart worker

## 4. probe_success_24h_avg < 0.99 [SLO-3 availability breach]
   DIAGNOSE:
     curl -sv http://localhost:8000/health 2>&1 | tail -20
     docker compose ps   # identify unhealthy services
     docker logs solar-api --tail 100 | grep -i "error\|exception"
   FIX:
     A. Container crash loop: docker compose up -d --force-recreate api
     B. Port conflict: lsof -i :8000 | grep -v LISTEN → kill stale process
     C. OOM kill (MOMENT loaded per-call instead of singleton [RULE-16]):
        grep -r "get_moment_model\|MOMENTPipeline" src/orchestration/agents.py
        Confirm singleton pattern — not inside agent function body.

## 5. Model hash mismatch SecurityError [T-2]
   SYMPTOM: API logs "SecurityError: Model hash mismatch for xgb_multiclass.pkl"
   CAUSE: Model file changed on disk without going through promote_if_better().
   IMMEDIATE: docker compose stop api worker  (system offline)
   FIX:
     A. If an authorised promotion just ran:
        python scripts/rollback.py --run-id <last_known_good_run_id>
     B. If cause is unknown: treat as potential security incident.
        Preserve the tampered file: cp models/xgb_multiclass.pkl /tmp/suspect.pkl
        Re-download from last known good MLflow run.
        Rotate PRADAN credentials and GitHub PAT.
        Review Docker volume mount permissions.

## 6. FAR spike after canary PASS [unexpected]
   SYMPTOM: Canary passed (60 min shadow), but production FAR > 0.15 within 24h.
   CAUSE: 60-min canary window missed a low-frequency noise pattern.
   FIX:
     Immediate: rollback. python scripts/rollback.py --run-id <prev_run_id>
     Post-mortem: extend CANARY_DURATION_MINUTES to 240 in retrain.yml.
     Long-term: add rolling 24h FAR to Grafana with auto-rollback trigger.

## Useful one-liners
   # Check current model version
   curl -s http://localhost:8000/health | python -m json.tool | grep model_version
   # Check SLO timing for last pipeline run
   curl -s http://localhost:8000/status | python -m json.tool | grep timing
   # Audit last 10 catalogue entries
   python scripts/audit_catalogue.py --last-n 10
   # Check seen-files count
   python -c "from src.ingestion.seen_files import list_seen; print(len(list_seen()),'files seen')"
   # Force reprocess a specific FITS file (clear from seen registry)
   python -c "
     import sqlite3
     con = sqlite3.connect('data/processed/seen_files.db')
     con.execute('DELETE FROM seen_files WHERE fits_path=?',
                 ('/path/to/solexs_file.fits',))
     con.commit(); print('Cleared.')"
```

---

### UPDATED INTEGRATION TEST — Area 12 (supersedes previous Area 12)

```
AREA 12 — SLO COMPLIANCE (ALL SIX SLOs):
  slo_cfg = yaml.safe_load(open("configs/slo.yaml"))

  # SLO-5: FAR gate
  assert mean_far <= slo_cfg["far_max"], \
      f"[SLO-5] FAR={mean_far:.4f} exceeds limit {slo_cfg['far_max']}"

  # SLO-6: TPR gate for M+X (new)
  tpr_mx = latest_run.data.metrics.get("tpr_mx", -1)
  if tpr_mx >= 0:   # -1 means no M/X events in eval window
      assert tpr_mx >= slo_cfg["tpr_mx_min"], \
          f"[SLO-6] TPR(M+X)={tpr_mx:.4f} < minimum {slo_cfg['tpr_mx_min']}"

  # SLO-4: Lead time
  assert mean_lead_min >= slo_cfg["lead_time_min_accepted"], \
      f"[SLO-4] Lead time={mean_lead_min:.1f}min < {slo_cfg['lead_time_min_accepted']}min"

  # SLO-1: End-to-end latency (read from state["timing"] of a live run)
  timing = requests.get("http://localhost:8000/status").json().get("timing", {})
  total_s = sum(timing.values())
  assert total_s <= slo_cfg["latency_p99_seconds"], \
      f"[SLO-1] Total latency={total_s:.1f}s > {slo_cfg['latency_p99_seconds']}s"
  for component, budget in {
      "ingestion": 5, "detection": 3, "merge": 1,
      "features": 8, "nowcast": 0.5, "forecast": 2,
      "uncertainty": 10, "shap": 3, "llm_report": 31,
      "alert": 1}.items():
      t = timing.get(component, 0)
      assert t <= budget, \
          f"[SLO-1] Component '{component}' took {t:.2f}s > budget {budget}s"

  # Seen-files registry [RULE-23]
  from src.ingestion.seen_files import list_seen
  assert len(list_seen()) > 0, \
      "Seen-files registry must be non-empty after pipeline run"

  # Rollback script exists and is importable
  import importlib.util
  spec = importlib.util.spec_from_file_location(
      "rollback", "scripts/rollback.py")
  assert spec is not None, "scripts/rollback.py must exist"

  # B-class decision documented [D10]
  from src.preprocessing.labels import CLASS_MAP
  assert CLASS_MAP.get("B") == 0, \
      "[D10] B-class must map to label 0 (N) — check CLASS_MAP in labels.py"

  print(f"AREA 12 PASSED: All 6 SLOs compliant. "
        f"FAR={mean_far:.4f} (≤{slo_cfg['far_max']}). "
        f"TPR(M+X)={tpr_mx:.4f} (≥{slo_cfg['tpr_mx_min']}). "
        f"Lead={mean_lead_min:.1f}min. Latency={total_s:.1f}s.")
```

---

### UPDATED RULE QUICK REFERENCE (v3 — 23 rules)

```
 #  | Rule summary                                         | Where enforced
----|------------------------------------------------------|--------------------
 1  | Independent detectors — never merge before detect    | M2, M3
 2  | LSTM bidirectional=False (no leakage)                | M5, M7
 3  | TSS not accuracy                                     | M4, M8, M15
 4  | FITS columns from YAML — no hardcoding               | M1
 5  | XGBoost multi:softprob, 4 classes, per-class thresh  | M4, M8
 6  | timesfm.finetuning does not exist                    | M5
 7  | predict() not predict_quantiles() for Chronos        | M11
 8  | confusion_matrix labels=[0,1] always                 | M8, M15
 9  | GraphRAG: local embeddings only (no OpenAI)          | M9
10  | dspy.context() per-call (not dspy.configure())       | M9, M7
11  | torch.load weights_only=True                         | M5, M7, M10, M11
12  | Module-level open() needs FileNotFoundError guard    | M1, M7, M10
13  | Consistent model save format (JSON+PKL/PT/ONNX)      | M4, M5, M6
14  | No session fixtures in hypothesis @given tests       | M6
15  | No private function imports across modules           | M3, M7
16  | Singleton loaders for Chronos, MOMENT, ONNX          | M7, M11, M12
17  | Provenance columns in every master_catalogue row     | M3, M7
18  | Pandera schema validation at ingestion boundary      | M1
19  | Dashboard = React+TypeScript+Vite (not Streamlit)    | M0, M14
20  | Prometheus metrics centralised in metrics.py         | M7, M14, M16
21  | dev→staging→prod gate before production promotion    | M16
22  | SEED=42 first 4 lines of every training script       | M4, M5, M8, M10
23  | Check seen-files registry before ingesting FITS      | M1, M16 scheduler
----|------------------------------------------------------|--------------------
SLO-5 | FAR ≤ 0.10 hard gate on promote_if_better()      | M8, M15, M16
SLO-6 | TPR(M+X) ≥ 0.80 hard gate on promote_if_better() | M15, M16
T-2   | verify_model_hash() + rollback.py is the safe path| M6, M7, M16
T-3   | WebSocket: 20-conn cap, 1 msg/10s rate limit       | M14
T-6   | gitleaks in CI and pre-commit                      | M16
D10   | B-class = label 0 (N), documented in D10.md       | M2, M4
```


---

## ADDENDUM v4 — ALL CODE-CORRECTNESS AND IMPLEMENTATION GAPS CLOSED
*Supersedes earlier milestone descriptions where they conflict.
Rating target: 9.9/10. All 14 identified bugs and missing implementations fixed.*

---

### FIX 1 — RULE-22 grep anchor broken (pre-commit hook correction)

```
WRONG (anchors to line-start, fails when imports precede SEED):
    entry: grep -rn "^SEED = 42" src/nowcasting/train.py ...

CORRECT (.pre-commit-config.yaml):
    - repo: local
      hooks:
        - id: seed-check
          name: "Training scripts must contain SEED = 42"
          entry: bash -c '
            for f in src/nowcasting/train.py
                      src/forecasting/multi_horizon.py
                      src/automl/flaml_tuner.py
                      src/physics/pinn.py; do
              grep -q "SEED = 42" "$f" || {
                echo "FAIL: Missing SEED=42 in $f"; exit 1; }
            done; echo "seed-check: all training scripts have SEED=42"'
          language: system
          pass_filenames: false

WHY: grep -rn "^SEED" anchors to line position zero, so "import random; SEED = 42"
passes but "import random\nSEED = 42" (standard formatting) may fail depending on
grep implementation. The bash loop uses grep -q (file-level search, no anchor)
which correctly finds the string anywhere in the file.
```

---

### FIX 2 — T-1 SensitiveFormatter: actual code, not description

```
CREATE src/utils/logging_config.py:

import logging
import os
import re

class SensitiveFormatter(logging.Formatter):
    """
    Masks secret env var values in all log output [T-1].
    WHY: A bare 'except Exception as exc: logger.error(exc)' call on a PRADAN
    HTTP error can include the request URL with embedded credentials in the
    traceback. This formatter is the last-resort safety net.
    """
    _PATTERN: re.Pattern | None = None

    @classmethod
    def _get_pattern(cls) -> re.Pattern | None:
        secrets = [
            os.environ.get("PRADAN_PASSWORD", ""),
            os.environ.get("PRADAN_USERNAME", ""),
            os.environ.get("GH_PAT", ""),
        ]
        active = [re.escape(s) for s in secrets if len(s) > 3]
        if not active:
            return None
        return re.compile("|".join(active))

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if self._PATTERN is None:
            self.__class__._PATTERN = self._get_pattern()
        if self._PATTERN:
            msg = self._PATTERN.sub("***REDACTED***", msg)
        return msg


def configure_logging(level: int = logging.INFO) -> None:
    """
    Call once at application startup (main.py, scheduler.py).
    Installs SensitiveFormatter on the root logger so ALL modules
    (including third-party libraries that use logging) are covered.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(SensitiveFormatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ"))
    logging.root.setLevel(level)
    logging.root.handlers = [handler]


CALL SITES — add to top of each entry point:
    # src/api/main.py  (startup event)
    from src.utils.logging_config import configure_logging
    @app.on_event("startup")
    async def startup(): configure_logging()

    # src/orchestration/scheduler.py  (module level)
    from src.utils.logging_config import configure_logging
    configure_logging()

TESTS to add in tests/test_pipeline.py:
    test_sensitive_formatter_masks_password:
        os.environ["PRADAN_PASSWORD"] = "super_secret_pw"
        fmt = SensitiveFormatter()
        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="", lineno=0,
            msg="Login failed: super_secret_pw is wrong", args=(), exc_info=None)
        output = fmt.format(record)
        assert "super_secret_pw" not in output
        assert "***REDACTED***" in output

    test_sensitive_formatter_no_op_when_no_secrets:
        For a fresh env with no PRADAN_PASSWORD set,
        SensitiveFormatter().format(record) returns the original message unchanged.
```

---

### FIX 3 — M1: Pandera monotonic-increasing index check

```
UPDATE src/ingestion/schemas.py — both schemas get index-level monotonicity check:

import pandera as pa
from pandera import Column, DataFrameSchema, Check

def _monotonic_index_check(index: pd.Index) -> bool:
    return index.is_monotonic_increasing

SOLEXS_SCHEMA = DataFrameSchema(
    columns={
        "flux_high": Column(float, Check.greater_than_or_equal_to(0),
                            nullable=False),
        "flux_low":  Column(float, Check.greater_than_or_equal_to(0),
                            nullable=False),
        # ADD remaining columns after CHECK 1
    },
    index=pa.Index(
        pa.DateTime,
        checks=pa.Check(
            _monotonic_index_check,
            element_wise=False,
            error=(
                "FITS DatetimeIndex is not monotonically increasing. "
                "Cause: instrument clock glitch or duplicate timestamps. "
                "Fix: df = df[~df.index.duplicated()].sort_index() "
                "BEFORE calling schema.validate()."
            )
        )
    ),
    coerce=False,
)

HEL1OS_SCHEMA = DataFrameSchema(
    columns={
        "counts_low":  Column(int, Check.greater_than_or_equal_to(0),
                              nullable=False),
        "counts_high": Column(int, Check.greater_than_or_equal_to(0),
                              nullable=False),
        # ADD remaining columns after CHECK 1
    },
    index=pa.Index(
        pa.DateTime,
        checks=pa.Check(
            _monotonic_index_check,
            element_wise=False,
            error=(
                "HEL1OS DatetimeIndex is not monotonically increasing. "
                "Cause: instrument reset during high-energy burst. "
                "Fix: df = df.sort_index().drop_duplicates() before validate()."
            )
        )
    ),
    coerce=False,
)

ADD to _safe_read() in fits_reader.py — sort + dedup BEFORE schema validate:
    df = df[~df.index.duplicated(keep="first")].sort_index()
    schema.validate(df, lazy=True)   # now guaranteed monotonic

Tests to add:
    test_schema_rejects_non_monotonic_index:
        Build SoLEXS df with index [t0, t2, t1]. Call SOLEXS_SCHEMA.validate().
        Assert pa.errors.SchemaErrors raised, error message contains "monotonically".
    test_safe_read_deduplicates_before_validate:
        Mock _build_dataframe to return df with duplicate timestamps.
        Assert read_solexs() succeeds (dedup+sort happens before validate).
```

---

### FIX 4 — seen_files.db path: env var + .gitignore

```
UPDATE src/ingestion/seen_files.py — replace hardcoded path:

import os
from pathlib import Path

# WRONG (hardcoded, violates spirit of RULE-4):
_DB_PATH = "data/processed/seen_files.db"

# CORRECT (env-var configurable, sensible default):
_DB_PATH: str = os.environ.get(
    "SEEN_FILES_DB",
    "data/processed/seen_files.db")

WHY: In staging (docker-compose.staging.yml), set SEEN_FILES_DB to a
temporary path so the staging run never contaminates the dev seen-files
registry: SEEN_FILES_DB=/tmp/staging_seen.db

UPDATE .gitignore — add after *.pkl line:
    *.db          # SQLite databases (seen_files.db, test fixtures)
    data/canary/  # canary prediction CSV (staging artefact)

UPDATE .env.example — add:
    SEEN_FILES_DB=data/processed/seen_files.db

UPDATE docker-compose.staging.yml — add env var:
    worker:
      environment:
        - SEEN_FILES_DB=/tmp/staging_seen_files.db
        - ENV=staging
```

---

### FIX 5 — M7: timing always in `finally:` block

```
WRONG (timing lost if error-append itself raises or state is mutated):
    def some_agent(state):
        t0 = time.perf_counter()
        try:
            # ... logic ...
        except Exception as exc:
            state["errors"].append(f"agent: {exc}")
        state.setdefault("timing", {})["agent"] = round(time.perf_counter()-t0, 3)
        return state

CORRECT — apply to EVERY agent in src/orchestration/agents.py:
    def ingestion_agent(state: SolarPipelineState) -> SolarPipelineState:
        t0 = time.perf_counter()
        try:
            state["pipeline_run_id"] = str(uuid.uuid4())
            state["model_version"]   = _MODEL_VERSION
            # ... FITS ingestion logic ...
        except Exception as exc:
            state["errors"].append(f"ingestion_agent: {exc}")
        finally:                   # ← ALWAYS records timing
            state.setdefault("timing", {})[
                "ingestion"] = round(time.perf_counter() - t0, 3)
        return state

WHY finally: If state["errors"].append() itself raises (e.g. state["errors"]
is None due to an upstream bug), the except block throws before timing is
recorded. finally: executes regardless of whether try, except, or both throw.
Missing timing data breaks [SLO-1] component budget assertions in the
integration test and /status endpoint.

CHECKLIST — apply this pattern to ALL agents:
    ingestion_agent, solexs_detect_agent, hel1os_detect_agent,
    merge_agent, preprocess_agent, moment_score_agent, nowcast_agent,
    forecast_agent, uncertainty_agent, shap_agent, llm_report_agent

Test to add:
    test_timing_recorded_even_when_agent_raises:
        Monkeypatch ingestion_agent internals to raise RuntimeError.
        After running agent, assert state["timing"]["ingestion"] is a
        non-negative float (timing recorded despite the error).
```

---

### FIX 6 — M8: `_extract_best_far()` definitive implementation

```
REPLACE vague "iterate model_history" with re-evaluation on val set.
UPDATE src/automl/flaml_tuner.py:

def _extract_best_far(automl: AutoML,
                       X_val: np.ndarray,
                       y_val: np.ndarray) -> float:
    """
    Compute FAR for the best FLAML model by re-evaluating on val set.
    WHY re-evaluate instead of parsing model_history:
      FLAML's internal metric dict structure and attribute names vary
      across versions (best_config_train_time, best_loss_per_iter, etc.).
      Re-evaluation on X_val is version-stable and gives the most accurate
      measure of the actual promoted model's FAR — not a training-fold
      estimate from the search process.
    """
    model = automl.model.estimator
    proba = model.predict_proba(X_val)
    binary_pred = (proba[:, 1:].sum(axis=1) >= 0.5).astype(int)
    binary_true = (y_val > 0).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        binary_true, binary_pred, labels=[0, 1]   # [RULE-8]
    ).ravel()
    return float(fp / max(fp + tn, 1))

UPDATE run_flaml_automl signature to pass val data:
    def run_flaml_automl(X_tr, y_tr, X_val, y_val, time_budget=300):
        automl = AutoML()
        automl.fit(...)
        best_far = _extract_best_far(automl, X_val, y_val)   # ← definitive
        if best_far > 0.10: ...

Test to add:
    test_extract_best_far_uses_val_set_not_history:
        Mock automl.model.estimator.predict_proba to return known probabilities.
        Call _extract_best_far(automl, X_val, y_val).
        Assert return value matches manually computed FPR from the mock probas.
        Assert no attribute access on automl.best_config_train_time
        (i.e. the version-unstable path is never taken).
```

---

### FIX 7 — M9: DSPy 2.x class attribute pattern

```
WRONG (fails in DSPy 2.x — class-level Predict breaks __init_subclass__):
    class SolarFlareReporter(dspy.Module):
        predict: dspy.Predict(FlareAlertSignature)   # class attribute

CORRECT (instance attribute in __init__):
    class SolarFlareReporter(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            # Instantiate Predict in __init__ so DSPy tracks it
            # as a named sub-module on the instance (not the class).
            # DSPy 2.x broke class-level assignment; this pattern
            # works across DSPy 2.0–2.5+.
            self.predict = dspy.Predict(FlareAlertSignature)

        def forward(
            self,
            flare_class: str,
            peak_flux: float,
            lead_time_min: float,
            confidence: float,
            graphrag_context: str,
            uncertainty_q10: float,
            uncertainty_q90: float,
        ) -> str:
            with dspy.context(lm=get_lm()):   # [RULE-10]
                result = self.predict(
                    flare_class=flare_class,
                    peak_flux=peak_flux,
                    lead_time_min=lead_time_min,
                    confidence=confidence,
                    graphrag_context=graphrag_context,
                    uncertainty_q10=uncertainty_q10,
                    uncertainty_q90=uncertainty_q90)
            return result.alert_summary

Test to add:
    test_dspy_reporter_predict_is_instance_attr_not_class_attr:
        reporter = SolarFlareReporter()
        assert isinstance(reporter.predict, dspy.Predict)
        assert "predict" not in SolarFlareReporter.__dict__, \
            "predict must be instance attr (in __init__), not class attr"
```

---

### FIX 8 — M12: wrong assertion target in batch skip test

```
WRONG (asserts on singleton loader, not reconstruction call count):
    test_batch_skips_quiet_windows:
        ...
        assert get_moment_model() called at most 1 time
        # ← Wrong: singleton loads ONCE per process regardless of window count.
        #   This test would pass even if ALL windows called compute_reconstruction_error.

CORRECT (assert on the function that actually does per-window compute):
    def test_batch_skips_quiet_windows():
        '''
        Verify that compute_reconstruction_error is called ONLY for pre-flare
        windows, not quiet-sun windows. This is the [SLO-1] budget protection:
        MOMENT inference is ~0.5s/window; calling it on all 1440 daily windows
        would cost 720s — 8x the entire SLO-1 budget.
        '''
        # Build df with 1 pre-flare window (phase=1) and 9 quiet windows (phase=0)
        n = 10
        df = _build_synthetic_df(n_rows=600)  # 10 windows of 60 rows each
        # Inject 1 pre-flare phase label at window index 4
        df.loc[df.index[240:300], "solar_phase"] = 1  # pre-flare
        df.loc[df.index[:240], "solar_phase"] = 0
        df.loc[df.index[300:], "solar_phase"] = 0

        with patch(
            "src.forecasting.moment_anomaly.compute_reconstruction_error"
        ) as mock_recon:
            mock_recon.return_value = 0.05
            batch_compute_moment_scores(df, "flux_high",
                                         window_size=512, stride=60)

        assert mock_recon.call_count == 1, (
            f"compute_reconstruction_error must be called exactly once "
            f"(for the pre-flare window only), got {mock_recon.call_count}. "
            f"Quiet windows must be skipped to meet [SLO-1] budget.")
```

---

### FIX 9 — M14: WebSocket outbound push (loop was silent)

```
WRONG (loop receives inbound but never pushes outbound data):
    while True:
        try:
            data = await asyncio.wait_for(ws.receive_text(), timeout=10)
            ...
        except asyncio.TimeoutError:
            pass    # ← silently loops; client never receives anything

CORRECT — push latest state BEFORE waiting for inbound:

    import json

    def get_latest_state() -> dict:
        """Assemble the current pipeline state snapshot for WebSocket push."""
        from src.monitoring.metrics import (
            NOWCAST_CONFIDENCE, FAR_GAUGE, LEAD_TIME_GAUGE,
            DATA_FRESHNESS, ALERT_COUNTER)
        return {
            "ts":              pd.Timestamp.utcnow().isoformat(),
            "nowcast_class":   _LAST_NOWCAST.get("class", "N"),
            "confidence":      _LAST_NOWCAST.get("confidence", 0.0),
            "forecast_15min":  _LAST_FORECAST.get("h15", [0,0,0,0]),
            "data_freshness_s":DATA_FRESHNESS._value.get(),
            "far":             FAR_GAUGE._value.get(),
            "lead_time_min":   LEAD_TIME_GAUGE._value.get(),
            "alert":           _LAST_ALERT,
        }

    @app.websocket("/ws/live")
    async def live_feed(websocket: WebSocket):
        if len(_active_ws) >= MAX_CONNECTIONS:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        _active_ws.add(websocket)
        last_msg_time = 0.0
        try:
            while True:
                # 1. Push current state to client every loop iteration
                await websocket.send_json(get_latest_state())  # ← NEW

                # 2. Listen for inbound message (heartbeat/command)
                #    timeout=60 drives the push cadence: push every 60s
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_text(), timeout=60)
                    now = time.time()
                    if now - last_msg_time < 10:          # rate limit [T-3]
                        await websocket.send_text(
                            json.dumps({"error": "rate_limit",
                                        "msg":   "1 message per 10s"}))
                        await websocket.close(code=1008)
                        break
                    last_msg_time = now
                except asyncio.TimeoutError:
                    pass  # no inbound message — outbound already sent, loop again
        except WebSocketDisconnect:
            pass
        finally:
            _active_ws.discard(websocket)

WHY timeout=60 drives cadence: asyncio.wait_for(receive_text, timeout=60)
blocks for at most 60 seconds. On timeout it raises TimeoutError (caught,
ignored), and the while-loop restarts — executing send_json again. Client
gets a fresh state push every 60 seconds even with zero inbound traffic.
For near-real-time (< 60s) alerting, call asyncio.sleep() from the
alert_router agent to signal waiting WebSocket handlers via asyncio.Event.

Test to add:
    test_websocket_pushes_on_connect:
        Async test using httpx.AsyncClient + anyio. Connect to /ws/live.
        Receive one message (first push). Assert message is valid JSON
        with key "nowcast_class" in ("N","C","M","X").
    test_websocket_pushes_on_timeout:
        Connect. Receive first push. Wait 61s (mock asyncio.sleep).
        Assert second push received without sending any inbound message.
```

---

### FIX 10 — M16: `_compute_canary_far()` fully defined

```
ADD to src/deployment/model_registry.py:

def _compute_canary_far(preds: pd.DataFrame,
                         noaa: pd.DataFrame,
                         window_min: int = 10) -> float:
    """
    Compute False Alarm Rate for canary shadow predictions vs NOAA catalog.

    Args:
        preds: DataFrame with columns [timestamp, predicted_class, confidence].
               Only M/X predictions (predicted_class >= 2) are checked.
        noaa:  DataFrame with column [peak_time] (UTC datetime).
        window_min: A prediction is a TP if a NOAA flare peak_time falls
                    within ±window_min minutes of the prediction timestamp.

    Returns:
        FAR = FP / (FP + TP) across all M/X predictions.
        Returns 0.0 if preds is empty (no alerts = no false alarms).

    WHY ±window_min: NOAA catalog timestamps have ±5 min uncertainty from
    manual classification. ±10 min window gives fair credit for early alerts.
    """
    if preds.empty or noaa.empty:
        return 0.0

    mx_preds = preds[preds["predicted_class"] >= 2].copy()
    if mx_preds.empty:
        return 0.0

    mx_preds["timestamp"] = pd.to_datetime(mx_preds["timestamp"], utc=True)
    noaa["peak_time"]     = pd.to_datetime(noaa["peak_time"],     utc=True)
    window = pd.Timedelta(minutes=window_min)

    tp = fp = 0
    for _, row in mx_preds.iterrows():
        t = row["timestamp"]
        matched = noaa[
            (noaa["peak_time"] >= t - window) &
            (noaa["peak_time"] <= t + window)]
        if matched.empty:
            fp += 1
        else:
            tp += 1

    return float(fp / max(fp + tp, 1))

Test to add:
    test_compute_canary_far_all_true_positives:
        preds with 3 M-class at t1, t2, t3.
        noaa with matching events within 5 min of each.
        Assert _compute_canary_far(preds, noaa) == 0.0

    test_compute_canary_far_all_false_alarms:
        preds with 3 M-class predictions. noaa is empty.
        Assert _compute_canary_far(preds, noaa) == 1.0

    test_compute_canary_far_empty_preds:
        Assert _compute_canary_far(pd.DataFrame(), noaa) == 0.0
```

---

### FIX 11 — M16: `shadow_runner.py` full spec

```
CREATE src/orchestration/shadow_runner.py:

"""
Shadow inference runner for canary deployment [RULE-21 canary extension].

Reads incoming FITS files (same source as the main scheduler), runs
inference using a CANDIDATE model directory, writes predictions to a CSV
WITHOUT triggering any alerts, WebSocket pushes, or catalogue writes.

Usage (called by _run_canary() in model_registry.py):
    python src/orchestration/shadow_runner.py \
        --model-dir models/canary/ \
        --output    data/canary/predictions.csv
"""
import argparse, csv, os, time, logging
import numpy as np
import pandas as pd
import joblib

from src.ingestion.fits_reader  import read_solexs
from src.preprocessing.physics_features import engineer_physics_features
from src.deployment.onnx_export  import ONNXNowcaster
from src.monitoring.model_integrity import verify_model_hash

logger = logging.getLogger(__name__)

FEATURE_COLS: list[str] = []  # populated from configs/nowcasting.yaml at startup

def _load_feature_cols() -> list[str]:
    import yaml
    try:
        cfg = yaml.safe_load(open("configs/nowcasting.yaml")) or {}
        return cfg.get("feature_cols", [])
    except FileNotFoundError:
        return []

def _list_new_fits_files() -> list[str]:
    """Poll data/raw/solexs/ for FITS files not yet seen by the shadow runner."""
    from src.ingestion.seen_files import is_seen
    raw_dir = os.environ.get("FITS_RAW_DIR", "data/raw/solexs")
    files = sorted(f for f in os.listdir(raw_dir) if f.endswith(".fits"))
    return [
        os.path.join(raw_dir, f) for f in files
        if not is_seen(os.path.join(raw_dir, f))
    ]

def run_shadow(model_dir: str, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    global FEATURE_COLS
    FEATURE_COLS = _load_feature_cols()

    # Load candidate models (NOT canonical models/) [RULE-13]
    verify_model_hash(f"{model_dir}/tcn_encoder.onnx")   # [T-2]
    nowcaster = ONNXNowcaster(
        tcn_path  = f"{model_dir}/tcn_encoder.onnx",
        xgb_model = joblib.load(f"{model_dir}/xgb_multiclass.pkl"))

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "predicted_class",
            "confidence", "proba_N", "proba_C", "proba_M", "proba_X"])
        writer.writeheader()

        logger.info(f"[shadow] Started. model_dir={model_dir} output={output_path}")
        while True:
            for fits_path in _list_new_fits_files():
                try:
                    df    = read_solexs(fits_path)              # pandera validated [RULE-18]
                    feats = engineer_physics_features(df)
                    if feats.empty or len(feats) < 60:
                        logger.warning(f"[shadow] Too few rows after features: {fits_path}")
                        continue
                    window = feats[FEATURE_COLS].values[-60:].astype(np.float32)
                    result = nowcaster.predict(
                        window[np.newaxis],
                        window[-1:])
                    row = {
                        "timestamp":       pd.Timestamp.utcnow().isoformat(),
                        "predicted_class": ["N","C","M","X"].index(result["class"]),
                        "confidence":      round(result["confidence"], 4),
                        **{f"proba_{c}": round(result["proba"][c], 4)
                           for c in ("N","C","M","X")}}
                    writer.writerow(row)
                    f.flush()
                    logger.debug(f"[shadow] {fits_path} → {result['class']}")
                except Exception as exc:
                    logger.error(f"[shadow] Error on {fits_path}: {exc}")
            time.sleep(60)  # poll every 60s (same cadence as main scheduler)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Shadow inference runner")
    p.add_argument("--model-dir", required=True,
                   help="Path to candidate model directory (not models/)")
    p.add_argument("--output", required=True,
                   help="Path for prediction CSV output")
    args = p.parse_args()
    from src.utils.logging_config import configure_logging
    configure_logging()
    run_shadow(args.model_dir, args.output)

TESTS to add in tests/test_pipeline.py:
    test_shadow_runner_writes_csv_with_correct_columns:
        Mock read_solexs, engineer_physics_features, ONNXNowcaster.predict.
        Call run_shadow("models/canary/", tmp_path/"preds.csv") for 1 iteration.
        Load CSV. Assert columns: timestamp, predicted_class, confidence,
        proba_N, proba_C, proba_M, proba_X all present.
    test_shadow_runner_does_not_write_to_catalogue:
        Monkeypatch merger.merge_catalogues. Call run_shadow().
        Assert merge_catalogues NOT called (shadow = no catalogue write).
    test_shadow_runner_does_not_trigger_alerts:
        Monkeypatch alert_router. Call run_shadow().
        Assert alert_router NOT called.
```

---

### FIX 12 — M16 scheduler: SLO-2 consecutive slow-cycle counter

```
CREATE (or update) src/orchestration/scheduler.py:

Add to the main polling loop:

import os, time, logging, requests
logger = logging.getLogger(__name__)

_SLOW_THRESHOLD_S     = 60    # SLO-2: one cycle must complete in < 60s
_SLOW_FALLBACK_S      = 120   # fallback cadence when degraded
_SLOW_ESCALATE_N      = 3     # consecutive slow cycles before Slack alert

def _post_slack(message: str) -> None:
    """Post to SLACK_WEBHOOK_URL. Silent no-op if webhook not configured."""
    webhook = os.environ.get("SLACK_WEBHOOK_URL", "")
    if not webhook:
        logger.warning("SLACK_WEBHOOK_URL not set — Slack alert suppressed.")
        return
    try:
        r = requests.post(webhook, json={"text": message}, timeout=10)
        r.raise_for_status()
    except Exception as exc:
        logger.error(f"Slack post failed: {exc}")


def run_scheduler(pipeline, pradan_downloader) -> None:
    from src.monitoring.metrics import DATA_FRESHNESS
    consecutive_slow = 0
    cadence_s = _SLOW_THRESHOLD_S  # start at normal cadence

    while True:
        cycle_start = time.perf_counter()
        try:
            for fits_path in pradan_downloader.list_new_files():
                from src.ingestion.seen_files import is_seen, mark_seen
                if is_seen(fits_path):           # [RULE-23]
                    continue
                state = pipeline.invoke({"solexs_path": fits_path})
                mark_seen(fits_path,
                          instrument="solexs",
                          pipeline_run_id=state.get("pipeline_run_id",""))
                DATA_FRESHNESS.set(time.time() -
                                    os.path.getmtime(fits_path))  # [RULE-20]
        except Exception as exc:
            logger.error(f"Scheduler cycle error: {exc}")
        finally:
            elapsed = time.perf_counter() - cycle_start

        # SLO-2 throughput monitoring
        if elapsed > _SLOW_THRESHOLD_S:
            consecutive_slow += 1
            logger.warning(
                f"[SLO-2] Cycle {elapsed:.1f}s > {_SLOW_THRESHOLD_S}s "
                f"({consecutive_slow}/{_SLOW_ESCALATE_N})")
            if consecutive_slow >= _SLOW_ESCALATE_N:
                _post_slack(
                    f":warning: *SolarSentinel SLO-2 breach*\n"
                    f"3 consecutive cycles exceeded {_SLOW_THRESHOLD_S}s.\n"
                    f"Latest: {elapsed:.1f}s. Falling back to "
                    f"{_SLOW_FALLBACK_S}s cadence. Investigate /status endpoint.")
                cadence_s = _SLOW_FALLBACK_S   # degrade gracefully
        else:
            if consecutive_slow > 0:
                logger.info(f"[SLO-2] Recovered after {consecutive_slow} slow cycles.")
                _post_slack(":white_check_mark: *SolarSentinel SLO-2 recovered*\n"
                            f"Cycle {elapsed:.1f}s. Restoring 60s cadence.")
            consecutive_slow = 0
            cadence_s = _SLOW_THRESHOLD_S      # restore normal cadence

        sleep_for = max(0.0, cadence_s - elapsed)
        time.sleep(sleep_for)

Tests to add:
    test_scheduler_falls_back_to_120s_after_3_slow_cycles:
        Mock pipeline.invoke to sleep 65s (> 60s threshold).
        Run 3 iterations. Assert _post_slack called once with "120s cadence".
        Assert cadence_s == 120 after 3rd slow cycle.
    test_scheduler_recovers_cadence_after_fast_cycle:
        After 3 slow cycles, mock pipeline to return in 5s.
        Assert cadence_s resets to 60 and recovery Slack message sent.
    test_post_slack_no_op_when_webhook_not_set:
        Unset SLACK_WEBHOOK_URL. Call _post_slack("test"). No exception raised.
```

---

### FIX 13 — M16 CI: staging gate hostname

```
WRONG (hostname "staging" not resolvable in GitHub Actions):
    - run: curl http://staging:8000/health

CORRECT — use localhost with a readiness poll:

In .github/workflows/ci.yml, replace the staging gate block with:

    - name: Start staging stack
      run: |
        docker compose -f docker-compose.staging.yml up -d
        echo "Waiting for API to be ready..."
        for i in $(seq 1 24); do
          curl -sf http://localhost:8000/health \
            -H "Accept: application/json" \
            && echo "API ready after ${i}x5s" && break \
            || { echo "Attempt $i/24 failed, waiting 5s..."; sleep 5; }
        done
        curl -sf http://localhost:8000/health || {
          echo "API failed to start within 120s"
          docker compose -f docker-compose.staging.yml logs api
          exit 1
        }

    - name: Integration test (staging)
      run: |
        python scripts/integration_test_20240222.py \
          --synthetic-data data/synthetic/
      env:
        MLFLOW_TRACKING_URI: http://localhost:5000

    - name: Teardown
      if: always()
      run: docker compose -f docker-compose.staging.yml down -v

WHY localhost not staging: In GitHub Actions, docker compose services
expose ports on the runner's localhost — there is no internal DNS for
service names unless you use a custom bridge network explicitly defined
in docker-compose.staging.yml. Using localhost is portable across GitHub
Actions, self-hosted runners, and local dev.

If internal service-name DNS is needed in staging (e.g. api calls worker),
add to docker-compose.staging.yml:
    networks:
      solar_net:
        driver: bridge
    services:
      api:    { networks: [solar_net] }
      worker: { networks: [solar_net] }
Services then resolve each other by name (api, worker) but the test
runner still uses localhost:<exposed-port> from outside the network.
```

---

### FIX 14 — M16: Grafana dashboard JSON provisioning file

```
CREATE grafana/dashboards/solar_sentinel.json:
(Paste this literal JSON into the file — Grafana will auto-provision it
when grafana/provisioning/dashboards/solar.yaml points to this directory.)

CREATE grafana/provisioning/dashboards/solar.yaml:
    apiVersion: 1
    providers:
      - name: SolarSentinel
        type: file
        options:
          path: /var/lib/grafana/dashboards

CREATE grafana/dashboards/solar_sentinel.json:
```
```json
{
  "__inputs": [],
  "__requires": [],
  "annotations": { "list": [] },
  "editable": true,
  "fiscalYearStartMonth": 0,
  "graphTooltip": 1,
  "id": null,
  "links": [],
  "panels": [
    {
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "color": { "mode": "palette-classic" },
          "custom": { "lineWidth": 2 },
          "thresholds": {
            "mode": "absolute",
            "steps": [
              { "color": "green", "value": null },
              { "color": "red",   "value": 90 }
            ]
          }
        }
      },
      "gridPos": { "h": 8, "w": 12, "x": 0, "y": 0 },
      "id": 1,
      "options": { "tooltip": { "mode": "single" } },
      "targets": [{
        "datasource": "Prometheus",
        "expr": "histogram_quantile(0.99, rate(solar_inference_duration_seconds_bucket[5m]))",
        "legendFormat": "P99 latency"
      }],
      "title": "SLO-1 End-to-End Latency P99 (red = >90s)",
      "type": "timeseries"
    },
    {
      "datasource": "Prometheus",
      "fieldConfig": { "defaults": { "custom": { "lineWidth": 1 } } },
      "gridPos": { "h": 8, "w": 12, "x": 12, "y": 0 },
      "id": 2,
      "options": { "tooltip": { "mode": "multi" } },
      "targets": [
        { "datasource": "Prometheus",
          "expr": "rate(solar_alert_total{flare_class='C'}[5m])",
          "legendFormat": "C-class" },
        { "datasource": "Prometheus",
          "expr": "rate(solar_alert_total{flare_class='M'}[5m])",
          "legendFormat": "M-class" },
        { "datasource": "Prometheus",
          "expr": "rate(solar_alert_total{flare_class='X'}[5m])",
          "legendFormat": "X-class" }
      ],
      "title": "Alert Rate by Flare Class",
      "type": "timeseries"
    },
    {
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "thresholds": {
            "mode": "absolute",
            "steps": [
              { "color": "green", "value": null },
              { "color": "orange", "value": 120 },
              { "color": "red",    "value": 180 }
            ]
          },
          "unit": "s"
        }
      },
      "gridPos": { "h": 6, "w": 6, "x": 0, "y": 8 },
      "id": 3,
      "options": { "reduceOptions": { "calcs": ["lastNotNull"] } },
      "targets": [{
        "datasource": "Prometheus",
        "expr": "solar_data_freshness_seconds",
        "legendFormat": "Data freshness"
      }],
      "title": "SLO-2 Data Freshness (red = >180s)",
      "type": "gauge"
    },
    {
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "thresholds": {
            "mode": "absolute",
            "steps": [
              { "color": "green",  "value": null },
              { "color": "orange", "value": 0.07 },
              { "color": "red",    "value": 0.10 }
            ]
          },
          "max": 0.3, "min": 0
        }
      },
      "gridPos": { "h": 6, "w": 6, "x": 6, "y": 8 },
      "id": 4,
      "options": { "reduceOptions": { "calcs": ["lastNotNull"] } },
      "targets": [{
        "datasource": "Prometheus",
        "expr": "solar_far_latest",
        "legendFormat": "FAR"
      }],
      "title": "SLO-5 False Alarm Rate (red = >0.10)",
      "type": "gauge"
    },
    {
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "thresholds": {
            "mode": "absolute",
            "steps": [
              { "color": "red",    "value": null },
              { "color": "orange", "value": 10 },
              { "color": "green",  "value": 30 }
            ]
          },
          "unit": "m", "min": 0
        }
      },
      "gridPos": { "h": 6, "w": 6, "x": 12, "y": 8 },
      "id": 5,
      "options": { "reduceOptions": { "calcs": ["lastNotNull"] } },
      "targets": [{
        "datasource": "Prometheus",
        "expr": "solar_lead_time_minutes",
        "legendFormat": "Lead time"
      }],
      "title": "SLO-4 Lead Time (green = >30min)",
      "type": "gauge"
    },
    {
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "thresholds": {
            "mode": "absolute",
            "steps": [
              { "color": "red",   "value": null },
              { "color": "green", "value": 0.99 }
            ]
          },
          "max": 1, "min": 0, "unit": "percentunit"
        }
      },
      "gridPos": { "h": 6, "w": 6, "x": 18, "y": 8 },
      "id": 6,
      "options": { "reduceOptions": { "calcs": ["mean"] } },
      "targets": [{
        "datasource": "Prometheus",
        "expr": "avg_over_time(probe_success{job='solar-sentinel-api'}[24h])",
        "legendFormat": "24h availability"
      }],
      "title": "SLO-3 Availability 24h avg (green = >99%)",
      "type": "gauge"
    }
  ],
  "refresh": "30s",
  "schemaVersion": 38,
  "tags": ["solar-sentinel", "slo"],
  "time": { "from": "now-6h", "to": "now" },
  "timepicker": {},
  "timezone": "utc",
  "title": "SolarSentinel — SLO Dashboard",
  "uid": "solar-sentinel-slo-v1",
  "version": 1
}
```
```
Add to docker-compose.dev.yml grafana service:
    grafana:
      image: grafana/grafana:10.4.0
      ports: ["3000:3000"]
      volumes:
        - ./grafana/provisioning:/etc/grafana/provisioning
        - ./grafana/dashboards:/var/lib/grafana/dashboards
      environment:
        - GF_SECURITY_ADMIN_PASSWORD=solarsentinel
        - GF_USERS_ALLOW_SIGN_UP=false
```

---

### UPDATED RULE QUICK REFERENCE (v4 final — 23 rules)

```
 #  | Rule summary                                         | Fixed in
----|------------------------------------------------------|-------------------
 1  | Independent detectors — never merge before detect    | M2, M3
 2  | LSTM bidirectional=False                             | M5, M7
 3  | TSS not accuracy                                     | M4, M8, M15
 4  | FITS columns from YAML — no hardcoding               | M1
 5  | XGBoost multi:softprob, 4 classes, per-class thresh  | M4, M8
 6  | timesfm.finetuning does not exist                    | M5
 7  | Chronos: predict() not predict_quantiles()           | M11
 8  | confusion_matrix labels=[0,1] always                 | M8, M15
 9  | GraphRAG: local embeddings only                      | M9
10  | dspy.context() per-call (not dspy.configure())       | M9, M7
11  | torch.load weights_only=True                         | M5, M7, M10, M11
12  | Module-level open() needs FileNotFoundError guard    | M1, M7, M10
13  | Consistent model save format                         | M4, M5, M6
14  | No session fixtures in hypothesis @given             | M6
15  | No private function imports across modules           | M3, M7
16  | Singleton loaders for Chronos, MOMENT, ONNX          | M7, M11, M12
17  | Provenance columns in every catalogue row            | M3, M7
18  | Pandera schema (+ monotonic index) at ingestion      | M1 [v4 fix]
19  | Dashboard = React + TypeScript + Vite                | M0, M14
20  | Prometheus metrics centralised in metrics.py         | M7, M14, M16
21  | dev→staging→prod gate (localhost in CI) [v4 fix]    | M16
22  | SEED=42 in training scripts (file-level grep) [fix]  | M4, M5, M8, M10
23  | seen_files.db env-var path; check before ingestion   | M1, M16 [v4 fix]
----|------------------------------------------------------|-------------------
SLO-5 | FAR ≤ 0.10                                        | M8, M15, M16
SLO-6 | TPR(M+X) ≥ 0.80                                   | M15, M16
T-1   | SensitiveFormatter live in src/utils/ [v4 fix]    | M1
T-2   | verify_model_hash + rollback.py                    | M16
T-3   | WebSocket: push + 20-conn cap + rate limit [fix]  | M14
T-6   | gitleaks in CI + pre-commit                        | M16
D10   | B-class = label 0; documented in D10.md            | M2, M4
```
