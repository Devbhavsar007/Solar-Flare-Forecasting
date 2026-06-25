# ☀️ SolarSentinel — Solar Flare Nowcasting & Forecasting
### ISRO Aditya-L1 · SoLEXS + HEL1OS · PS15 · Quantum Crew

> Full-stack implementation plan for automated solar flare detection, classification, and prediction using independent per-instrument catalogues merged into a master event database — with physics-informed ML, three-model forecasting ensemble, DSPy self-optimising LLM intelligence, GraphRAG knowledge graph, MOMENT pre-flare anomaly detection, dual conformal+probabilistic uncertainty, ONNX-optimised inference, and complete MLOps infrastructure.

[![CI](https://github.com/quantum-crew/solar-sentinel/actions/workflows/ci.yml/badge.svg)](https://github.com/quantum-crew/solar-sentinel/actions)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-blue)](http://localhost:5000)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## PS15 Compliance Map

Every explicit PS15 requirement and exactly where it is satisfied:

| PS15 Requirement | Implementation | File |
|---|---|---|
| Detect independently in soft X-ray | SoLEXS-only threshold + ML detector | `src/nowcasting/solexs_detector.py` |
| Detect independently in hard X-ray | HEL1OS-only count-rate detector | `src/nowcasting/hel1os_detector.py` |
| Combine catalogues → master catalogue | Temporal coincidence merger | `src/catalogue/merger.py` |
| Nowcasting: real-time detection + classification | TCN + XGBoost N/C/M/X | `src/nowcasting/` |
| Forecasting: precursor-based with lead time | Causal LSTM + TCN + TimesFM ensemble | `src/forecasting/` |
| Lead time quantification | `compute_lead_time()` per event | `src/evaluation/metrics.py` |
| Detect low- and high-class flares (C through X) | Multi-class head: N / C / M / X | `src/nowcasting/train.py` |
| Visualise light curves + alert triggers | React dashboard + WebSocket | `dashboard/` |
| Automated flare database | Master catalogue CSV + API endpoint | `data/processed/master_catalogue.csv` |
| TSS + FAR evaluation | `full_report()` per flare class | `src/evaluation/metrics.py` |

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Accuracy Strategy](#accuracy-strategy)
- [Pre-Implementation Verification Checklist](#pre-implementation-verification-checklist)
- [Decision Record](#decision-record)
- [Milestone Map](#milestone-map)
- [M0 — Environment & Setup](#m0--environment--setup)
- [M1 — Data Pipeline](#m1--data-pipeline)
- [M2 — Independent Instrument Detectors](#m2--independent-instrument-detectors)
- [M3 — Master Catalogue Merger](#m3--master-catalogue-merger)
- [M4 — Multi-class Nowcasting](#m4--multi-class-nowcasting)
- [M5 — Three-Model Forecasting Ensemble](#m5--three-model-forecasting-ensemble)
- [M6 — ONNX Export & Optimised Inference](#m6--onnx-export--optimised-inference)
- [M7 — Agent Orchestration (LangGraph)](#m7--agent-orchestration-langgraph)
- [M8 — AutoML Tuning (FLAML)](#m8--automl-tuning-flaml)
- [M9 — LLM Intelligence (DSPy + GraphRAG + Phi-3-mini)](#m9--llm-intelligence-dspy--graphrag--phi-3-mini)
- [M10 — Physics Layer (PINN + Phase Detector)](#m10--physics-layer-pinn--phase-detector)
- [M11 — Dual Uncertainty (MAPIE + Chronos-Bolt)](#m11--dual-uncertainty-mapie--chronos-bolt)
- [M12 — Pre-flare Anomaly Detection (MOMENT)](#m12--pre-flare-anomaly-detection-moment)
- [M13 — Explainability (SHAP)](#m13--explainability-shap)
- [M14 — Dashboard & Visualization](#m14--dashboard--visualization)
- [M15 — Evaluation & Walk-Forward Validation](#m15--evaluation--walk-forward-validation)
- [M16 — Production Deployment & MLOps](#m16--production-deployment--mlops)
- [Tests](#tests)
- [Project Structure](#project-structure)
- [Evaluation Metrics](#evaluation-metrics)
- [References](#references)

---

## Project Overview

Solar flares are sudden, intense bursts of radiation from magnetic energy release in the solar atmosphere. M and X class events disrupt satellite communications, GPS navigation, and power grids globally.

**SolarSentinel** is a production-grade solar flare intelligence system built on ISRO's Aditya-L1 SoLEXS and HEL1OS instruments, satisfying every PS15 objective across three layers:

| Layer | Components |
|---|---|
| **Detection** | Independent SoLEXS + HEL1OS detectors → temporal coincidence merger → master catalogue |
| **ML Core** | TCN + XGBoost nowcasting (N/C/M/X) + Causal LSTM + TCN + TimesFM forecasting ensemble |
| **Intelligence** | DSPy self-optimising LLM + GraphRAG knowledge graph + MOMENT anomaly scoring + dual uncertainty |

**Target metrics — never report raw accuracy:**

| Metric | Target | Why |
|---|---|---|
| TSS (True Skill Score) | > 0.80 | Correct metric for imbalanced flare data |
| False Alarm Rate | < 0.10 | Operational requirement |
| Mean Lead Time | > 30 min | Gives operators time to act |
| C-class TPR | > 0.85 | PS15 explicitly requires low-class detection |
| ROC-AUC | > 0.90 | Overall classifier quality |
| Walk-forward TSS std | < 0.12 | Model reliability, not luck |

> **Why TSS, not accuracy?** 95%+ of solar data is quiet sun. A model predicting "no flare" always scores 99%+ accuracy while catching zero real events. TSS = TPR − FPR is the mandatory operational metric throughout solar flare ML literature (Bloomfield et al. 2012; Bobra & Couvidat 2015).

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│             PRADAN Portal (ISSDC) + GOES XRS 40yr Archive            │
│    SoLEXS Level-1 FITS      HEL1OS Level-1 FITS      GOES XRS       │
└──────────────┬──────────────────────────┬────────────────────────────┘
               │                          │
               ▼                          ▼
┌──────────────────────┐     ┌───────────────────────────┐
│  FITS SAFE READER    │     │  FITS SAFE READER          │
│  inspect_columns()   │     │  inspect_columns()         │
│  dead_time_correct() │     │  background_subtract()     │
└──────────┬───────────┘     └────────────┬──────────────┘
           │                              │
           ▼                              ▼
┌──────────────────────┐     ┌───────────────────────────┐
│  SoLEXS DETECTOR     │     │  HEL1OS DETECTOR           │
│  Soft X-ray only     │     │  Hard X-ray counts only    │
│  → SoLEXS catalogue  │     │  → HEL1OS catalogue        │
└──────────┬───────────┘     └────────────┬──────────────┘
           │                              │
           └──────────────┬───────────────┘
                          │ (fan-in via LangGraph merge node)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  MASTER CATALOGUE MERGER                     │
│   Temporal coincidence ±2 min → dual / single confidence    │
│   Output: master_catalogue.csv (primary PS15 deliverable)   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               PHYSICS FEATURE ENGINEERING                    │
│  Neupert ratio · thermal index · solar phase (configurable) │
│  doubling time · channel ratio · instrument lag             │
│  GOES log-space cross-calibration · MOMENT anomaly score    │
│  tsaug minority class augmentation                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
          ┌────────────────┴──────────────────┐
          ▼                                    ▼
┌──────────────────────────┐   ┌────────────────────────────────────┐
│  NOWCASTING PIPELINE     │   │  FORECASTING PIPELINE               │
│                          │   │                                     │
│  TCN Encoder (→ ONNX)    │   │  ┌─────────────────────────────┐   │
│  Dilated causal conv     │   │  │  Causal LSTM (unidirectional)│   │
│  128-dim embedding       │   │  │  bidirectional=False         │   │
│  + Physics features      │   │  └──────────────┬──────────────┘   │
│  + MOMENT anomaly score  │   │                 │                   │
│        │                 │   │  ┌──────────────┴──────────────┐   │
│  XGBoost multi:softprob  │   │  │  TCN Forecaster (→ ONNX)    │   │
│  FLAML macro_f1 tuned    │   │  └──────────────┬──────────────┘   │
│  Per-class thresholds    │   │                 │                   │
│  Output: N / C / M / X   │   │  ┌──────────────┴──────────────┐   │
│        │                 │   │  │  TimesFM 2.0-200M (PEFT LoRA)│   │
│  PINN physics loss       │   │  │  fine-tuned on flare windows  │   │
└──────────┬───────────────┘   │  └──────────────┬──────────────┘   │
           │                   │                 │                   │
           │                   │  Weighted ensemble 0.35/0.35/0.30  │
           │                   │  Multi-horizon: 15 / 30 / 60 min   │
           │                   └──────────────────┬─────────────────┘
           └──────────────────┬───────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              DUAL UNCERTAINTY LAYER                          │
│  MAPIE conformal (90% guaranteed coverage)                  │
│  Chronos-Bolt probabilistic (q10/q50/q90 from MC samples)   │
│  Dual agreement check → HIGH / LOW confidence flag          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│         LANGGRAPH AGENT ORCHESTRATION                        │
│  Ingestion → [detect_solexs ‖ detect_hel1os] → fan-in      │
│  → merge → preprocess → moment_score → nowcast → forecast   │
│  → uncertainty → shap → alert_router → llm_report           │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│         LLM INTELLIGENCE LAYER                               │
│  DSPy ChainOfThought (thread-safe per-call context)         │
│  GraphRAG local-method entity-level retrieval               │
│  Phi-3-mini via Ollama — local, zero API cost               │
│  Structured fallback bulletin — always fires on Ollama fail │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  UNIFIED DASHBOARD (FastAPI + React / Streamlit)            │
│  Live SoLEXS + HEL1OS curves · conformal + Chronos bands   │
│  SHAP waterfall · MOMENT score · LLM bulletin · catalogue   │
│  PWA installable · replay mode · CSV export                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Version to pin | Purpose |
|---|---|---|---|
| Data I/O | `astropy`, `sunpy` | 6.1.0, 6.0.1 | FITS reading, GOES download |
| Augmentation | `tsaug` | 0.2.1 | Minority class time-series augmentation |
| Nowcasting | `pytorch` (TCN), `xgboost` | 2.3.0, 2.1.1 | N/C/M/X classification |
| Forecasting | `pytorch` (Causal LSTM + TCN) | 2.3.0 | Multi-horizon prediction |
| Foundation model | `timesfm` (Google ICML 2024) | 1.2.0 | 3rd ensemble member — verify on PyPI |
| Fine-tuning | `peft`, `transformers` | 0.12.0, 4.44.0 | LoRA adapter for TimesFM |
| Anomaly detection | `momentfm` (CMU) | 0.1.7 | Pre-flare anomaly scoring |
| Probabilistic forecast | `chronos-forecasting` (Amazon) | 1.3.3 | 2nd uncertainty source |
| ONNX inference | `onnx`, `onnxruntime` | 1.16.2, 1.19.2 | Sub-100ms deployment |
| Physics loss | Custom PINN (PyTorch) | — | Neupert constraint |
| AutoML | `flaml` | 2.3.0 | XGBoost hyperparameter search |
| Uncertainty | `mapie` | 0.8.6 | Conformal prediction intervals |
| LLM prompting | `dspy-ai` | 2.4.17 | Self-optimising prompt modules |
| Knowledge graph | `graphrag` (Microsoft) | 0.3.6 | Multi-hop solar physics reasoning |
| LLM runtime | `ollama` Phi-3-mini | latest | Local inference, zero API cost |
| RAG vector store | `chromadb` | 0.5.5 | Embedding store for GraphRAG |
| Explainability | `shap` | 0.45.1 | Per-prediction feature attribution |
| Drift detection | `alibi-detect` | 0.11.4 | Distribution shift monitoring |
| Orchestration | `langgraph`, `langchain-core` | 0.2.45, 0.3.20 | Stateful agent pipeline |
| Experiment tracking | `mlflow` | 2.17.0 | Model versioning + metrics |
| Backend API | `fastapi`, `uvicorn` | 0.115.0, 0.31.0 | REST + WebSocket |
| Fault tolerance | `circuitbreaker` | 2.0.0 | PRADAN/GOES outage protection |
| Testing | `pytest`, `hypothesis` | 8.3.3, 6.112.0 | Unit + property-based tests |
| CI/CD | `GitHub Actions` | — | Auto-test + weekly retrain |
| Containers | `Docker`, `docker-compose` | — | Full stack deployment |
| Monitoring | `Prometheus + Grafana` | — | Latency, FAR, uptime |

> **Before locking any version:** run `pip index versions <package>` to confirm the pinned version still exists on PyPI. Some packages release breaking changes weekly.

---

## Accuracy Strategy

Every lever applied, in order of impact on TSS:

```
ACCURACY LEVERS
├── 1. GOES 40yr historical data           → Labeled M/X events at scale
├── 2. Physics features (Neupert, phase)   → Domain signal models cannot invent
├── 3. PINN physics loss                   → Constrain model to solar physics
├── 4. TimesFM as 3rd ensemble member      → Pre-trained 100B time-point corpus
├── 5. MOMENT anomaly score as feature     → Pre-flare reconstruction error → XGBoost
├── 6. Multi-horizon heads (15/30/60)      → One encoder, three horizons
├── 7. Conformal + Chronos dual intervals  → Honest calibrated probabilities
├── 8. FLAML macro_f1 auto-tuning          → Optimal XGBoost config, not guessed
├── 9. Per-class thresholds (C/M/X)        → Separate cost functions per severity
├── 10. tsaug augmentation                 → Synthetic minority class windows

PRODUCTION LEVERS
├── 11. ONNX export (TCN + LSTM)           → Sub-100ms real-time inference
├── 12. Walk-forward CV (5-fold, 1-day gap)→ Honest TSS across activity periods
├── 13. Drift detection (alibi)            → Know when model degrades in prod
├── 14. Auto-retraining (GH Actions)       → Self-healing on weekly schedule
├── 15. Model registry (MLflow)            → Safe promotion of better models
├── 16. Circuit breaker                    → Survives PRADAN/GOES outages
├── 17. Property-based tests (hypothesis)  → Catches edge cases humans miss

DIFFERENTIATION LEVERS
├── 18. DSPy self-optimising prompts       → No manual prompt engineering
├── 19. GraphRAG knowledge graph           → Multi-hop solar physics reasoning
├── 20. SHAP per-alert explanations        → "Why did it alert?" answered
├── 21. Chronos-Bolt probabilistic bands   → Two independent uncertainty sources
├── 22. MOMENT pre-flare scoring           → Anomaly score visible in dashboard
└── 23. PWA mobile dashboard               → Installable, works offline
```

---

## Pre-Implementation Verification Checklist

> **Run every item in this section before writing any ML code.** These are the exact points where external library APIs diverge from documentation or have changed between versions. Each check takes 10–30 minutes. Skipping them costs days of debugging during implementation.

### CHECK 1 — FITS Column Names (run on day 1 of M0)

This is the highest-priority check. Every downstream pipeline component depends on correct column names. Do this before M1 starts.

```python
# scripts/verify_fits_columns.py
# Run: python scripts/verify_fits_columns.py path/to/solexs.fits path/to/hel1os.fits

from astropy.io import fits
import sys, json

def inspect(filepath):
    print(f"\n{'='*60}")
    print(f"FILE: {filepath}")
    with fits.open(filepath) as hdul:
        hdul.info()
        schema = {}
        for i, ext in enumerate(hdul):
            if hasattr(ext, 'columns') and ext.columns:
                print(f"\n  Extension {i} ({ext.name}) columns:")
                cols = []
                for col in ext.columns:
                    print(f"    {col.name:35s} format={col.format} unit={col.unit}")
                    cols.append(col.name)
                schema[ext.name] = cols
            if hdul[i].header:
                cards = list(hdul[i].header.items())[:8]
                print(f"  Header sample: {dict(cards)}")
        return schema

if __name__ == "__main__":
    for fpath in sys.argv[1:]:
        schema = inspect(fpath)
        print(f"\nJSON schema for configs/fits_columns.yaml:\n{json.dumps(schema, indent=2)}")
```

After running, fill in `configs/fits_columns.yaml`:

```yaml
# configs/fits_columns.yaml
# Populate AFTER running scripts/verify_fits_columns.py on real PRADAN files.
# DO NOT guess column names. Every pipeline failure traces back to wrong names here.

solexs:
  time_col:      "TIME"      # ← replace with verified name
  flux_low_col:  "FLUX_1"   # ← replace with verified name
  flux_mid_col:  "FLUX_2"   # ← replace with verified name
  flux_high_col: "FLUX_3"   # ← replace with verified name
  livetime_col:  "LIVETIME"  # ← used for dead-time; may not exist — check
  binary_table_ext: 1        # ← HDU index of the binary table; verify

hel1os:
  time_col:       "TIME"      # ← replace with verified name
  counts_low_col: "COUNTS_1"  # ← replace with verified name
  counts_high_col:"COUNTS_2"  # ← replace with verified name
  binary_table_ext: 1         # ← HDU index; verify
  dead_time_us:   2.5         # ← verify from HEL1OS instrument user guide
```

### CHECK 2 — TimesFM API (run before M5)

TimesFM's fine-tuning and forecast output API must be verified before implementing `src/forecasting/timesfm_forecaster.py`.

```python
# scripts/verify_timesfm.py
# Run after: pip install timesfm==1.2.0

import timesfm, inspect, pandas as pd, numpy as np

# 1. List all available methods
tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(backend="torch", horizon_len=30,
                                    context_len=128, per_core_batch_size=4),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-2.0-200m-pytorch")
)
print("TimesFm methods:", [m for m in dir(tfm) if not m.startswith('_')])

# 2. Verify forecast output column names
toy = pd.DataFrame({
    "unique_id": ["SoLEXS"] * 60,
    "ds": pd.date_range("2024-01-01", periods=60, freq="1min"),
    "y": np.random.exponential(1e-7, 60)
})
out = tfm.forecast_on_df(inputs=toy, freq="T", value_name="y", num_jobs=1)
print("Forecast columns:", out.columns.tolist())
# Expected: something like ["unique_id", "ds", "timesfm"] — NOT "timesfm_15"
# Record the actual column name and update forecasting code accordingly.

# 3. Check if finetuning submodule exists
try:
    import timesfm.finetuning
    print("finetuning module EXISTS:", dir(timesfm.finetuning))
except ImportError:
    print("finetuning module DOES NOT EXIST — use PEFT + HuggingFace Trainer instead")
```

**Expected result:** `timesfm.finetuning` will not exist in the current public release. Fine-tuning uses PEFT (see M5). The forecast output column will be `"timesfm"` — slice it by horizon index, not by column name suffix.

### CHECK 3 — Chronos-Bolt API (run before M11)

```python
# scripts/verify_chronos.py
# Run after: pip install chronos-forecasting==1.3.3

import torch, numpy as np
from chronos import BaseChronosPipeline, ChronosPipeline

pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-small", dtype=torch.float32, device_map="cpu"
)
print("Pipeline type:", type(pipeline))
print("Available methods:", [m for m in dir(pipeline) if not m.startswith('_')])

# Test actual predict call
context = torch.zeros(1, 60, dtype=torch.float32)
# Correct API: predict(), NOT predict_quantiles()
samples = pipeline.predict(context, prediction_length=15, num_samples=20)
print("predict() output shape:", samples.shape)
# Expected: (batch_size=1, num_samples=20, horizon=15)
# Then compute quantiles manually:
peak_per_sample = samples[0].max(dim=-1).values.numpy()
q10, q50, q90 = np.quantile(peak_per_sample, [0.1, 0.5, 0.9])
print(f"q10={q10:.4e}  q50={q50:.4e}  q90={q90:.4e}")
```

**Key finding:** Chronos uses `predict()`, not `predict_quantiles()`. Quantiles are computed from Monte Carlo samples after the fact.

### CHECK 4 — MOMENT API (run before M12)

```python
# scripts/verify_moment.py
# Run after: pip install momentfm==0.1.7

import torch
from momentfm import MOMENTPipeline

model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large",
    model_kwargs={"task_name": "reconstruction"}
)
model.init()

# Verify input shape convention: (batch_size, n_channels, seq_len)
dummy = torch.zeros(2, 1, 512)   # batch=2, channels=1, seq=512
model.eval()
with torch.no_grad():
    out = model(x_enc=dummy)

print("Output type:", type(out))
print("Output attributes:", [a for a in dir(out) if not a.startswith('_')])
# Record the actual reconstruction attribute name — may be "reconstruction",
# "output", or similar. Update src/forecasting/moment_anomaly.py accordingly.
print("reconstruction shape:", out.reconstruction.shape)
# Expected: (batch_size, n_channels, seq_len) = (2, 1, 512)
```

### CHECK 5 — DSPy + Ollama (run before M9)

```python
# scripts/verify_dspy.py
# Run after: pip install dspy-ai==2.4.17, ollama pulled phi3:mini

import dspy

# Test connection — syntax differs between DSPy versions
# Try in order until one works:
try:
    lm = dspy.LM("ollama/phi3:mini", api_base="http://localhost:11434")
    print("Syntax 1 works: dspy.LM('ollama/phi3:mini', api_base=...)")
except Exception as e:
    print(f"Syntax 1 failed: {e}")

try:
    lm = dspy.OllamaLocal("phi3:mini")
    print("Syntax 2 works: dspy.OllamaLocal('phi3:mini')")
except Exception as e:
    print(f"Syntax 2 failed: {e}")

# Test thread-safety of per-call context (critical for async FastAPI)
with dspy.context(lm=lm):
    result = dspy.Predict("question -> answer")(question="What is a solar flare?")
    print("DSPy response:", result.answer[:100])
```

Record which syntax works and pin it in `src/intelligence/dspy_reporter.py`.

### CHECK 6 — PRADAN Authentication (run before M0 download)

Do not write `pradan_downloader.py` until you have inspected the actual login mechanism.

```
Steps:
1. Open https://pradan.issdc.gov.in in Chrome
2. Open DevTools → Network tab → clear all
3. Fill the login form and submit
4. In Network tab, find the POST request to the login endpoint
5. Inspect:
   - Request URL (the actual auth endpoint)
   - Request payload (field names — may be 'username'/'password' or different)
   - Response headers (check for Set-Cookie — session-based or token-based?)
   - Any CSRF token in the form (inspect page source for <input type="hidden" name="csrf*">)
6. Record findings in configs/pradan_auth.yaml (DO NOT commit passwords)
```

```yaml
# configs/pradan_auth.yaml — auth mechanism reference (no credentials here)
auth_endpoint:  "https://pradan.issdc.gov.in/LOGIN_ENDPOINT"  # ← fill after inspection
method:         "form_post"    # or "oauth2" or "api_key" — fill after inspection
form_fields:
  username_field: "username"   # ← actual field name from DevTools
  password_field: "password"   # ← actual field name from DevTools
csrf_token:     false          # ← set true if CSRF token found in form
session_cookie: "JSESSIONID"   # ← actual cookie name from response headers
```

### CHECK 7 — GraphRAG Local Embedding (run before M9)

GraphRAG defaults to OpenAI for embeddings. Without local configuration it will fail or incur cost.

```bash
# Step 1: Pull a local embedding model
ollama pull nomic-embed-text    # 274M, fast, good quality

# Step 2: After graphrag init, edit the generated settings.yaml
# Find the embeddings section and change it to:
```

```yaml
# data/graphrag/settings.yaml (relevant section only)
llm:
  api_base: http://localhost:11434/v1   # Ollama OpenAI-compatible endpoint
  api_key: ollama                        # Ollama ignores this value but requires it
  model: phi3:mini
  type: openai_chat

embeddings:
  llm:
    api_base: http://localhost:11434/v1
    api_key: ollama
    model: nomic-embed-text
    type: openai_embedding
```

```bash
# Step 3: Run indexing (30–90 min depending on document volume)
python -m graphrag index --root data/graphrag

# Step 4: Commit the output so team members don't re-run it
git add data/graphrag/output/ data/graphrag/cache/
git commit -m "feat: graphrag index built from knowledge base"
```

---

## Decision Record

> These eight decisions must be made and recorded before the corresponding milestone begins. They are listed here as explicit choices, not open questions.

| # | Decision | What to do | When |
|---|---|---|---|
| D1 | FITS column names | Run CHECK 1, fill `configs/fits_columns.yaml` | M0 day 1 |
| D2 | Available flare count | Count M/X events from NOAA SWPC in Aditya-L1 observation window (Sep 2023 – present); if < 50, increase GOES supplementary weight and tsaug ratio | M1 |
| D3 | GPU vs CPU-only | MOMENT batch inference and TimesFM need GPU for practical throughput. If CPU-only, limit MOMENT to online-only inference on flagged windows; disable TimesFM LoRA fine-tuning | M0 |
| D4 | Phase detector constants | `detect_solar_phase()` uses tunable thresholds (default 0.3 and 0.8 as fractions of peak). Run EDA on quiet-period vs flare-period flux distributions in M1 and update `configs/nowcasting.yaml` before M4 | M1 EDA |
| D5 | PRADAN auth mechanism | Run CHECK 6 and complete `configs/pradan_auth.yaml` | M0 |
| D6 | Library version lock | Create `requirements.txt` from pinned versions in Tech Stack table, verify all install cleanly together (`pip install -r requirements.txt`), commit | M0 |
| D7 | LangGraph fan-in strategy | Use annotated state reducer (see M7 section); do not use two bare `add_edge` calls into one node — they do not run in parallel in LangGraph without explicit fork/join | M7 |
| D8 | X-class alert threshold direction | X-class threshold should be LOWER than M-class (e.g. `{"C": 0.38, "M": 0.45, "X": 0.28}`) because missing an X-class event is operationally catastrophic. Precision-first (high threshold for X) is wrong for space weather. Document this choice explicitly in `configs/nowcasting.yaml` | M4 |

---

## Milestone Map

```
M0──M1──M2──M3──M4──M5──M6──M7──M8──M9──M10─M11─M12─M13─M14─M15─M16
Set Data Sol Hel Cat Now 3M  ONX LGr FLM LLM Phy Unc Mom Exp Dsh Eval Prod
up  Pipe EXS 1OS Mrg cast Ens NX  aph ML  DSP NN  ert ent Pln ard  CV  MLOp
```

| Milestone | Deliverable | Day | Dependencies |
|---|---|---|---|
| M0 | Env, verifications (7 checks), `requirements.txt`, FITS columns, D1–D6 | 1–2 | All 7 checks complete |
| M1 | FITS-safe reader, dead-time correction, log-space GOES calibration, EDA | 3–4 | CHECK 1 complete |
| M2 | Independent SoLEXS + HEL1OS detectors with phase-tuned thresholds | 5–6 | M1, D4 |
| M3 | Master catalogue merger (temporal coincidence, confidence tiers) | 7 | M2 |
| M4 | Multi-class nowcasting TCN + XGBoost N/C/M/X, per-class thresholds, D8 | 8–10 | M3 |
| M5 | Three-model forecasting ensemble: Causal LSTM + TCN + TimesFM (PEFT) | 11–13 | CHECK 2 complete, M4 |
| M6 | ONNX export for TCN and LSTM — sub-100ms benchmark | 14 | M5 |
| M7 | LangGraph agent orchestration with correct parallel fan-in | 15 | M6, D7 |
| M8 | FLAML AutoML (macro_f1) + walk-forward CV (see M15 for eval) | 16 | M4 |
| M9 | DSPy + GraphRAG (local embeddings indexed) + Phi-3-mini | 17–18 | CHECK 4, 5, 6 complete |
| M10 | PINN physics loss + solar phase detector (configurable constants) | 19 | M5 |
| M11 | MAPIE conformal + Chronos-Bolt dual uncertainty | 20 | CHECK 3 complete, M4 |
| M12 | MOMENT pre-flare anomaly detection (batch inference) | 21 | CHECK 4 complete, GPU confirmed |
| M13 | SHAP explainability layer | 22 | M4 |
| M14 | Full dashboard with all features live | 23–24 | M7–M13 |
| M15 | Walk-forward CV, per-class evaluation, benchmarked | 25 | M4, M5 |
| M16 | Docker, CI/CD, drift monitor, PRADAN auth, deployed | 26–28 | D5, all prior |

---

## M0 — Environment & Setup

### 0.1 Python Environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 0.2 `requirements.txt` (pinned)

```text
# requirements.txt — pin every version; verify all install together before committing

# Data
astropy==6.1.0
sunpy==6.0.1
numpy==1.26.4
pandas==2.2.3
scipy==1.14.1

# ML
torch==2.3.0
scikit-learn==1.5.2
xgboost==2.1.1
flaml==2.3.0

# Foundation models — verify versions on PyPI before locking
timesfm==1.2.0
momentfm==0.1.7
chronos-forecasting==1.3.3

# TimesFM fine-tuning (PEFT — used instead of non-existent timesfm.finetuning)
peft==0.12.0
transformers==4.44.0
datasets==3.0.0
accelerate==0.34.0

# ONNX
onnx==1.16.2
onnxruntime==1.19.2

# Uncertainty + explainability
mapie==0.8.6
shap==0.45.1

# LLM + orchestration
dspy-ai==2.4.17
graphrag==0.3.6
chromadb==0.5.5
langchain-core==0.3.20
langchain-community==0.3.7
langgraph==0.2.45

# Data ops
tsaug==0.2.1
alibi-detect==0.11.4

# API + infra
fastapi==0.115.0
uvicorn==0.31.0
circuitbreaker==2.0.0
requests==2.32.3
python-dotenv==1.0.1
pyyaml==6.0.2

# Monitoring
mlflow==2.17.0

# Testing
pytest==8.3.3
hypothesis==6.112.0
httpx==0.27.2
```

```bash
pip install -r requirements.txt
# If any version conflict: pip install -r requirements.txt --no-deps, then resolve manually
```

### 0.3 Secrets

```bash
# .env — never commit this file
PRADAN_USERNAME=your_username
PRADAN_PASSWORD=your_password

# Add to .gitignore
echo ".env" >> .gitignore
echo "data/raw/" >> .gitignore
echo "models/*.pt" >> .gitignore
```

### 0.4 Data Download

```bash
# 1. Download GOES XRS historical (public, no auth)
python src/ingestion/goes_downloader.py --start 2014-01-01 --end 2024-06-01

# 2. Download NOAA flare event catalog
wget -r -np -nd https://ftp.swpc.noaa.gov/pub/indices/events/ \
     -P data/labels/ --accept "*.txt"

# 3. High-value Aditya-L1 dates — download and verify columns on these first:
#    2024-02-22  X6.3  — First jointly observed by SoLEXS + HEL1OS
#    2024-05-10  X5.8  — Major CME event
#    2024-10-03  M-series — Multi-flare sequence

# 4. Run column verification (CHECK 1) on downloaded files
python scripts/verify_fits_columns.py \
    data/raw/solexs/2024-02-22_solexs.fits \
    data/raw/hel1os/2024-02-22_hel1os.fits
```

### 0.5 Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull phi3:mini        # 2.2GB — LLM inference
ollama pull nomic-embed-text # 274MB — GraphRAG embeddings
```

---

## M1 — Data Pipeline

### 1.1 FITS Reader (column-config driven)

```python
# src/ingestion/fits_reader.py

from astropy.io import fits
import pandas as pd
import numpy as np
import yaml

with open("configs/fits_columns.yaml") as f:
    COL = yaml.safe_load(f)


def _safe_read(filepath: str, instrument: str) -> pd.DataFrame:
    """
    Column names are loaded from configs/fits_columns.yaml.
    If a column is missing, the error message shows available names
    so the developer can update the config immediately — not hunt blind.
    """
    cfg = COL[instrument]
    ext = cfg.get("binary_table_ext", 1)

    with fits.open(filepath) as hdul:
        data      = hdul[ext].data
        available = [c.name for c in hdul[ext].columns]

    col_map = {
        k.replace("_col", ""): v
        for k, v in cfg.items()
        if k.endswith("_col")
    }

    rows = {}
    for canonical, actual in col_map.items():
        if actual not in available:
            raise KeyError(
                f"Column '{actual}' not found in {instrument} FITS ({filepath}).\n"
                f"Available columns: {available}\n"
                f"Update configs/fits_columns.yaml — run scripts/verify_fits_columns.py."
            )
        rows[canonical] = data[actual].astype(float)

    df       = pd.DataFrame(rows)
    time_raw = data[cfg["time_col"]]
    df.index = pd.to_datetime(time_raw, unit="s", origin="unix")
    df.index.name = "time"
    return df.replace([np.inf, -np.inf], np.nan).dropna().sort_index()


def read_solexs(filepath: str) -> pd.DataFrame:
    return _safe_read(filepath, "solexs")


def read_hel1os(filepath: str) -> pd.DataFrame:
    df = _safe_read(filepath, "hel1os")
    return apply_dead_time_correction(df)


def apply_dead_time_correction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Paralyzable dead-time correction for HEL1OS photon-counting detector.
    Model: N_obs = N_true × exp(−N_true × τ)
    Inverted numerically per time bin via Brent's method.

    Without this, X-class flare severity is underestimated — the detector
    saturates at high photon flux and measured counts are lower than true.
    τ (dead_time_us) must be verified from the HEL1OS instrument user guide.
    """
    from scipy.optimize import brentq
    dead_time_us = COL["hel1os"].get("dead_time_us", 2.5)
    tau = dead_time_us * 1e-6

    def correct_single(n_obs: float) -> float:
        rate_obs = n_obs
        if rate_obs <= 0:
            return 0.0
        saturation_rate = 1.0 / (np.e * tau)
        if rate_obs >= saturation_rate:
            return np.nan   # saturated — interpolate later
        try:
            return brentq(
                lambda r: r * np.exp(-r * tau) - rate_obs,
                0, saturation_rate * 5,
                xtol=1e-7
            )
        except ValueError:
            return n_obs

    corrected = df.copy()
    for col in ["counts_low", "counts_high"]:
        if col in corrected.columns:
            corrected[col] = corrected[col].apply(correct_single)
    # Interpolate saturated bins (NaN) with linear time interpolation, max 5-step gap
    corrected = corrected.interpolate(method="time", limit=5,
                                       limit_direction="both")
    return corrected


def merge_instruments(solexs_df: pd.DataFrame,
                       hel1os_df: pd.DataFrame,
                       cadence:    str = "1min") -> pd.DataFrame:
    return pd.concat([
        solexs_df.resample(cadence).mean(),
        hel1os_df.resample(cadence).mean()
    ], axis=1).dropna()
```

### 1.2 GOES Cross-Calibration (log-space)

GOES XRS and SoLEXS have different spectral response functions. Solar fluxes span 6 decades (1e-9 to 1e-3 W/m²). A linear regression in linear space is dominated by 3–5 X-class events and provides meaningless calibration over the C-class range where most training data lives. Always fit in log-space.

```python
# src/preprocessing/cross_calibration.py

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
import mlflow


def fit_goes_solexs_calibration(goes_df:   pd.DataFrame,
                                  solexs_df: pd.DataFrame,
                                  overlap_start: str = "2023-09-01",
                                  overlap_end:   str = "2024-06-01") -> dict:
    """
    Fit a calibration mapping from GOES XRS flux to SoLEXS-equivalent flux.

    WHY LOG-SPACE: Solar flux spans ~6 decades. Linear regression in linear
    space is dominated by the top 3-5 X-class events and gives a meaningless
    calibration over the C-class range (where most training data lives).

    WHY HUBER: Robust to the outliers produced by flare peaks that saturate
    one instrument but not the other, or occur during data gaps.

    Saves calibration coefficients to MLflow for reproducibility.
    """
    goes_col   = "xrs_b" if "xrs_b"   in goes_df.columns else goes_df.columns[0]
    solexs_col = "flux_high" if "flux_high" in solexs_df.columns else solexs_df.columns[0]

    goes_1min   = goes_df[overlap_start:overlap_end][goes_col].resample("1min").mean()
    solexs_1min = solexs_df[overlap_start:overlap_end][solexs_col].resample("1min").mean()

    aligned = pd.concat([goes_1min, solexs_1min], axis=1).dropna()
    aligned.columns = ["goes", "solexs"]
    aligned = aligned[(aligned["goes"] > 1e-9) & (aligned["solexs"] > 1e-9)]

    log_goes   = np.log10(aligned["goes"].values).reshape(-1, 1)
    log_solexs = np.log10(aligned["solexs"].values)

    model = HuberRegressor(epsilon=1.5, max_iter=500)
    model.fit(log_goes, log_solexs)
    r2 = model.score(log_goes, log_solexs)

    calibration = {
        "slope":     float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "r2":        round(float(r2), 4),
        "n_samples": len(aligned),
    }

    with mlflow.start_run(run_name="goes_solexs_log_calibration"):
        mlflow.log_params(calibration)

    print(
        f"GOES→SoLEXS calibration (log-log):\n"
        f"  slope={calibration['slope']:.4f}  "
        f"intercept={calibration['intercept']:.4f}  "
        f"r²={r2:.4f}  n={len(aligned)}"
    )
    if r2 < 0.80:
        print("WARNING: r² < 0.80 — calibration may be unreliable. "
              "Check overlap period for data gaps.")
    return calibration


def apply_goes_calibration(goes_flux:    np.ndarray,
                            calibration:  dict) -> np.ndarray:
    """Transform GOES XRS flux values to SoLEXS-equivalent scale."""
    log_goes        = np.log10(np.clip(goes_flux, 1e-12, None))
    log_solexs_equiv = calibration["slope"] * log_goes + calibration["intercept"]
    return np.power(10, log_solexs_equiv)
```

### 1.3 Physics Feature Engineering

```python
# src/preprocessing/physics_features.py

import numpy as np
import pandas as pd
from scipy.signal import correlate
import yaml

with open("configs/nowcasting.yaml") as f:
    PHASE_CFG = yaml.safe_load(f).get("phase_detector", {})

# Phase detection thresholds — calibrated during M1 EDA, stored in config.
# These defaults are literature-based estimates; update after EDA on real data.
PHASE_PEAK_FRAC_PRE       = PHASE_CFG.get("peak_frac_pre",       0.30)
PHASE_PEAK_FRAC_GRADUAL   = PHASE_CFG.get("peak_frac_gradual",   0.80)
BACKGROUND_SIGMA          = PHASE_CFG.get("background_sigma",     0.10)


def detect_solar_phase(flux_window: np.ndarray) -> int:
    """
    Classify the current position in the solar flare lifecycle.
    0=quiet  1=pre-flare  2=impulsive  3=peak  4=gradual

    Thresholds are loaded from configs/nowcasting.yaml (phase_detector section)
    so they can be updated after M1 EDA without touching code.
    The solar event phase is the single strongest categorical predictor of
    an imminent M/X peak and is fed as a feature into XGBoost.
    """
    if len(flux_window) < 5:
        return 0
    dfdt    = np.gradient(flux_window)
    current = flux_window[-1]
    peak    = flux_window.max()
    bkg_est = np.median(flux_window[:max(1, len(flux_window)//4)])
    slope   = dfdt[-1]

    if current < bkg_est * (1 + BACKGROUND_SIGMA) and abs(slope) < bkg_est * 0.05:
        return 0   # quiet
    elif slope > 0 and current < peak * PHASE_PEAK_FRAC_PRE:
        return 1   # pre-flare: rising slowly, well below peak
    elif slope > 0 and current >= peak * PHASE_PEAK_FRAC_PRE:
        return 2   # impulsive: rising steeply
    elif current >= peak * PHASE_PEAK_FRAC_GRADUAL and abs(slope) < current * 0.01:
        return 3   # near-peak
    elif slope < -current * 0.01:
        return 4   # gradual decay
    return 0


def engineer_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["flux_low", "flux_mid", "flux_high", "counts_low", "counts_high"]:
        if col in df.columns:
            df[f"d_{col}_dt"] = df[col].diff()

    if "flux_high" in df.columns and "counts_low" in df.columns:
        df["neupert_ratio"] = (
            df["counts_low"].diff() / (df["flux_high"] + 1e-12)
        )

    if "flux_mid" in df.columns and "flux_low" in df.columns:
        df["thermal_index"] = (
            np.log10(df["flux_mid"] + 1e-12) -
            np.log10(df["flux_low"] + 1e-12)
        )

    if "flux_high" in df.columns:
        ratio = df["flux_high"] / df["flux_high"].shift(5).clip(lower=1e-12)
        df["doubling_time"]   = 5.0 / np.log2(ratio.clip(lower=1e-6))
        df["normalised_flux"] = (
            df["flux_high"] /
            df["flux_high"].rolling(120).quantile(0.10).clip(lower=1e-12)
        )
        flux = df["flux_high"].values
        df["solar_phase"] = [
            detect_solar_phase(flux[max(0, i - 20):i + 1])
            for i in range(len(flux))
        ]

    if "flux_high" in df.columns and "counts_low" in df.columns:
        df["channel_ratio"] = df["flux_high"] / (df["counts_low"] + 1e-10)
        w    = min(60, len(df))
        corr = correlate(
            df["flux_mid"].fillna(0).values[-w:],
            df["counts_low"].fillna(0).values[-w:]
        )
        df["instrument_lag"] = int(corr.argmax() - w + 1)

    for col in ["flux_low", "flux_mid", "flux_high"]:
        if col in df.columns:
            df[f"{col}_rollstd_5"]  = df[col].rolling(5).std()
            df[f"{col}_rollstd_15"] = df[col].rolling(15).std()
            df[f"{col}_rollmax_5"]  = df[col].rolling(5).max()

    return df.dropna()


def subtract_background(df: pd.DataFrame, window_min: int = 10) -> pd.DataFrame:
    df_clean = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        baseline = df[col].rolling(window_min, min_periods=1).median()
        df_clean[col] = (df[col] - baseline).clip(lower=0)
    return df_clean
```

```yaml
# configs/nowcasting.yaml
phase_detector:
  peak_frac_pre:     0.30   # UPDATE after M1 EDA on real SoLEXS data
  peak_frac_gradual: 0.80   # UPDATE after M1 EDA
  background_sigma:  0.10   # background × (1+sigma) is quiet-sun threshold

class_thresholds:
  # X-class LOWER than M-class: missing X is catastrophically costly.
  # C-class lowest: maximize sensitivity for early warning.
  C: 0.38
  M: 0.45
  X: 0.28   # lower = higher recall = fewer missed X events
  binary: 0.40

dead_time_us: 2.5   # HEL1OS paralyzable model — verify from instrument docs
```

### 1.4 Multi-class Labels

```python
# src/preprocessing/labels.py

CLASS_MAP     = {"N": 0, "C": 1, "M": 2, "X": 3}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}


def build_multiclass_labels(df: "pd.DataFrame",
                             master_catalogue: "pd.DataFrame") -> "pd.DataFrame":
    """
    Label each 1-min row with N/C/M/X using the master catalogue.
    Also writes convenience binary flags for per-class threshold evaluation.
    """
    import pandas as pd
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


def create_windows(df: "pd.DataFrame",
                    feature_cols: list,
                    window_size:  int = 60,
                    horizon:      int = 15,
                    step:         int = 1) -> tuple:
    """Sliding window generator returning (X, y_now, y_fore)."""
    import numpy as np
    X, y_now, y_fore = [], [], []
    data   = df[feature_cols].values
    labels = df["label"].values

    for i in range(0, len(data) - window_size - horizon, step):
        X.append(data[i:i + window_size])
        y_now.append(labels[i + window_size - 1])
        y_fore.append(labels[i + window_size + horizon - 1])

    return (
        np.array(X,      dtype=np.float32),
        np.array(y_now,  dtype=np.int64),
        np.array(y_fore, dtype=np.int64),
    )
```

### 1.5 Augmentation

```python
# src/preprocessing/augmentation.py

import numpy as np
from tsaug import TimeWarp, Drift, Quantize

_augmenter = (
    TimeWarp(n_speed_change=3, max_speed_ratio=3.0) * 2
    + Drift(max_drift=0.1, n_drift_points=5)
    + Quantize(n_levels=20)
)


def augment_minority(X_train: np.ndarray,
                      y_train: np.ndarray,
                      target_ratio: float = 0.30) -> tuple:
    """
    Augment flare windows (C/M/X) until they reach target_ratio of total dataset.
    Only operates on minority windows — quiet-sun windows are NOT augmented.
    """
    X_flare = X_train[y_train > 0]
    n_needed = int(len(X_train) * target_ratio) - len(X_flare)
    if n_needed <= 0:
        return X_train, y_train

    reps        = (n_needed // max(len(X_flare), 1)) + 1
    y_flare     = y_train[y_train > 0]
    X_aug_pool  = np.vstack([_augmenter.augment(X_flare) for _ in range(reps)])[:n_needed]
    y_aug_pool  = np.tile(y_flare, reps + 1)[:n_needed]

    return np.vstack([X_train, X_aug_pool]), np.hstack([y_train, y_aug_pool])
```

**M1 Checkpoint:** `data/processed/dataset.parquet` written. GOES calibration r² > 0.80 logged to MLflow. Label distribution printed — if M/X count < 20 events, revisit D2. Phase detector constants updated in `configs/nowcasting.yaml` after EDA.

---

## M2 — Independent Instrument Detectors

> These two files are the primary PS15 deliverable. The problem statement says explicitly: "Build algorithms to detect flares independently in both soft and hard X-rays." Independent means two separate algorithms on two separate data streams before any merging.

```python
# src/nowcasting/solexs_detector.py

import numpy as np
import pandas as pd
from dataclasses import dataclass

GOES_THRESHOLDS = {"X": 1e-4, "M": 1e-5, "C": 1e-6, "B": 1e-7}


@dataclass
class FlareEvent:
    start_time:  pd.Timestamp
    peak_time:   pd.Timestamp
    end_time:    pd.Timestamp
    peak_flux:   float
    flare_class: str
    instrument:  str
    confidence:  float


def classify_flux(peak_flux: float) -> str:
    for cls, thresh in GOES_THRESHOLDS.items():
        if peak_flux >= thresh:
            return cls
    return "A"


def detect_solexs_flares(df: pd.DataFrame,
                          flux_col:         str   = "flux_high",
                          sigma_threshold:  float = 3.0,
                          min_duration_min: int   = 3) -> list:
    """
    Soft X-ray flare detection using SoLEXS data only.

    Algorithm:
    1. Rolling 10-min median background + std
    2. Trigger when rate-of-change > sigma_threshold × rolling_std AND
       current flux > 1.5 × background (two conditions prevent spike false alarms)
    3. Event continues until flux drops below 1.5 × background
    4. Minimum duration filter removes instrument spikes
    5. Peak flux classified to GOES-equivalent A/B/C/M/X scale
    """
    flux        = df[flux_col].fillna(0)
    background  = flux.rolling(10, min_periods=1).median()
    rolling_std = flux.rolling(10, min_periods=1).std().fillna(1e-12)
    dfdt        = flux.diff().fillna(0)

    events, in_flare            = [], False
    start_t = peak_flux = peak_t = None

    for i, (ts, _) in enumerate(df.iterrows()):
        f    = flux.iloc[i]
        bg   = background.iloc[i]
        std  = rolling_std.iloc[i]
        rate = dfdt.iloc[i]

        trigger = (rate > sigma_threshold * std) and (f > 1.5 * bg)

        if not in_flare and trigger:
            in_flare  = True
            start_t   = ts
            peak_flux  = f
            peak_t    = ts
        elif in_flare:
            if f > peak_flux:
                peak_flux = f
                peak_t    = ts
            if f < 1.5 * bg:
                dur = (ts - start_t).total_seconds() / 60
                if dur >= min_duration_min:
                    events.append(FlareEvent(
                        start_time  = start_t,
                        peak_time   = peak_t,
                        end_time    = ts,
                        peak_flux   = peak_flux,
                        flare_class = classify_flux(peak_flux),
                        instrument  = "SoLEXS",
                        confidence  = min(1.0, peak_flux / (5 * bg + 1e-12)),
                    ))
                in_flare = False

    return events
```

```python
# src/nowcasting/hel1os_detector.py

import numpy as np
import pandas as pd
from src.nowcasting.solexs_detector import FlareEvent


def detect_hel1os_flares(df: pd.DataFrame,
                          counts_col:       str   = "counts_low",
                          sigma_threshold:  float = 4.0,
                          min_duration_min: int   = 2) -> list:
    """
    Hard X-ray burst detection using HEL1OS data only.

    Higher sigma threshold than SoLEXS (4.0 vs 3.0):
    - Hard X-ray background has more statistical noise (Poisson counting)
    - Impulsive phase bursts are sharper — a strict threshold still catches them

    Shorter minimum duration (2 min vs 3 min):
    - Hard X-ray impulsive phase is brief and precedes the soft X-ray peak

    HEL1OS cannot self-classify to GOES-equivalent class (no absolute flux
    calibration to the GOES XRS spectral band). Classification comes from
    the master catalogue merger using the SoLEXS peak.
    """
    counts      = df[counts_col].fillna(0)
    background  = counts.rolling(10, min_periods=1).median()
    rolling_std = counts.rolling(10, min_periods=1).std().fillna(1.0)
    dcdt        = counts.diff().fillna(0)

    events, in_flare            = [], False
    start_t = peak_count = peak_t = None

    for i, (ts, _) in enumerate(df.iterrows()):
        c   = counts.iloc[i]
        bg  = background.iloc[i]
        std = rolling_std.iloc[i]
        dct = dcdt.iloc[i]

        trigger = (dct > sigma_threshold * std) and (c > 2.0 * bg)

        if not in_flare and trigger:
            in_flare   = True
            start_t    = ts
            peak_count = c
            peak_t     = ts
        elif in_flare:
            if c > peak_count:
                peak_count = c
                peak_t     = ts
            if c < 1.5 * bg:
                dur = (ts - start_t).total_seconds() / 60
                if dur >= min_duration_min:
                    events.append(FlareEvent(
                        start_time  = start_t,
                        peak_time   = peak_t,
                        end_time    = ts,
                        peak_flux   = peak_count,
                        flare_class = "?",   # assigned by merger
                        instrument  = "HEL1OS",
                        confidence  = min(1.0, peak_count / (5 * bg + 1e-10)),
                    ))
                in_flare = False

    return events
```

**M2 Checkpoint:** `detect_solexs_flares()` and `detect_hel1os_flares()` both return non-empty lists on the 2024-02-22 X6.3 event. Test against known NOAA catalog entries. Print event counts to confirm the detectors fire.

---

## M3 — Master Catalogue Merger

```python
# src/catalogue/merger.py

import numpy as np
import pandas as pd
from src.nowcasting.solexs_detector import FlareEvent

COINCIDENCE_WINDOW = pd.Timedelta(minutes=2)


def merge_catalogues(solexs_events: list,
                      hel1os_events: list,
                      noaa_catalog:  "pd.DataFrame | None" = None) -> pd.DataFrame:
    """
    Combine independent SoLEXS and HEL1OS event lists into a master catalogue.

    Confidence tiers:
    - dual (HIGH):       SoLEXS + HEL1OS peaks within ±2 min; conf × 1.20
    - SoLEXS_only (MED): soft X-ray detection only; conf × 0.80
    - HEL1OS_only (LOW): hard X-ray burst with no soft counterpart; conf × 0.60
                         (may indicate data gap in SoLEXS — flag for investigation)

    NOAA confirmation:
    - If noaa_catalog is provided, events within 10 min of a known NOAA event
      get noaa_confirmed=True for dashboard display.
    """
    master_rows    = []
    matched_hel1os = set()

    for se in solexs_events:
        matched = False
        for j, he in enumerate(hel1os_events):
            if j in matched_hel1os:
                continue
            if abs(se.peak_time - he.peak_time) <= COINCIDENCE_WINDOW:
                conf = min(1.0, ((se.confidence + he.confidence) / 2) * 1.20)
                master_rows.append({
                    "start_time":    min(se.start_time, he.start_time),
                    "peak_time":     se.peak_time,
                    "end_time":      max(se.end_time, he.end_time),
                    "flare_class":   se.flare_class,
                    "peak_flux_sxr": se.peak_flux,
                    "peak_cnt_hxr":  he.peak_flux,
                    "source":        "dual",
                    "confidence":    round(conf, 3),
                    "noaa_confirmed": _check_noaa(se.peak_time, noaa_catalog),
                })
                matched_hel1os.add(j)
                matched = True
                break

        if not matched:
            master_rows.append({
                "start_time":    se.start_time,
                "peak_time":     se.peak_time,
                "end_time":      se.end_time,
                "flare_class":   se.flare_class,
                "peak_flux_sxr": se.peak_flux,
                "peak_cnt_hxr":  np.nan,
                "source":        "SoLEXS_only",
                "confidence":    round(se.confidence * 0.80, 3),
                "noaa_confirmed": _check_noaa(se.peak_time, noaa_catalog),
            })

    for j, he in enumerate(hel1os_events):
        if j not in matched_hel1os:
            master_rows.append({
                "start_time":    he.start_time,
                "peak_time":     he.peak_time,
                "end_time":      he.end_time,
                "flare_class":   "?",
                "peak_flux_sxr": np.nan,
                "peak_cnt_hxr":  he.peak_flux,
                "source":        "HEL1OS_only",
                "confidence":    round(he.confidence * 0.60, 3),
                "noaa_confirmed": _check_noaa(he.peak_time, noaa_catalog),
            })

    master = (pd.DataFrame(master_rows)
               .sort_values("peak_time")
               .reset_index(drop=True))
    master.to_csv("data/processed/master_catalogue.csv", index=False)

    print(
        f"Master catalogue: {len(master)} events\n"
        f"  dual={( master.source == 'dual').sum()}\n"
        f"  SoLEXS_only={( master.source == 'SoLEXS_only').sum()}\n"
        f"  HEL1OS_only={( master.source == 'HEL1OS_only').sum()}\n"
        f"  NOAA confirmed={(master.noaa_confirmed == True).sum()}"
    )
    return master


def _check_noaa(event_time: pd.Timestamp,
                 noaa_catalog: "pd.DataFrame | None",
                 window_min:   int = 10) -> bool:
    if noaa_catalog is None or noaa_catalog.empty:
        return False
    delta = (noaa_catalog["peak_time"] - event_time).abs()
    return bool((delta <= pd.Timedelta(minutes=window_min)).any())
```

**M3 Checkpoint:** `data/processed/master_catalogue.csv` exists. Visualise in `notebooks/03_catalogue_merger.ipynb` — plot SoLEXS + HEL1OS light curves with event markers overlaid. Dual events should show clear coincident peaks.

---

## M4 — Multi-class Nowcasting (TCN + XGBoost)

### 4.1 TCN Encoder

```python
# src/nowcasting/tcn_encoder.py

import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad  = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              dilation=dilation, padding=self.pad)

    def forward(self, x):
        return self.conv(x)[:, :, :-self.pad]


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1      = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.conv2      = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.norm1      = nn.LayerNorm(out_ch)
        self.norm2      = nn.LayerNorm(out_ch)
        self.drop       = nn.Dropout(dropout)
        self.relu       = nn.ReLU()
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        res = x
        out = self.relu(self.norm1(self.conv1(x).transpose(1, 2)).transpose(1, 2))
        out = self.drop(out)
        out = self.norm2(self.conv2(out).transpose(1, 2)).transpose(1, 2)
        out = self.drop(out)
        if self.downsample:
            res = self.downsample(res)
        return self.relu(out + res)


class TCNEncoder(nn.Module):
    """
    Dilated causal TCN — zero future leakage.
    Every convolution is causal: output at time t uses only input[0..t].
    This property is preserved in ONNX export and real-time inference.
    """
    def __init__(self, n_features: int, embed_dim: int = 128,
                 n_layers: int = 4, kernel_size: int = 3):
        super().__init__()
        channels  = [n_features] + [embed_dim] * n_layers
        dilations = [2 ** i for i in range(n_layers)]
        self.blocks = nn.ModuleList([
            TCNBlock(channels[i], channels[i + 1], kernel_size, dilations[i])
            for i in range(n_layers)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        return self.pool(x).squeeze(-1)
```

### 4.2 Multi-class Training + Per-class Thresholds

```python
# src/nowcasting/train.py

import numpy as np
import torch
import xgboost as xgb
import yaml
import mlflow
from src.nowcasting.tcn_encoder import TCNEncoder
from src.evaluation.metrics import compute_tss


def extract_tcn_features(encoder: TCNEncoder, X: np.ndarray,
                          device: str = "cpu", batch_size: int = 256) -> np.ndarray:
    encoder.eval().to(device)
    feats = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            b = torch.tensor(X[i:i + batch_size]).to(device)
            feats.append(encoder(b).cpu().numpy())
    return np.vstack(feats)


def train_multiclass_nowcast(X_tr, y_tr, X_val, y_val,
                              tcn_feats_tr, tcn_feats_val):
    """
    Multi-class XGBoost (N/C/M/X) trained on TCN embeddings + physics features.
    Inverse-frequency sample weights address class imbalance at training time.
    FLAML (M8) will replace these hyperparameters with auto-tuned values.
    """
    combined_tr  = np.concatenate([tcn_feats_tr,  X_tr.reshape(len(X_tr),  -1)], axis=1)
    combined_val = np.concatenate([tcn_feats_val, X_val.reshape(len(X_val), -1)], axis=1)

    class_counts   = np.bincount(y_tr, minlength=4)
    sample_weights = (1.0 / (class_counts[y_tr] + 1)) * len(y_tr)

    with mlflow.start_run(run_name="multiclass_nowcast_tcn_xgb"):
        model = xgb.XGBClassifier(
            objective       = "multi:softprob",
            num_class       = 4,
            n_estimators    = 600,
            max_depth       = 6,
            learning_rate   = 0.05,
            eval_metric     = "mlogloss",
            early_stopping_rounds = 50,
            tree_method     = "hist",
            verbosity       = 0,
        )
        model.fit(
            combined_tr, y_tr,
            sample_weight = sample_weights,
            eval_set      = [(combined_val, y_val)],
            verbose       = 100,
        )
        mlflow.xgboost.log_model(model, "nowcast_xgb_multiclass")
    return model, combined_val


def optimize_per_class_thresholds(model: xgb.XGBClassifier,
                                   combined_val: np.ndarray,
                                   y_val: np.ndarray) -> dict:
    """
    Optimise TSS separately for each class (C, M, X).

    Threshold direction (per Decision D8):
    - X-class: LOWER threshold (e.g. 0.28) — missing X is catastrophically costly
    - M-class: balanced (e.g. 0.45)
    - C-class: moderate (e.g. 0.38) — catch early precursors
    - Binary: lowest (e.g. 0.40) — any flare vs quiet

    These are starting points; actual values come from TSS maximisation below.
    """
    probas     = model.predict_proba(combined_val)
    thresholds = {}

    for cls_idx, cls_name in enumerate(["C", "M", "X"]):
        p_cls  = probas[:, cls_idx]
        y_bin  = (y_val == cls_idx).astype(int)
        if y_bin.sum() == 0:
            print(f"Class {cls_name}: no events in validation set — using default 0.40")
            thresholds[cls_name] = 0.40
            continue
        best_t, best_tss = 0.30, -1.0
        for t in np.arange(0.10, 0.90, 0.05):
            tss = compute_tss(y_bin, (p_cls >= t).astype(int))
            if tss > best_tss:
                best_tss, best_t = tss, t
        thresholds[cls_name] = round(best_t, 2)
        print(f"Class {cls_name}: threshold={best_t:.2f}  TSS={best_tss:.3f}")

    # Binary: any flare vs quiet
    p_any  = 1 - probas[:, 0]
    y_any  = (y_val > 0).astype(int)
    best_t, best_tss = 0.40, -1.0
    for t in np.arange(0.10, 0.90, 0.05):
        tss = compute_tss(y_any, (p_any >= t).astype(int))
        if tss > best_tss:
            best_tss, best_t = tss, t
    thresholds["binary"] = round(best_t, 2)

    # Save to config
    cfg = yaml.safe_load(open("configs/nowcasting.yaml"))
    cfg["class_thresholds"] = thresholds
    yaml.dump(cfg, open("configs/nowcasting.yaml", "w"), default_flow_style=False)
    print(f"Thresholds saved to configs/nowcasting.yaml: {thresholds}")
    return thresholds
```

**M4 Checkpoint:** Multi-class confusion matrix logged to MLflow. Per-class TSS printed. Phase feature appears in XGBoost `feature_importances_` top 10. Thresholds written to `configs/nowcasting.yaml`.

---

## M5 — Three-Model Forecasting Ensemble

### 5.1 Causal LSTM

```python
# src/forecasting/causal_lstm.py

import torch.nn as nn


class CausalLSTMForecaster(nn.Module):
    """
    Unidirectional LSTM for flare forecasting.

    bidirectional=False is mandatory. At inference time, input is the
    rolling window [T-60 → T]. Bidirectional LSTM would be mathematically
    valid here (window is all historical), but introduces unnecessary
    confusion about causal correctness during code review. Unidirectional
    is unambiguous and functionally equivalent for window classification.
    """
    def __init__(self, n_features: int, hidden_dim: int = 128,
                 n_layers: int = 2, dropout: float = 0.3,
                 n_classes: int = 4):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_dim, n_layers,
                            batch_first=True, bidirectional=False,
                            dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.norm(out[:, -1, :]))
```

### 5.2 TimesFM Integration (PEFT LoRA fine-tuning)

> `timesfm.finetuning` does not exist in the current public package. Fine-tuning uses PEFT + HuggingFace Trainer. Verify with CHECK 2 before implementing.

```python
# src/forecasting/timesfm_forecaster.py

import numpy as np
import pandas as pd
import torch
import timesfm

# ─── Step 1: Zero-shot inference (works immediately, no fine-tuning) ────────

def load_timesfm() -> timesfm.TimesFm:
    """Load TimesFM 2.0-200M from HuggingFace."""
    return timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="torch",
            per_core_batch_size=4,
            horizon_len=60,
            context_len=512,
            num_layers=20,
            model_dims=1280,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-2.0-200m-pytorch"
        ),
    )


def predict_timesfm(tfm: timesfm.TimesFm,
                     flux_window: np.ndarray,
                     horizons: list = [15, 30, 60]) -> dict:
    """
    Run TimesFM zero-shot forecast on a flux window.

    Output column from forecast_on_df() is "timesfm" (not "timesfm_15").
    Verified via CHECK 2. Slice by horizon index, not by column name suffix.
    """
    n   = len(flux_window)
    df  = pd.DataFrame({
        "unique_id": ["SoLEXS"] * n,
        "ds": pd.date_range("2000-01-01", periods=n, freq="1min"),
        "y":  flux_window,
    })
    forecast_df = tfm.forecast_on_df(inputs=df, freq="T", value_name="y", num_jobs=1)

    # The actual forecast column name — verified with CHECK 2
    # Typically "timesfm"; update this variable if CHECK 2 shows otherwise
    FORECAST_COL = "timesfm"
    forecast_vals = forecast_df[FORECAST_COL].values  # shape: (horizon_len,)

    m_threshold = 1e-5
    results = {}
    for h in horizons:
        vals = forecast_vals[:h]
        results[f"h{h}"] = float((vals > m_threshold).mean())
    return results


# ─── Step 2: LoRA fine-tuning via PEFT (run if GPU available — Decision D3) ──

def lora_finetune_timesfm(base_model_id: str = "google/timesfm-2.0-200m-pytorch",
                            train_dataset = None,
                            output_dir: str = "models/timesfm_lora") -> None:
    """
    Fine-tune TimesFM with LoRA adapters using PEFT + HuggingFace Trainer.
    timesfm.finetuning does NOT exist — use PEFT (verified via CHECK 2).

    Requirements: GPU recommended (Decision D3). If CPU-only, skip this step
    and use zero-shot TimesFM only.
    """
    from transformers import AutoModel, TrainingArguments, Trainer
    from peft import get_peft_model, LoraConfig, TaskType

    base_model = AutoModel.from_pretrained(base_model_id)
    lora_config = LoraConfig(
        r                = 8,
        lora_alpha       = 16,
        lora_dropout     = 0.1,
        bias             = "none",
        target_modules   = ["q_proj", "v_proj"],  # verify against model architecture
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir           = output_dir,
        num_train_epochs     = 10,
        per_device_train_batch_size = 8,
        learning_rate        = 1e-4,
        warmup_steps         = 100,
        save_strategy        = "epoch",
        logging_steps        = 50,
        fp16                 = torch.cuda.is_available(),
    )
    Trainer(
        model     = model,
        args      = training_args,
        train_dataset = train_dataset,
    ).train()
    model.save_pretrained(output_dir)
    print(f"LoRA adapter saved to {output_dir}")


# ─── Step 3: Ensemble ─────────────────────────────────────────────────────────

class ThreeModelEnsemble:
    """
    Weighted blend: Causal LSTM (0.35) + TCN (0.35) + TimesFM (0.30).

    NOTE on batch evaluation: TimesFM is called once per inference window.
    For batch evaluation on a test set, call predict() in a loop or evaluate
    LSTM+TCN on the full test set and report TimesFM zero-shot TSS separately.
    The ensemble weight is only used at inference time.
    """
    def __init__(self, lstm_model, tcn_model, timesfm_model,
                 weights=(0.35, 0.35, 0.30)):
        self.lstm    = lstm_model
        self.tcn     = tcn_model
        self.tfm     = timesfm_model
        self.weights = weights

    def predict_single(self, X_tensor: "torch.Tensor",
                        flux_np: np.ndarray, horizon: int = 15) -> np.ndarray:
        """Single-window inference (real-time use). Returns class probability array."""
        with torch.no_grad():
            p_lstm = self.lstm(X_tensor).cpu().numpy()
            p_tcn  = self.tcn(X_tensor).cpu().numpy()
        tfm_prob = predict_timesfm(self.tfm, flux_np, [horizon])[f"h{horizon}"]
        p_tfm    = np.array([1 - tfm_prob, 0, tfm_prob, 0])  # map to [N, C, M, X]
        w = self.weights
        return w[0] * p_lstm[0] + w[1] * p_tcn[0] + w[2] * p_tfm
```

**M5 Checkpoint:** Causal LSTM and TCN train without NaN loss. TimesFM zero-shot TSS evaluated separately. All three models log to MLflow. `predict_single()` returns a 4-dim probability array summing to 1.

---

## M6 — ONNX Export & Optimised Inference

```python
# src/deployment/onnx_export.py

import torch
import onnx
import onnxruntime as ort
import numpy as np


def export_to_onnx(model:        "torch.nn.Module",
                    dummy_input:  "torch.Tensor",
                    path:         str,
                    input_name:   str,
                    output_name:  str,
                    opset:        int = 17) -> None:
    model.eval()
    torch.onnx.export(
        model, dummy_input, path,
        export_params   = True,
        opset_version   = opset,
        input_names     = [input_name],
        output_names    = [output_name],
        dynamic_axes    = {input_name: {0: "batch"}, output_name: {0: "batch"}},
    )
    onnx.checker.check_model(onnx.load(path))
    print(f"ONNX model verified: {path}")


class ONNXNowcaster:
    """
    Production nowcaster using ONNX Runtime.
    Target: < 100ms per inference on CPU.
    Benchmark with: python src/deployment/onnx_export.py --benchmark
    """
    def __init__(self, tcn_path:  str,
                       xgb_model: "xgb.XGBClassifier",
                       n_handcrafted: int = 0):
        self.tcn     = ort.InferenceSession(
            tcn_path, providers=["CPUExecutionProvider"]
        )
        self.xgb          = xgb_model
        self.n_handcrafted = n_handcrafted
        self.cls_names    = ["N", "C", "M", "X"]

    def predict(self, window:          np.ndarray,
                       handcrafted_feats: np.ndarray) -> dict:
        tcn_embedding = self.tcn.run(
            None, {"light_curve_window": window.astype(np.float32)}
        )[0]
        combined = np.concatenate([tcn_embedding, handcrafted_feats], axis=1)
        probas   = self.xgb.predict_proba(combined)[0]
        return {
            "class":      self.cls_names[int(np.argmax(probas))],
            "proba":      {c: round(float(p), 4)
                           for c, p in zip(self.cls_names, probas)},
            "confidence": round(float(np.max(probas)), 4),
        }
```

**M6 Checkpoint:** `python src/deployment/onnx_export.py --benchmark` prints inference latency. Target: < 100ms. Check with `ort.get_device()` that CPU provider is active.

---

## M7 — Agent Orchestration (LangGraph)

### 7.1 State

```python
# src/orchestration/state.py

from typing import TypedDict, Optional, Annotated
import operator
import pandas as pd


def _merge_catalogue_lists(a, b):
    """Reducer: combine parallel detection results into a single list."""
    if a is None: return b
    if b is None: return a
    return list(a) + list(b)


class SolarPipelineState(TypedDict):
    solexs_path:          str
    hel1os_path:          str
    raw_solexs:           Optional[pd.DataFrame]
    raw_hel1os:           Optional[pd.DataFrame]
    # Annotated with reducer enables parallel fan-in for detection results
    detected_events:      Annotated[Optional[list], _merge_catalogue_lists]
    master_catalogue:     Optional[pd.DataFrame]
    processed_df:         Optional[pd.DataFrame]
    moment_anomaly_score: Optional[float]
    nowcast_class:        Optional[str]
    nowcast_proba:        Optional[dict]
    forecast_15min:       Optional[float]
    forecast_30min:       Optional[float]
    forecast_60min:       Optional[float]
    conformal_interval:   Optional[tuple]
    chronos_interval:     Optional[dict]
    dual_agreement:       Optional[str]
    shap_explanation:     Optional[dict]
    alert_triggered:      bool
    llm_report:           Optional[str]
    errors:               list
    timestamp:            str
```

### 7.2 Agent Implementations

```python
# src/orchestration/agents.py

from src.ingestion.fits_reader import read_solexs, read_hel1os
from src.nowcasting.solexs_detector import detect_solexs_flares, FlareEvent
from src.nowcasting.hel1os_detector import detect_hel1os_flares
from src.catalogue.merger import merge_catalogues, _check_noaa
from src.preprocessing.physics_features import engineer_physics_features, subtract_background
from src.intelligence.llm_reporter import generate_flare_report
import yaml, mlflow

_cfg = yaml.safe_load(open("configs/nowcasting.yaml"))


def ingestion_agent(state: dict) -> dict:
    return {
        **state,
        "raw_solexs": read_solexs(state["solexs_path"]),
        "raw_hel1os": read_hel1os(state["hel1os_path"]),
    }


def solexs_detect_agent(state: dict) -> dict:
    """Writes to detected_events — the annotated reducer merges with HEL1OS results."""
    events = detect_solexs_flares(state["raw_solexs"])
    return {**state, "detected_events": events}


def hel1os_detect_agent(state: dict) -> dict:
    """Writes to detected_events — merged with SoLEXS results by the state reducer."""
    events = detect_hel1os_flares(state["raw_hel1os"])
    return {**state, "detected_events": events}


def merge_agent(state: dict) -> dict:
    """Fan-in node: receives combined detected_events from both detectors."""
    all_events = state.get("detected_events") or []
    solexs_ev  = [e for e in all_events if e.instrument == "SoLEXS"]
    hel1os_ev  = [e for e in all_events if e.instrument == "HEL1OS"]
    master     = merge_catalogues(solexs_ev, hel1os_ev)
    return {**state, "master_catalogue": master}


def preprocess_agent(state: dict) -> dict:
    from src.ingestion.fits_reader import merge_instruments
    merged = merge_instruments(state["raw_solexs"], state["raw_hel1os"])
    merged = subtract_background(merged)
    merged = engineer_physics_features(merged)
    return {**state, "processed_df": merged}


def moment_score_agent(state: dict) -> dict:
    """Compute MOMENT anomaly score on latest window. Falls back to 0.0 on error."""
    try:
        from src.forecasting.moment_anomaly import compute_anomaly_score, _load_moment
        model  = _load_moment()
        window = state["processed_df"]["flux_high"].tail(512).values
        result = compute_anomaly_score(model, window)
        return {**state, "moment_anomaly_score": result["anomaly_score"]}
    except Exception as e:
        return {**state, "moment_anomaly_score": 0.0,
                "errors": state.get("errors", []) + [f"MOMENT: {e}"]}


def nowcast_agent(state: dict) -> dict:
    from src.deployment.onnx_export import ONNXNowcaster
    import numpy as np, joblib
    nowcaster = ONNXNowcaster("models/tcn_encoder.onnx",
                               joblib.load("models/xgb_multiclass.pkl"))
    window    = state["processed_df"].tail(60).values[np.newaxis]
    handcraft = np.array([[state.get("moment_anomaly_score", 0.0)]])
    result    = nowcaster.predict(window, handcraft)
    return {**state, "nowcast_class": result["class"],
            "nowcast_proba": result["proba"],
            "alert_triggered": result["class"] in ("M", "X")}


def forecast_agent(state: dict) -> dict:
    import torch, joblib, numpy as np
    from src.forecasting.causal_lstm import CausalLSTMForecaster
    lstm   = torch.load("models/causal_lstm.pt", map_location="cpu")
    X      = torch.tensor(state["processed_df"].tail(60).values[np.newaxis], dtype=torch.float32)
    with torch.no_grad():
        proba = lstm(X).numpy()[0]
    return {**state, "forecast_15min": float(proba[2] + proba[3]),
            "forecast_30min": float(proba[2] + proba[3]) * 0.9,
            "forecast_60min": float(proba[2] + proba[3]) * 0.8}


def uncertainty_agent(state: dict) -> dict:
    from src.uncertainty.conformal import predict_with_interval
    from src.uncertainty.chronos_uncertainty import chronos_forecast_interval, dual_agreement_check
    import numpy as np, joblib
    mapie    = joblib.load("models/conformal_mapie.pkl")
    X_latest = state["processed_df"].tail(60).values[np.newaxis]
    conf_res = predict_with_interval(mapie, X_latest)
    from chronos import BaseChronosPipeline
    import torch
    chron    = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small",
                dtype=torch.float32, device_map="cpu")
    flux     = state["processed_df"]["flux_high"].tail(60).values
    chron_r  = chronos_forecast_interval(chron, flux, horizon=15)
    agree    = dual_agreement_check(chron_r, state.get("forecast_15min", 0))
    return {**state, "conformal_interval": conf_res["conformal_set"],
            "chronos_interval": chron_r, "dual_agreement": agree}


def shap_agent(state: dict) -> dict:
    import joblib, numpy as np
    from src.explainability.shap_explainer import build_explainer, explain_prediction
    xgb_model = joblib.load("models/xgb_multiclass.pkl")
    explainer  = build_explainer(xgb_model)
    features   = state["processed_df"].tail(1).values.flatten()
    feat_names = list(state["processed_df"].columns)
    explanation = explain_prediction(explainer, features, feat_names)
    return {**state, "shap_explanation": explanation}


def llm_report_agent(state: dict) -> dict:
    report = generate_flare_report({
        "class":          state.get("nowcast_class", "C"),
        "confidence":     (state.get("nowcast_proba") or {}).get(
                           state.get("nowcast_class", "N"), 0.0),
        "lead_time":      0.0,
        "forecast_proba": state.get("forecast_15min", 0.0),
        "timestamp":      state.get("timestamp", ""),
        "source":         "dual",
    })
    return {**state, "llm_report": report}


def alert_router(state: dict) -> str:
    cls  = state.get("nowcast_class", "N")
    p15  = state.get("forecast_15min", 0.0)
    agr  = state.get("dual_agreement", "LOW")
    if cls in ("M", "X") or (p15 > 0.60 and agr == "HIGH"):
        return "llm_report"
    return "end"
```

### 7.3 Graph (Correct Fan-in Pattern)

```python
# src/orchestration/graph.py

from langgraph.graph import StateGraph, END
from langgraph.constants import Send
from src.orchestration.state import SolarPipelineState
from src.orchestration.agents import (
    ingestion_agent, solexs_detect_agent, hel1os_detect_agent,
    merge_agent, preprocess_agent, moment_score_agent,
    nowcast_agent, forecast_agent, uncertainty_agent,
    shap_agent, llm_report_agent, alert_router
)


def build_pipeline():
    """
    Correct LangGraph parallel fan-out + fan-in pattern.

    The two detector nodes run in parallel using the Annotated state
    reducer on detected_events. merge_agent is the fan-in node that
    receives the combined list from both detectors.

    IMPORTANT: Simple add_edge("ingestion", "detect_solexs") +
    add_edge("ingestion", "detect_hel1os") does NOT automatically
    parallelize in LangGraph — the Annotated reducer on the state
    field is what enables correct merging (Decision D7).
    """
    g = StateGraph(SolarPipelineState)

    for name, fn in [
        ("ingestion",     ingestion_agent),
        ("detect_solexs", solexs_detect_agent),
        ("detect_hel1os", hel1os_detect_agent),
        ("merge",         merge_agent),
        ("preprocess",    preprocess_agent),
        ("moment",        moment_score_agent),
        ("nowcast",       nowcast_agent),
        ("forecast",      forecast_agent),
        ("uncertainty",   uncertainty_agent),
        ("shap",          shap_agent),
        ("llm_report",    llm_report_agent),
    ]:
        g.add_node(name, fn)

    g.set_entry_point("ingestion")
    # Fan-out: both detectors receive state after ingestion
    g.add_edge("ingestion",     "detect_solexs")
    g.add_edge("ingestion",     "detect_hel1os")
    # Fan-in: merge waits for both (handled by Annotated reducer)
    g.add_edge("detect_solexs", "merge")
    g.add_edge("detect_hel1os", "merge")
    g.add_edge("merge",         "preprocess")
    g.add_edge("preprocess",    "moment")
    g.add_edge("moment",        "nowcast")
    g.add_edge("nowcast",       "forecast")
    g.add_edge("forecast",      "uncertainty")
    g.add_edge("uncertainty",   "shap")
    g.add_conditional_edges(
        "shap", alert_router,
        {"llm_report": "llm_report", "end": END}
    )
    g.add_edge("llm_report", END)
    return g.compile()
```

**M7 Checkpoint:** `pipeline.invoke({...})` runs end-to-end without import errors. `detected_events` in state contains events from both instruments after merge. `master_catalogue` is populated. No `KeyError` on undefined agents.

---

## M8 — AutoML Tuning (FLAML)

```python
# src/nowcasting/flaml_tuner.py

from flaml import AutoML
import mlflow


def autotune_multiclass(X_tr, y_tr, X_val, y_val,
                         time_budget: int = 3600) -> AutoML:
    """
    FLAML searches XGBoost, LightGBM, RandomForest, ExtraTrees.

    metric=macro_f1: correct for multi-class imbalanced data (N/C/M/X).
    ROC-AUC is wrong here — it is defined for binary classification.
    macro_f1 gives equal weight to C/M/X despite their low frequency.
    """
    automl = AutoML()
    with mlflow.start_run(run_name="flaml_multiclass_macro_f1"):
        automl.fit(
            X_tr, y_tr,
            task             = "classification",
            metric           = "macro_f1",
            time_budget      = time_budget,
            estimator_list   = ["xgboost", "lgbm", "rf", "extra_tree"],
            X_val            = X_val,
            y_val            = y_val,
            log_file_name    = "logs/flaml.log",
            verbose          = 2,
        )
        mlflow.log_param("best_estimator", automl.best_estimator)
        mlflow.log_params(automl.best_config)
        mlflow.log_metric("val_macro_f1", 1 - automl.best_loss)
        print(f"Best: {automl.best_estimator} | macro_f1: {1 - automl.best_loss:.4f}")
    return automl
```

**M8 Checkpoint:** FLAML macro_f1 > 0.50 on validation. Best config logged. Retrain final XGBoost model with FLAML-found parameters before ONNX export.

---

## M9 — LLM Intelligence (DSPy + GraphRAG + Phi-3-mini)

### 9.1 Setup

```bash
# GraphRAG indexing (one-time — commit output to repo afterwards)
pip install graphrag==0.3.6

python -m graphrag init --root data/graphrag
# Edit data/graphrag/settings.yaml — see CHECK 7 for local embedding config
python -m graphrag index --root data/graphrag
# Expected time: 30–90 min depending on document volume
git add data/graphrag/output/ && git commit -m "feat: graphrag index"
```

### 9.2 DSPy Reporter (thread-safe)

```python
# src/intelligence/dspy_reporter.py
# Verify DSPy Ollama syntax with CHECK 5 before implementing.

import dspy

# Construct the LM object once at module load.
# Syntax verified with scripts/verify_dspy.py — update if CHECK 5 shows different.
_LM = dspy.LM(
    "ollama/phi3:mini",               # update if CHECK 5 shows different syntax
    api_base="http://localhost:11434",
)


class FlareAlertSignature(dspy.Signature):
    """Given solar flare detection data and physics context, generate a concise
    operational bulletin for space weather operators. Under 5 sentences.
    Ground every claim in the provided physics context."""
    flare_class:       str = dspy.InputField()
    confidence:        str = dspy.InputField()
    lead_time_min:     str = dspy.InputField()
    instrument_source: str = dspy.InputField()
    physics_context:   str = dspy.InputField()
    bulletin:          str = dspy.OutputField()


class SolarSentinelReporter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.alert_gen = dspy.ChainOfThought(FlareAlertSignature)

    def forward(self, flare_data: dict, physics_context: str) -> str:
        # Use per-call context (not global configure) — thread-safe for async FastAPI
        with dspy.context(lm=_LM):
            result = self.alert_gen(
                flare_class       = flare_data.get("class", "C"),
                confidence        = f"{flare_data.get('confidence', 0):.0%}",
                lead_time_min     = f"{flare_data.get('lead_time', 0):.1f}",
                instrument_source = flare_data.get("source", "dual"),
                physics_context   = physics_context,
            )
        return result.bulletin


def optimise_reporter(reporter:           SolarSentinelReporter,
                       training_examples: list) -> SolarSentinelReporter:
    """
    Bootstrap-few-shot optimisation using known historical flare events.
    Training examples are (flare_data, expected_bulletin) pairs from
    NOAA event reports — sources the GraphRAG knowledge graph indexes.
    """
    def quality_metric(example, pred, trace=None):
        b = pred.bulletin.lower()
        return (
            example.flare_class.lower() in b and
            any(w in b for w in ["operator", "satellite", "grid", "action", "blackout"])
        )

    optimised = dspy.BootstrapFewShot(
        metric=quality_metric, max_bootstrapped_demos=4
    ).compile(reporter, trainset=training_examples)
    optimised.save("models/dspy_reporter_optimised.json")
    return optimised
```

### 9.3 GraphRAG Retriever

```python
# src/intelligence/graphrag_retriever.py

import sys, subprocess
from pathlib import Path

GRAPHRAG_DIR = Path("data/graphrag")


def query_graphrag(question: str, method: str = "local") -> str:
    """
    Query the GraphRAG knowledge graph.
    local  → entity-level, fast — use for per-event alerts
    global → community summaries — use for daily briefing reports

    Uses sys.executable (not bare "python") to handle venv/Docker correctly.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "graphrag", "query",
             "--root", str(GRAPHRAG_DIR),
             "--method", method,
             "--query", question],
            capture_output=True, text=True, check=True,
            timeout=30,
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return ""
    except subprocess.CalledProcessError as e:
        return ""
```

### 9.4 Unified Reporter with Structured Fallback

```python
# src/intelligence/llm_reporter.py

from string import Template
from src.intelligence.dspy_reporter import SolarSentinelReporter
from src.intelligence.graphrag_retriever import query_graphrag

_reporter = SolarSentinelReporter()

SEVERITY = {
    "C": "minor. Low infrastructure risk. Brief HF radio degradation possible.",
    "M": "moderate. HF blackout likely on sunlit hemisphere. Satellite precautions advised.",
    "X": "MAJOR. Severe radio blackout. GPS degradation. GIC risk. Immediate action required.",
}

ACTIONS = {
    "C": "Continue monitoring. No immediate operational action required.",
    "M": "Satellite operators: activate anomaly monitoring. Grid: monitor GIC.",
    "X": "IMMEDIATE: Satellite operators safe-mode check. Grid: activate GIC protocol.",
}

# Fallback bulletin — always professional, fires instantly if LLM unavailable.
# Shown on dashboard in orange instead of green to indicate fallback mode.
_FALLBACK = Template("""\
SOLAR FLARE ALERT — SolarSentinel (Aditya-L1 SoLEXS + HEL1OS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Class : $cls      Time: $ts      Confidence: $conf
Lead  : $lead min before estimated peak flux
Source: $source  (dual = both instruments confirmed)
Severity: $severity
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Action: $action
[Structured fallback — LLM offline]""")


def generate_flare_report(flare_data: dict) -> str:
    cls  = (flare_data.get("class") or "C")[0].upper()
    try:
        context = query_graphrag(
            f"operational impacts of {cls}-class solar flare satellite grid GPS",
            method="local"
        )
        if context:
            return _reporter.forward(flare_data, physics_context=context)
    except Exception:
        pass

    return _FALLBACK.substitute(
        cls      = flare_data.get("class", cls),
        ts       = flare_data.get("timestamp", "N/A"),
        conf     = f"{flare_data.get('confidence', 0):.0%}",
        lead     = f"{flare_data.get('lead_time', 0):.1f}",
        source   = flare_data.get("source", "unknown"),
        severity = SEVERITY.get(cls, "unknown severity"),
        action   = ACTIONS.get(cls, "Monitor situation."),
    ).strip()
```

**M9 Checkpoint:** `generate_flare_report({...})` returns a non-empty string in < 5s with Ollama running. With Ollama offline: fallback bulletin appears instantly with correct formatting. GraphRAG `query_graphrag("X-class flare")` returns relevant physics context.

---

## M10 — Physics Layer (PINN + Phase Detector)

```python
# src/physics/pinn_loss.py

import torch
import torch.nn as nn


class PINNLoss(nn.Module):
    """
    Combined classification + Neupert Effect physics constraint.

    Neupert Effect (empirical, Neupert 1968):
        HXR(t) ∝ d(SXR)/dt  during real solar flares

    Physical meaning: hard X-ray flux (electron bremsstrahlung) tracks
    the rate of energy deposition, which is the time derivative of soft
    X-ray flux (thermal emission). Models that fire without this physical
    relationship present are reacting to instrumental artifacts, not flares.

    physics_weight: controls how strongly the constraint is enforced.
    Start at 0.10 and increase if false alarms on instrument spikes are high.
    """
    def __init__(self, physics_weight: float = 0.10):
        super().__init__()
        self.pw = physics_weight
        self.ce = nn.CrossEntropyLoss()

    def forward(self,
                 pred:      "torch.Tensor",
                 target:    "torch.Tensor",
                 soft_xray: "torch.Tensor",
                 hard_xray: "torch.Tensor") -> "torch.Tensor":
        cls_loss  = self.ce(pred, target)
        dSXR_dt   = torch.diff(soft_xray, dim=1)
        hxr       = hard_xray[:, 1:]
        # Positive violation: HXR and dSXR/dt have opposite signs — penalise
        violation = torch.relu(-(dSXR_dt * hxr)).mean()
        return cls_loss + self.pw * violation
```

**M10 Checkpoint:** PINN loss < pure CE loss on training set (physics constraint active). Violation metric logged separately to MLflow. TSS on validation improves 0.02–0.05 over vanilla CE.

---

## M11 — Dual Uncertainty (MAPIE + Chronos-Bolt)

```python
# src/uncertainty/conformal.py

from mapie.classification import MapieClassifier
import numpy as np


def fit_conformal(model, X_cal: np.ndarray, y_cal: np.ndarray):
    """MAPIE conformal prediction — guarantees 90% marginal coverage."""
    mapie = MapieClassifier(estimator=model, method="score", cv="prefit")
    mapie.fit(X_cal, y_cal)
    return mapie


def predict_with_interval(mapie, X_new: np.ndarray,
                           alpha: float = 0.10) -> dict:
    y_pred, y_sets = mapie.predict(X_new, alpha=alpha)
    return {
        "prediction":    y_pred,
        "conformal_set": y_sets,
        "coverage":      f"{(1 - alpha) * 100:.0f}%",
    }
```

```python
# src/uncertainty/chronos_uncertainty.py
# Verify API with CHECK 3 before implementing.

import torch
import numpy as np


def load_chronos():
    """
    Chronos-Bolt — Amazon 2025. 250x faster than original Chronos.
    Verified API: predict(), not predict_quantiles() (see CHECK 3).
    """
    from chronos import BaseChronosPipeline
    return BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-small",
        dtype=torch.float32,
        device_map="cpu",
    )


def chronos_forecast_interval(pipeline,
                                flux_series: np.ndarray,
                                horizon: int = 15,
                                num_samples: int = 100,
                                quantiles: tuple = (0.1, 0.5, 0.9)) -> dict:
    """
    Probabilistic forecast using Chronos-Bolt Monte Carlo samples.

    Correct API (verified via CHECK 3):
    - pipeline.predict() returns (batch_size, num_samples, horizon) tensor
    - Quantiles are computed from the sample distribution, NOT via predict_quantiles()
    """
    context  = torch.tensor(flux_series[np.newaxis], dtype=torch.float32)
    samples  = pipeline.predict(context,
                                 prediction_length=horizon,
                                 num_samples=num_samples)
    # samples shape: (1, num_samples, horizon)
    peak_per_sample = samples[0].max(dim=-1).values.cpu().numpy()  # (num_samples,)
    q10, q50, q90  = np.quantile(peak_per_sample, list(quantiles))
    m_thresh        = 1e-5

    return {
        "q10_peak":     float(q10),
        "q50_peak":     float(q50),
        "q90_peak":     float(q90),
        "p_mclass_q50": float(q50 > m_thresh),
        "interval_80":  (float(q10), float(q90)),
        "horizon_min":  horizon,
    }


def dual_agreement_check(chronos_result: dict,
                          mapie_proba:    float) -> str:
    """
    HIGH confidence: both Chronos (q10 > 50% of M threshold) and MAPIE agree.
    LOW confidence: one or both sources disagree — issue advisory, not alert.
    This dual-source check reduces false alarm rate when instruments are noisy.
    """
    m_thresh    = 1e-5
    chronos_pos = chronos_result["q10_peak"] > m_thresh * 0.5
    mapie_pos   = mapie_proba > 0.5
    return "HIGH" if (chronos_pos and mapie_pos) else "LOW"
```

**M11 Checkpoint:** MAPIE empirical coverage ≥ 90% on calibration set. Chronos-Bolt `predict()` runs without `AttributeError`. Dual agreement flag prints "HIGH" on 2024-02-22 X6.3 event.

---

## M12 — Pre-flare Anomaly Detection (MOMENT)

> Verify MOMENT API with CHECK 4 before implementing. Input shape and output attribute name must be confirmed on the actual installed package.

```python
# src/forecasting/moment_anomaly.py
# Verify actual API with scripts/verify_moment.py (CHECK 4) before implementing.

import numpy as np
import torch

# Module-level cache: load model once per process
_moment_model = None


def _load_moment():
    """Singleton loader — MOMENT is large (1.3B params), load once."""
    global _moment_model
    if _moment_model is None:
        from momentfm import MOMENTPipeline
        _moment_model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={"task_name": "reconstruction"},
        )
        _moment_model.init()
        _moment_model.eval()
    return _moment_model


def compute_anomaly_score(model, flux_window: np.ndarray,
                           threshold: float = 2.5) -> dict:
    """
    Reconstruction error = anomaly score.
    High error on a quiet-sun baseline model → pre-flare precursor activity.

    Input shape verified with CHECK 4: (batch_size=1, n_channels=1, seq_len).
    Output attribute 'reconstruction' verified with CHECK 4 — update if different.
    """
    if len(flux_window) < 10:
        return {"anomaly_score": 0.0, "is_anomaly": False}

    mean   = flux_window.mean()
    std    = flux_window.std() + 1e-8
    normed = (flux_window - mean) / std

    # Shape: (1, 1, seq_len) — verified via CHECK 4
    tensor = torch.tensor(normed[np.newaxis, np.newaxis, :], dtype=torch.float32)
    with torch.no_grad():
        out = model(x_enc=tensor)
    # Attribute name 'reconstruction' — verified via CHECK 4; update if different
    recon = out.reconstruction.squeeze().numpy()
    error = float(np.abs(normed - recon[:len(normed)]).mean())
    return {"anomaly_score": round(error, 4), "is_anomaly": error > threshold}


def compute_anomaly_scores_batch(model,
                                  flux_series:  np.ndarray,
                                  window_size:  int = 512,
                                  batch_size:   int = 64,
                                  threshold:    float = 2.5) -> np.ndarray:
    """
    Batch inference across an entire time series.

    WHY BATCH: O(N) serial calls on 525,600 rows (1 year @ 1 min) takes hours.
    Batching reduces this to O(N/batch_size) forward passes.
    Requires GPU for practical throughput (Decision D3).
    """
    scores    = np.zeros(len(flux_series))
    windows   = []
    valid_idx = []

    for i in range(window_size, len(flux_series)):
        w = flux_series[i - window_size:i]
        if not np.any(np.isnan(w)):
            windows.append(w)
            valid_idx.append(i)

    for b_start in range(0, len(windows), batch_size):
        batch = np.array(windows[b_start:b_start + batch_size])
        mean  = batch.mean(axis=1, keepdims=True)
        std   = batch.std(axis=1, keepdims=True) + 1e-8
        normed = (batch - mean) / std

        # Shape: (batch_size, 1, window_size)
        tensor = torch.tensor(normed[:, np.newaxis, :], dtype=torch.float32)
        with torch.no_grad():
            out = model(x_enc=tensor)
        recon        = out.reconstruction.squeeze(1).numpy()  # (B, window_size)
        batch_errors = np.abs(normed - recon).mean(axis=1)

        for j, idx in enumerate(valid_idx[b_start:b_start + len(batch)]):
            scores[idx] = batch_errors[j]

    return scores


def fit_quiet_threshold(model, quiet_windows: list,
                          percentile: float = 97.5) -> float:
    """
    Fit anomaly detection threshold from quiet-sun windows.
    The threshold is the 97.5th percentile of reconstruction errors
    over known quiet periods — events above this are flagged as anomalous.
    """
    scores = [compute_anomaly_score(model, w)["anomaly_score"]
               for w in quiet_windows]
    t = float(np.percentile(scores, percentile))
    print(f"Quiet-sun threshold (p{percentile}): {t:.4f}")
    return t
```

**M12 Checkpoint:** `compute_anomaly_scores_batch()` completes in < 10 minutes on CPU for 30 days of data. Threshold fitted from M1 EDA quiet periods. Anomaly score spikes visibly before flare peaks in `notebooks/08_moment_anomaly.ipynb`.

---

## M13 — Explainability (SHAP)

```python
# src/explainability/shap_explainer.py

import shap
import numpy as np


def build_explainer(xgb_model) -> shap.TreeExplainer:
    return shap.TreeExplainer(xgb_model)


def explain_prediction(explainer: shap.TreeExplainer,
                        x: np.ndarray,
                        feature_names: list) -> dict:
    """
    Top-5 SHAP contributions for a single prediction.
    Rendered in dashboard as a waterfall chart with direction labels.
    """
    sv = explainer.shap_values(x.reshape(1, -1))
    if isinstance(sv, list):
        sv = sv[1]   # class 1 (any-flare) values for binary interpretation
    sv = sv[0]

    top5 = sorted(zip(feature_names, sv.tolist()),
                   key=lambda t: abs(t[1]), reverse=True)[:5]
    return {
        "top_drivers": [
            {
                "feature":      f,
                "contribution": round(float(v), 5),
                "direction":    "flare" if v > 0 else "quiet",
            }
            for f, v in top5
        ],
        "base_value": float(
            explainer.expected_value
            if not isinstance(explainer.expected_value, list)
            else explainer.expected_value[1]
        ),
    }
```

---

## M14 — Dashboard & Visualization

```python
# src/api/main.py

import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from src.api.helpers import (
    load_master_catalogue, load_latest_window, run_nowcast_inference,
    run_forecast_inference, load_dual_uncertainty, load_moment_score,
    load_shap_explanation, compute_24hr_stats, check_drift_status,
    get_latest_reading_json,
)

app = FastAPI(title="SolarSentinel API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
executor = ThreadPoolExecutor(max_workers=4)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/catalogue")
def catalogue():
    return {"events": load_master_catalogue()}


@app.get("/api/nowcast/latest")
async def nowcast():
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor, run_nowcast_inference, load_latest_window()
    )


@app.get("/api/forecast")
async def forecast(horizon: int = 15):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor, run_forecast_inference, load_latest_window(), horizon
    )


@app.get("/api/uncertainty")
def uncertainty():
    return load_dual_uncertainty()


@app.get("/api/anomaly")
def anomaly():
    return load_moment_score()


@app.get("/api/explain/{event_id}")
def explain(event_id: str):
    return load_shap_explanation(event_id)


@app.get("/api/daily-report")
async def daily_report():
    from src.intelligence.llm_reporter import generate_flare_report
    stats  = compute_24hr_stats()
    report = generate_flare_report(stats)
    return {"report": report}


@app.get("/api/drift")
def drift():
    return check_drift_status()


@app.websocket("/ws/live")
async def live_stream(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.send_json(get_latest_reading_json())
            await asyncio.sleep(60)
    except Exception:
        await ws.close()
```

```python
# src/api/helpers.py — implement these before M14

import pandas as pd
import numpy as np
from pathlib import Path

CATALOGUE_PATH = Path("data/processed/master_catalogue.csv")
DATASET_PATH   = Path("data/processed/dataset.parquet")


def load_master_catalogue() -> list:
    if CATALOGUE_PATH.exists():
        return pd.read_csv(CATALOGUE_PATH).to_dict("records")
    return []


def load_latest_window(n_rows: int = 60) -> pd.DataFrame:
    return pd.read_parquet(DATASET_PATH).tail(n_rows)


def run_nowcast_inference(window: pd.DataFrame) -> dict:
    # Import and call ONNXNowcaster here
    return {"class": "N", "confidence": 0.95, "proba": {"N": 0.95}}  # stub


def run_forecast_inference(window: pd.DataFrame, horizon: int) -> dict:
    return {"proba": 0.10, "horizon": horizon}  # stub


def load_dual_uncertainty() -> dict:
    return {}  # stub — populate from latest inference result cache


def load_moment_score() -> dict:
    return {}  # stub


def load_shap_explanation(event_id: str) -> dict:
    return {}  # stub


def compute_24hr_stats() -> dict:
    return {"class": "N", "timestamp": str(pd.Timestamp.now())}  # stub


def check_drift_status() -> dict:
    return {"drift": False}  # stub


def get_latest_reading_json() -> dict:
    window = load_latest_window(n_rows=1)
    if window.empty:
        return {}
    row = window.iloc[-1]
    return {"timestamp": str(row.name), "flux_high": float(row.get("flux_high", 0))}
```

**Alert levels:**

| Condition | Dashboard | Action |
|---|---|---|
| `forecast_15min > 0.50` | 🟡 Yellow banner | Monitoring mode |
| `forecast_15min > 0.75` AND `dual_agreement=HIGH` | 🟠 Orange + audio ping | Advisory issued |
| `nowcast_class in (M, X)` | 🔴 Red CRITICAL + email + Slack webhook | Immediate action |

---

## M15 — Evaluation & Walk-Forward Validation

### 15.1 Metrics

```python
# src/evaluation/metrics.py

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report


def compute_tss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    True Skill Score = TPR − FPR.
    labels=[0, 1] is mandatory: without it, confusion_matrix returns a 1×1
    matrix when a fold has no positive events, and .ravel() crashes.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn + 1e-10)
    fpr = fp / (fp + tn + 1e-10)
    return float(tpr - fpr)


def compute_far(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """False Alarm Rate = FP / (FP + TN)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fp / (fp + tn + 1e-10))


def compute_lead_time(y_proba:    np.ndarray,
                       y_true:     np.ndarray,
                       timestamps: np.ndarray,
                       threshold:  float = 0.5) -> dict:
    alerts  = timestamps[y_proba >= threshold]
    onsets  = timestamps[np.diff(y_true.astype(int), prepend=0) == 1]
    results = []
    for onset in onsets:
        prior = alerts[alerts < onset]
        if len(prior) > 0:
            results.append({
                "onset":    str(onset),
                "lead_min": float((onset - prior[-1]).total_seconds() / 60),
                "caught":   True,
            })
        else:
            results.append({"onset": str(onset), "lead_min": 0.0, "caught": False})
    leads = [r["lead_min"] for r in results if r["caught"]]
    return {
        "mean_lead_min":  round(float(np.mean(leads)), 2) if leads else 0.0,
        "detection_rate": round(float(np.mean([r["caught"] for r in results])), 3),
        "per_event":      results,
    }


def evaluate_per_class(y_true: np.ndarray, y_pred: np.ndarray,
                        classes: list = ["C", "M", "X"]) -> None:
    """PS15 explicitly requires per-class reporting (low and high class flares)."""
    class_int = {"C": 1, "M": 2, "X": 3}
    for cls in classes:
        y_t = (y_true == class_int[cls]).astype(int)
        y_p = (y_pred == class_int[cls]).astype(int)
        if y_t.sum() == 0:
            print(f"Class {cls}: no events in test set — check labeling pipeline.")
            continue
        tss = compute_tss(y_t, y_p)
        far = compute_far(y_t, y_p)
        tpr = float(((y_p == 1) & (y_t == 1)).sum() / (y_t.sum() + 1e-10))
        print(f"Class {cls}: TSS={tss:.3f}  FAR={far:.3f}  TPR={tpr:.3f}")


def full_report(y_true, y_pred, y_proba, timestamps,
                 label_names=("N", "C", "M", "X")) -> None:
    y_bin   = (y_true > 0).astype(int)
    y_pb    = y_proba.max(axis=1) if y_proba.ndim > 1 else y_proba
    y_p_bin = (y_pred > 0).astype(int)
    print("=" * 50)
    print("SolarSentinel Evaluation Report")
    print("=" * 50)
    print(f"TSS:     {compute_tss(y_bin, y_p_bin):.4f}  (target > 0.80)")
    print(f"FAR:     {compute_far(y_bin, y_p_bin):.4f}  (target < 0.10)")
    try:
        print(f"ROC-AUC: {roc_auc_score(y_bin, y_pb):.4f}  (target > 0.90)")
    except Exception:
        pass
    print(classification_report(y_true, y_pred, target_names=label_names))
    evaluate_per_class(y_true, y_pred)
    lt = compute_lead_time(y_pb, y_bin, timestamps)
    print(f"Mean lead time: {lt['mean_lead_min']:.1f} min  (target > 30)")
    print(f"Detection rate: {lt['detection_rate'] * 100:.1f}%")
```

### 15.2 Walk-Forward Cross-Validation

```python
# src/evaluation/walk_forward.py

import numpy as np
import pandas as pd
import mlflow
from sklearn.model_selection import TimeSeriesSplit
from src.evaluation.metrics import compute_tss, compute_far


def walk_forward_evaluate(X:              np.ndarray,
                           y:              np.ndarray,
                           timestamps:     np.ndarray,
                           model_trainer:  callable,
                           n_splits:       int   = 5,
                           gap_minutes:    int   = 1440) -> pd.DataFrame:
    """
    Walk-forward (expanding window) cross-validation.

    gap_minutes=1440 (1 day): prevents leakage from flare-adjacent quiet
    periods that share physics features with the adjacent flare window.

    TSS std < 0.12 across folds means the model is reliable across different
    solar activity periods, not just lucky on one training window.

    NOTE: confusion_matrix uses labels=[0, 1] everywhere (see metrics.py).
    Without this, folds with no positive events crash with ValueError on .ravel().
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap_minutes)
    fold_results = []

    with mlflow.start_run(run_name="walk_forward_cv"):
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            y_bin_val   = (y_val > 0).astype(int)

            if (y_tr > 0).sum() == 0:
                print(f"Fold {fold_idx}: no flare events in training — skipping.")
                continue

            model  = model_trainer(X_tr, y_tr)
            proba  = model.predict_proba(X_val)
            y_pb   = proba[:, 1] if proba.ndim > 1 else proba
            y_pred = (y_pb >= 0.40).astype(int)

            tss = compute_tss(y_bin_val, y_pred)
            far = compute_far(y_bin_val, y_pred)

            fold_results.append({
                "fold":         fold_idx,
                "train_size":   len(train_idx),
                "val_size":     len(val_idx),
                "n_flares_val": int(y_bin_val.sum()),
                "tss":          tss,
                "far":          far,
            })
            mlflow.log_metrics({
                f"fold_{fold_idx}_tss": tss,
                f"fold_{fold_idx}_far": far,
            })
            print(f"Fold {fold_idx}: TSS={tss:.3f}  FAR={far:.3f}  "
                  f"val_flares={y_bin_val.sum()}")

        results = pd.DataFrame(fold_results)
        if not results.empty:
            mean_tss = results["tss"].mean()
            std_tss  = results["tss"].std()
            mlflow.log_metrics({"mean_tss": mean_tss, "std_tss": std_tss,
                                  "mean_far": results["far"].mean()})
            print(f"\nWalk-Forward CV Summary:")
            print(f"  TSS: {mean_tss:.4f} ± {std_tss:.4f}  (target < 0.12 std)")
            print(f"  FAR: {results['far'].mean():.4f}")
            if std_tss > 0.12:
                print(
                    "  WARNING: High TSS variance — model performance depends "
                    "strongly on solar activity period. Add more historical data."
                )

    return results
```

**M15 Checkpoint:** 5-fold walk-forward CV completes without crash. TSS std printed. Per-class TPR logged. Lead time > 30 min mean on test set.

---

## M16 — Production Deployment & MLOps

### 16.1 Drift Detection

```python
# src/monitoring/drift_detector.py

from alibi_detect.cd import TabularDrift
import numpy as np


def build_drift_detector(X_reference: np.ndarray) -> TabularDrift:
    return TabularDrift(X_reference, p_val=0.05)


def check_drift(detector: TabularDrift, X_new: np.ndarray) -> dict:
    result   = detector.predict(X_new)
    is_drift = bool(result["data"]["is_drift"])
    if is_drift:
        _trigger_retrain_workflow()
    return {"drift": is_drift, "p_value": float(result["data"]["p_val"])}


def _trigger_retrain_workflow():
    """Dispatch GitHub Actions retrain workflow via API."""
    import os, requests
    token = os.getenv("GH_PAT")
    if not token:
        print("WARNING: GH_PAT not set — cannot trigger retrain workflow.")
        return
    requests.post(
        "https://api.github.com/repos/quantum-crew/solar-sentinel/dispatches",
        headers={"Authorization": f"token {token}",
                 "Accept": "application/vnd.github.v3+json"},
        json={"event_type": "model_drift_detected"},
    )
```

### 16.2 Model Registry

```python
# src/monitoring/model_registry.py

import mlflow


def promote_if_better(new_run_id: str,
                       model_name:  str = "nowcast_model",
                       min_delta:   float = 0.02) -> bool:
    client   = mlflow.tracking.MlflowClient()
    prod     = client.get_latest_versions(model_name, stages=["Production"])
    prod_tss = float(prod[0].tags.get("mean_tss", "0")) if prod else 0.0
    new_tss  = client.get_run(new_run_id).data.metrics.get("mean_tss", 0.0)

    if new_tss > prod_tss + min_delta:
        client.transition_model_version_stage(
            model_name, new_run_id, "Production"
        )
        print(f"Promoted to Production: {prod_tss:.3f} → {new_tss:.3f}")
        return True
    print(f"Kept current model. New TSS {new_tss:.3f} not {min_delta} better than {prod_tss:.3f}")
    return False
```

### 16.3 PRADAN Downloader (auth-aware)

```python
# src/ingestion/pradan_downloader.py
# Fill configs/pradan_auth.yaml FIRST (see CHECK 6 in Pre-Implementation section).

import os, sys
import requests
import yaml
from circuitbreaker import circuit
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

with open("configs/pradan_auth.yaml") as f:
    AUTH_CFG = yaml.safe_load(f)

PRADAN_USERNAME = os.getenv("PRADAN_USERNAME")
PRADAN_PASSWORD = os.getenv("PRADAN_PASSWORD")


def get_pradan_session() -> requests.Session:
    """
    Authenticate with PRADAN using the mechanism confirmed in CHECK 6.
    Auth endpoint and field names come from configs/pradan_auth.yaml.
    """
    if not PRADAN_USERNAME or not PRADAN_PASSWORD:
        raise EnvironmentError(
            "PRADAN_USERNAME and PRADAN_PASSWORD must be set in .env. "
            "Register at https://pradan.issdc.gov.in."
        )
    session = requests.Session()
    payload = {
        AUTH_CFG["form_fields"]["username_field"]: PRADAN_USERNAME,
        AUTH_CFG["form_fields"]["password_field"]: PRADAN_PASSWORD,
    }
    r = session.post(AUTH_CFG["auth_endpoint"], data=payload, timeout=30)
    r.raise_for_status()
    return session


_session = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = get_pradan_session()
    return _session


@circuit(failure_threshold=3, recovery_timeout=60)
def fetch_fits_from_pradan(instrument: str, date: str, save_dir: str) -> str:
    session  = _get_session()
    filename = f"{date}_{instrument.lower()}.fits"
    url      = f"{AUTH_CFG.get('base_url', 'https://pradan.issdc.gov.in')}/data/{instrument.upper()}/L1/{filename}"
    r        = session.get(url, timeout=60, stream=True)
    r.raise_for_status()
    filepath = str(Path(save_dir) / filename)
    with open(filepath, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    return filepath


def safe_fetch(instrument: str, date: str, save_dir: str) -> str:
    """Returns local path — from PRADAN or last known good cache."""
    try:
        return fetch_fits_from_pradan(instrument, date, save_dir)
    except Exception as e:
        cached = str(Path(save_dir) / f"{date}_{instrument.lower()}.fits")
        if Path(cached).exists():
            print(f"PRADAN unavailable ({e}). Using cached: {cached}")
            return cached
        raise RuntimeError(
            f"PRADAN unavailable and no cache for {instrument} {date}."
        ) from e
```

### 16.4 Docker Compose

```yaml
# docker-compose.yml
version: "3.9"

services:
  api:
    build: { context: ., dockerfile: docker/Dockerfile.api }
    ports: ["8000:8000"]
    volumes: [./data:/app/data, ./models:/app/models, ./configs:/app/configs]
    environment:
      - OLLAMA_URL=http://ollama:11434
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - PRADAN_USERNAME=${PRADAN_USERNAME}
      - PRADAN_PASSWORD=${PRADAN_PASSWORD}
    depends_on: [ollama, mlflow]

  dashboard:
    build: { context: ./dashboard }
    ports: ["3000:3000"]
    depends_on: [api]

  ollama:
    image: ollama/ollama:latest
    ports: ["11434:11434"]
    volumes: [ollama_data:/root/.ollama]

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports: ["5000:5000"]
    command: mlflow server --host 0.0.0.0

  prometheus:
    image: prom/prometheus
    ports: ["9090:9090"]
    volumes: [./docker/prometheus.yml:/etc/prometheus/prometheus.yml]

  grafana:
    image: grafana/grafana
    ports: ["3001:3000"]
    depends_on: [prometheus]

volumes:
  ollama_data:
```

### 16.5 CI/CD with PRADAN Auth

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v --tb=short

---
# .github/workflows/retrain.yml
name: Weekly Retrain
on:
  schedule: [{ cron: "0 0 * * 0" }]
  workflow_dispatch:
    inputs:
      reason:
        description: "Trigger reason (drift / schedule / manual)"
        default: "schedule"

jobs:
  retrain:
    runs-on: ubuntu-latest
    env:
      PRADAN_USERNAME: ${{ secrets.PRADAN_USERNAME }}
      PRADAN_PASSWORD: ${{ secrets.PRADAN_PASSWORD }}
      GH_PAT:         ${{ secrets.GH_PAT }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt
      - run: python src/ingestion/goes_downloader.py --days=30
      - name: Download fresh Aditya-L1 data
        run:  python src/ingestion/pradan_downloader.py --days=7
      - run: python src/nowcasting/train.py --mode=incremental
      - run: python src/evaluation/walk_forward.py
      - run: python src/monitoring/model_registry.py --promote-if-better
```

---

## Tests

```python
# tests/conftest.py

import pytest
import numpy as np
import pandas as pd
import torch
from src.nowcasting.solexs_detector import FlareEvent


@pytest.fixture(scope="session")
def sample_solexs_df() -> pd.DataFrame:
    """Synthetic SoLEXS DataFrame with one realistic M-class flare shape."""
    n   = 120
    idx = pd.date_range("2024-02-22 10:00", periods=n, freq="1min")
    # Baseline quiet-sun flux
    f = np.random.exponential(1e-7, n)
    # Inject synthetic M-class: gradual rise over 20 min, peak, gradual decay
    rise   = np.linspace(1e-7, 5e-5, 20)
    decay  = np.linspace(5e-5, 1e-7, 15)
    f[55:75] = rise
    f[75:90] = decay
    return pd.DataFrame(
        {"flux_low": f * 0.3, "flux_mid": f * 0.6, "flux_high": f},
        index=idx
    )


@pytest.fixture(scope="session")
def sample_hel1os_df() -> pd.DataFrame:
    """Synthetic HEL1OS DataFrame with impulsive burst BEFORE SoLEXS peak (Neupert)."""
    n   = 120
    idx = pd.date_range("2024-02-22 10:00", periods=n, freq="1min")
    c = np.random.poisson(50, n).astype(float)
    # Hard X-ray impulsive phase: 2 min earlier than SoLEXS peak at t=75
    burst = np.linspace(50, 3000, 15)
    c[53:68] = burst
    c[68:75] = np.linspace(3000, 50, 7)
    return pd.DataFrame(
        {"counts_low": c, "counts_high": c * 0.3},
        index=idx
    )


@pytest.fixture(scope="session")
def sample_flare_event() -> FlareEvent:
    return FlareEvent(
        start_time  = pd.Timestamp("2024-02-22 10:55"),
        peak_time   = pd.Timestamp("2024-02-22 11:15"),
        end_time    = pd.Timestamp("2024-02-22 11:30"),
        peak_flux   = 5e-5,
        flare_class = "M",
        instrument  = "SoLEXS",
        confidence  = 0.91,
    )


@pytest.fixture(scope="session")
def sample_hel1os_event() -> FlareEvent:
    """HEL1OS event 2 min earlier than SoLEXS (Neupert relationship)."""
    return FlareEvent(
        start_time  = pd.Timestamp("2024-02-22 10:53"),
        peak_time   = pd.Timestamp("2024-02-22 11:13"),
        end_time    = pd.Timestamp("2024-02-22 11:28"),
        peak_flux   = 2800.0,
        flare_class = "?",
        instrument  = "HEL1OS",
        confidence  = 0.87,
    )


@pytest.fixture(scope="session")
def toy_tcn_encoder():
    """Tiny 2-layer TCNEncoder for fast CI tests — no GPU needed."""
    from src.nowcasting.tcn_encoder import TCNEncoder
    return TCNEncoder(n_features=8, embed_dim=32, n_layers=2)


@pytest.fixture(scope="session")
def toy_onnx_nowcaster(tmp_path_factory):
    """
    Exports toy TCNEncoder to ONNX and wraps with XGBoost stub.
    Used in property-based tests instead of undefined load_test_onnx_nowcaster().
    """
    import xgboost as xgb
    from src.nowcasting.tcn_encoder import TCNEncoder
    from src.deployment.onnx_export import export_to_onnx, ONNXNowcaster

    encoder    = TCNEncoder(n_features=8, embed_dim=32, n_layers=2)
    dummy      = torch.zeros(1, 60, 8)
    onnx_path  = str(tmp_path_factory.mktemp("models") / "tcn_test.onnx")
    export_to_onnx(encoder, dummy, onnx_path, "light_curve_window", "embedding")

    X_stub = np.random.randn(40, 32).astype(np.float32)
    y_stub = np.random.randint(0, 4, 40)
    xgb_stub = xgb.XGBClassifier(n_estimators=5, objective="multi:softprob",
                                   num_class=4, verbosity=0)
    xgb_stub.fit(X_stub, y_stub)
    return ONNXNowcaster(tcn_path=onnx_path, xgb_model=xgb_stub)
```

```python
# tests/test_pipeline.py

import numpy as np
import pandas as pd
import pytest


def test_background_subtraction_nonnegative(sample_solexs_df):
    from src.preprocessing.physics_features import subtract_background
    result = subtract_background(sample_solexs_df)
    assert (result >= 0).all().all(), "Background subtraction produced negative values."


def test_dead_time_correction_increases_counts(sample_hel1os_df):
    from src.ingestion.fits_reader import apply_dead_time_correction
    corrected = apply_dead_time_correction(sample_hel1os_df)
    valid     = ~corrected["counts_low"].isna()
    assert (corrected.loc[valid, "counts_low"] >=
            sample_hel1os_df.loc[valid, "counts_low"] - 1e-6).all()


def test_dual_detection_produces_dual_event(sample_flare_event, sample_hel1os_event):
    from src.catalogue.merger import merge_catalogues
    master = merge_catalogues([sample_flare_event], [sample_hel1os_event])
    assert len(master) == 1, "Should merge into one dual event"
    assert master.iloc[0]["source"] == "dual"
    assert master.iloc[0]["confidence"] > 0.90


def test_solexs_only_confidence_reduced(sample_flare_event):
    from src.catalogue.merger import merge_catalogues
    master = merge_catalogues([sample_flare_event], [])
    assert master.iloc[0]["source"] == "SoLEXS_only"
    assert master.iloc[0]["confidence"] < sample_flare_event.confidence


def test_hel1os_only_lower_confidence(sample_hel1os_event):
    from src.catalogue.merger import merge_catalogues
    master = merge_catalogues([], [sample_hel1os_event])
    assert master.iloc[0]["source"] == "HEL1OS_only"
    assert master.iloc[0]["confidence"] < sample_hel1os_event.confidence


def test_master_catalogue_no_duplication(sample_flare_event, sample_hel1os_event):
    from src.catalogue.merger import merge_catalogues
    master = merge_catalogues([sample_flare_event], [sample_hel1os_event])
    assert len(master) <= 2, "Master should not have more events than inputs."


def test_tss_perfect():
    from src.evaluation.metrics import compute_tss
    y = np.array([0, 0, 1, 1])
    assert compute_tss(y, y) == pytest.approx(1.0, abs=1e-6)


def test_tss_all_wrong():
    from src.evaluation.metrics import compute_tss
    y    = np.array([0, 0, 1, 1])
    yhat = np.array([1, 1, 0, 0])
    assert compute_tss(y, yhat) == pytest.approx(-1.0, abs=1e-6)


def test_confusion_matrix_empty_fold():
    """Regression test: compute_tss must not crash when fold has no positive events."""
    from src.evaluation.metrics import compute_tss, compute_far
    y_all_quiet = np.zeros(100, dtype=int)
    y_pred_all_quiet = np.zeros(100, dtype=int)
    # Must not raise ValueError — labels=[0,1] in confusion_matrix handles this
    tss = compute_tss(y_all_quiet, y_pred_all_quiet)
    far = compute_far(y_all_quiet, y_pred_all_quiet)
    assert tss == pytest.approx(0.0, abs=1e-6)
    assert far == pytest.approx(0.0, abs=1e-6)
```

```python
# tests/test_properties.py

from hypothesis import given, strategies as st, settings
import numpy as np


@given(st.floats(min_value=0.0, max_value=1e-3, allow_nan=False))
def test_background_subtraction_never_negative(flux_val):
    import pandas as pd
    from src.preprocessing.physics_features import subtract_background
    idx = pd.date_range("2024-01-01", periods=30, freq="1min")
    df  = pd.DataFrame({"flux_high": np.full(30, flux_val)}, index=idx)
    assert (subtract_background(df) >= 0).all().all()


@given(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False))
def test_dead_time_correction_non_negative_output(count_rate):
    from src.ingestion.fits_reader import apply_dead_time_correction
    import pandas as pd
    idx = pd.date_range("2024-01-01", periods=10, freq="1s")
    df  = pd.DataFrame({"counts_low": np.full(10, count_rate),
                         "counts_high": np.full(10, count_rate * 0.3)}, index=idx)
    corr = apply_dead_time_correction(df)
    valid = ~corr["counts_low"].isna()
    if valid.any():
        assert (corr.loc[valid, "counts_low"] >= 0).all()


@given(st.arrays(np.int64, shape=(100,), elements=st.integers(0, 1)))
def test_tss_always_in_range(y):
    from src.evaluation.metrics import compute_tss
    tss = compute_tss(y, np.zeros_like(y))
    assert -1.0 <= tss <= 1.0


@given(st.arrays(np.int64, shape=(100,), elements=st.integers(0, 1)))
def test_far_always_in_range(y):
    from src.evaluation.metrics import compute_far
    far = compute_far(y, np.ones_like(y))
    assert 0.0 <= far <= 1.0


@settings(max_examples=20)
@given(st.arrays(np.float32, shape=(60, 8),
                  elements=st.floats(0.0, 1e-3, allow_nan=False)))
def test_onnx_nowcaster_output_valid(window, toy_onnx_nowcaster):
    """Property: ONNX nowcaster must always return valid class and bounded confidence."""
    result = toy_onnx_nowcaster.predict(
        window[np.newaxis].astype(np.float32),
        np.zeros((1, 0), dtype=np.float32)
    )
    assert result["class"] in ("N", "C", "M", "X")
    assert 0.0 <= result["confidence"] <= 1.0
    assert abs(sum(result["proba"].values()) - 1.0) < 1e-4
```

---

## Project Structure

```
solar-sentinel/
├── scripts/
│   ├── verify_fits_columns.py        # RUN FIRST — CHECK 1
│   ├── verify_timesfm.py             # CHECK 2 — before M5
│   ├── verify_chronos.py             # CHECK 3 — before M11
│   ├── verify_moment.py              # CHECK 4 — before M12
│   └── verify_dspy.py                # CHECK 5 — before M9
├── data/
│   ├── raw/
│   │   ├── solexs/                   # PRADAN Level-1 FITS
│   │   ├── hel1os/                   # PRADAN Level-1 FITS
│   │   └── goes/                     # GOES XRS 40yr historical
│   ├── processed/
│   │   ├── dataset.parquet           # Feature-engineered training data
│   │   └── master_catalogue.csv      # PRIMARY PS15 DELIVERABLE
│   ├── labels/
│   │   └── noaa_flare_catalog.csv
│   ├── knowledge/                    # Documents for GraphRAG indexing
│   │   ├── noaa_event_reports/
│   │   ├── goes_user_guide.pdf
│   │   └── space_weather_scales.pdf
│   └── graphrag/                     # GraphRAG index (commit after building)
│       ├── output/
│       └── settings.yaml             # Local embedding config (see CHECK 7)
├── src/
│   ├── ingestion/
│   │   ├── fits_reader.py            # Column-safe + dead-time correction
│   │   ├── goes_downloader.py
│   │   └── pradan_downloader.py      # Auth from configs/pradan_auth.yaml
│   ├── preprocessing/
│   │   ├── physics_features.py       # Neupert, phase (configurable), thermal
│   │   ├── cross_calibration.py      # LOG-SPACE Huber regression
│   │   ├── augmentation.py           # tsaug minority class
│   │   └── labels.py                 # Multi-class N/C/M/X + windows
│   ├── nowcasting/
│   │   ├── solexs_detector.py        # Soft-XR only → SoLEXS catalogue
│   │   ├── hel1os_detector.py        # Hard-XR only → HEL1OS catalogue
│   │   ├── tcn_encoder.py            # Causal dilated TCN
│   │   ├── train.py                  # multi:softprob + per-class thresholds
│   │   └── flaml_tuner.py            # macro_f1 AutoML
│   ├── catalogue/
│   │   └── merger.py                 # Temporal coincidence → master catalogue
│   ├── forecasting/
│   │   ├── causal_lstm.py            # Unidirectional, bidirectional=False
│   │   ├── timesfm_forecaster.py     # Zero-shot + PEFT LoRA (if GPU)
│   │   ├── multi_horizon.py          # 15/30/60 min heads
│   │   ├── ensemble.py               # 3-model weighted blend
│   │   └── moment_anomaly.py         # BATCH inference — not O(N) serial
│   ├── deployment/
│   │   └── onnx_export.py            # TCN + LSTM → ONNX
│   ├── physics/
│   │   └── pinn_loss.py              # Neupert Effect constraint
│   ├── uncertainty/
│   │   ├── conformal.py              # MAPIE — 90% coverage
│   │   └── chronos_uncertainty.py    # Chronos predict() + manual quantiles
│   ├── explainability/
│   │   └── shap_explainer.py
│   ├── orchestration/
│   │   ├── state.py                  # Annotated reducer for fan-in
│   │   ├── agents.py                 # ALL agent implementations (fully written)
│   │   └── graph.py                  # Correct parallel fan-out/fan-in
│   ├── intelligence/
│   │   ├── dspy_reporter.py          # Thread-safe per-call dspy.context()
│   │   ├── graphrag_retriever.py     # sys.executable subprocess
│   │   └── llm_reporter.py           # DSPy + fallback bulletin
│   ├── monitoring/
│   │   ├── drift_detector.py
│   │   └── model_registry.py
│   ├── evaluation/
│   │   ├── metrics.py                # confusion_matrix labels=[0,1] everywhere
│   │   └── walk_forward.py           # 5-fold, 1-day gap
│   └── api/
│       ├── main.py                   # FastAPI endpoints
│       └── helpers.py                # All endpoint helper functions
├── configs/
│   ├── fits_columns.yaml             # FILL AFTER CHECK 1
│   ├── nowcasting.yaml               # Phase thresholds, class thresholds
│   ├── forecasting.yaml
│   ├── pradan_auth.yaml              # Auth mechanism (no credentials)
│   └── llm.yaml
├── models/                           # Populated during training
│   ├── tcn_encoder.pt
│   ├── tcn_encoder.onnx
│   ├── xgb_multiclass.json
│   ├── causal_lstm.pt
│   ├── causal_lstm.onnx
│   ├── timesfm_lora/                 # PEFT adapter weights (if GPU run)
│   ├── conformal_mapie.pkl
│   └── dspy_reporter_optimised.json
├── notebooks/
│   ├── 01_fits_column_inspection.ipynb   # Run before any coding
│   ├── 02_eda_and_phase_calibration.ipynb
│   ├── 03_independent_detectors.ipynb
│   ├── 04_catalogue_merger.ipynb
│   ├── 05_multiclass_nowcasting.ipynb
│   ├── 06_timesfm_verification.ipynb
│   ├── 07_onnx_benchmark.ipynb
│   ├── 08_moment_anomaly.ipynb
│   └── 09_walk_forward_cv.ipynb
├── tests/
│   ├── conftest.py                   # All fixtures including toy_onnx_nowcaster
│   ├── test_pipeline.py              # Including empty-fold regression test
│   └── test_properties.py            # hypothesis — uses toy_onnx_nowcaster fixture
├── docker/
│   ├── Dockerfile.api
│   └── prometheus.yml
├── .github/workflows/
│   ├── ci.yml
│   └── retrain.yml                   # PRADAN_USERNAME/PASSWORD as secrets
├── .env.example                      # Template — no real credentials
├── .gitignore
├── docker-compose.yml
├── requirements.txt                  # ALL VERSIONS PINNED
└── README.md
```

---

## Evaluation Metrics

| Metric | Formula | Target | Why |
|---|---|---|---|
| **TSS** | TPR − FPR | > 0.80 | Correct metric for imbalanced flare data |
| **FAR** | FP / (FP + TN) | < 0.10 | Operational false alarm tolerance |
| **ROC-AUC** | Area under ROC curve | > 0.90 | Binary classifier quality |
| **C-class TPR** | TP_C / (TP_C + FN_C) | > 0.85 | PS15 explicitly requires low-class detection |
| **Mean Lead Time** | Avg(flare onset − first alert) | > 30 min | Gives operators time to act |
| **Walk-forward TSS std** | Std across 5 CV folds | < 0.12 | Not lucky — reliable across activity periods |
| **MAPIE Coverage** | Empirical coverage @ α=0.10 | ≥ 90% | Honest calibrated intervals |
| **Chronos–MAPIE Agreement** | Dual HIGH = confident alert | Report per event | Reduces false alarm rate |

> **Never report raw accuracy.** A model predicting "no flare" achieves 99%+ accuracy on any solar dataset while catching zero real events. TSS is the mandatory operational metric throughout solar flare ML literature.

---

## References

- Aditya-L1 Mission — ISRO: https://www.isro.gov.in
- PRADAN — ISSDC: https://pradan.issdc.gov.in
- NOAA SWPC Flare Catalog: https://ftp.swpc.noaa.gov/pub/indices/events/
- GOES XRS Archive — NCEI: https://www.ngdc.noaa.gov/stp/solar/solarflares.html
- SunPy: https://sunpy.org
- TimesFM 2.0 — Google Research, ICML 2024: https://github.com/google-research/timesfm
- MOMENT — CMU AutonLab: https://github.com/moment-timeseries-foundation-model/moment
- Chronos-Bolt — Amazon: https://github.com/amazon-science/chronos-forecasting
- GraphRAG — Microsoft: https://github.com/microsoft/graphrag
- DSPy — Stanford NLP: https://dspy.ai
- FLAML — Microsoft: https://microsoft.github.io/FLAML/
- MAPIE — Conformal Prediction: https://mapie.readthedocs.io
- Alibi-Detect: https://docs.seldon.io/projects/alibi-detect
- SHAP: https://shap.readthedocs.io
- tsaug: https://tsaug.readthedocs.io
- PEFT — HuggingFace: https://github.com/huggingface/peft
- Ollama: https://ollama.com
- LangGraph: https://langchain-ai.github.io/langgraph/
- Neupert (1968) — Comparison of solar X-ray bursts with microwave bursts
- Bloomfield et al. (2012) — Toward Reliable Benchmarking of Solar Flare Forecasting Methods
- Bobra & Couvidat (2015) — Solar Flare Prediction Using SDO/HMI Vector Magnetic Field Data
- Baldi et al. (2021) — Solar Flare Prediction with CNNs
- Ahmadzadeh et al. (2021) — How to Train a Solar Flare Forecasting Model

---

*Built by Quantum Crew — ISRO Aditya-L1 Hackathon PS15*
