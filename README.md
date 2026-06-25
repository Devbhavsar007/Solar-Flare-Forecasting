# ☀️ SolarSentinel — Solar Flare Nowcasting & Forecasting System
### ISRO Aditya-L1 | SoLEXS + HEL1OS | PS15 | Quantum Crew

> Automated detection, classification, and prediction of solar flares using combined Soft and Hard X-ray time-series data from ISRO's Aditya-L1 mission — with physics-informed ML, LangGraph orchestration, RAG-grounded LLM alerts, SHAP explainability, conformal uncertainty, and full MLOps infrastructure.

[![CI](https://github.com/quantum-crew/solar-sentinel/actions/workflows/ci.yml/badge.svg)](https://github.com/quantum-crew/solar-sentinel/actions)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-blue)](http://localhost:5000)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Accuracy Strategy](#accuracy-strategy)
- [Milestone Map](#milestone-map)
- [M0 — Environment & Setup](#m0--environment--setup)
- [M1 — Data Pipeline](#m1--data-pipeline)
- [M2 — Nowcasting Pipeline (TCN + XGBoost)](#m2--nowcasting-pipeline-tcn--xgboost)
- [M3 — Forecasting Pipeline (BiLSTM + TCN Ensemble)](#m3--forecasting-pipeline-bilstm--tcn-ensemble)
- [M3.5 — Agent Orchestration (LangGraph)](#m35--agent-orchestration-langgraph)
- [M3.6 — AutoML Tuning (FLAML)](#m36--automl-tuning-flaml)
- [M3.7 — LLM Alert Intelligence (RAG + Phi-3-mini)](#m37--llm-alert-intelligence-rag--phi-3-mini)
- [M3.8 — Physics-Informed Neural Network (PINN)](#m38--physics-informed-neural-network-pinn)
- [M3.9 — Conformal Prediction & Calibration](#m39--conformal-prediction--calibration)
- [M3.10 — Explainability (SHAP)](#m310--explainability-shap)
- [M4 — Dashboard & Visualization](#m4--dashboard--visualization)
- [M5 — Evaluation & Optimization](#m5--evaluation--optimization)
- [M6 — Production Deployment & MLOps](#m6--production-deployment--mlops)
- [Project Structure](#project-structure)
- [Evaluation Metrics](#evaluation-metrics)
- [References](#references)

---

## Project Overview

Solar flares are sudden, intense bursts of radiation from magnetic energy release in the solar atmosphere. M and X class flares disrupt satellite communications, GPS navigation, and power grids.

**SolarSentinel** is a full-stack, production-grade solar flare intelligence system built on Aditya-L1's SoLEXS and HEL1OS instruments, with three layers:

| Layer | What it does |
|---|---|
| **ML Core** | TCN+XGBoost nowcasting + BiLSTM+TCN forecasting with physics-informed loss |
| **Intelligence** | RAG-grounded LLM alert bulletins + SHAP per-prediction explanations |
| **MLOps** | Drift detection, auto-retraining, model registry, conformal uncertainty |

**Target metrics (not accuracy — that metric is meaningless here):**

| Metric | Target |
|---|---|
| TSS (True Skill Score) | > 0.80 |
| False Alarm Rate | < 0.10 |
| Mean Lead Time | > 30 min |
| C-class TPR | > 0.85 |
| ROC-AUC | > 0.90 |

> **Why not 99% accuracy?** With 95%+ quiet-sun samples, a model that always predicts "no flare" scores 99% accuracy while being completely useless. TSS = TPR − FPR is the correct operational metric.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                  PRADAN Portal (ISSDC) + GOES Archive               │
│     SoLEXS Level-1 FITS   HEL1OS Level-1 FITS   GOES XRS 40yr     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│               LANGGRAPH AGENT ORCHESTRATION LAYER                   │
│  [Ingestion] → [Preprocess] → [Router] → [Nowcast] → [Forecast]    │
│                                               ↓            ↓        │
│                                         [Alert Agent]               │
│                                               ↓                     │
│                                     [RAG + LLM Report Agent]        │
│                                     [SHAP Explanation Agent]        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DATA INGESTION LAYER                           │
│  FITS Reader → Dead-time Correction → Resampling → Background Sub  │
│  + Physics Feature Engineering (Neupert, Thermal Index, Lag)        │
│  + Time-Series Augmentation (tsaug — minority class oversampling)   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                  ▼
┌─────────────────────────┐     ┌─────────────────────────────────┐
│   NOWCASTING PIPELINE   │     │      FORECASTING PIPELINE        │
│                         │     │                                  │
│  Raw Flux Window        │     │  Rolling Sequence [T-60→T]      │
│        │                │     │          │                       │
│  [TCN Encoder]          │     │   ┌──────┴──────┐               │
│  Dilated causal conv    │     │   ▼             ▼               │
│  Multi-scale patterns   │     │ [BiLSTM]     [TCN]              │
│  128-dim embedding      │     │   │             │               │
│        │                │     │   └──────┬──────┘               │
│  + Physics features:    │     │          ▼                       │
│    dF/dt, Neupert ratio │     │   Weighted Ensemble              │
│    thermal index        │     │   Multi-Horizon Heads            │
│    instrument lag       │     │   P(flare | 15/30/60 min)       │
│    doubling time        │     │   Lead Time Estimate             │
│        │                │     │                                  │
│  [XGBoost + FLAML]      │     │   [PINN Loss during training]   │
│  + Platt Calibration    │     │   Physics-constrained gradients  │
│  + Conformal Intervals  │     │                                  │
│  + SHAP Explanations    │     └─────────────────┬───────────────┘
└──────────┬──────────────┘                        │
           │                                        │
           └───────────────────┬────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   LLM INTELLIGENCE LAYER                            │
│   Phi-3-mini / Mistral 7B via Ollama (runs fully local)            │
│   RAG: ChromaDB vectorstore of solar physics docs + NOAA reports   │
│   Output: "M2.3-class flare at 10:33 UTC. HF blackout likely..."   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│              MASTER FLARE CATALOGUE + ALERT ENGINE                  │
│              Drift Monitor | Model Registry | Auto-Retrain          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│              UNIFIED DASHBOARD (FastAPI + React/Streamlit)           │
│  Live light curves | Conformal bands | SHAP waterfall | LLM report │
│  Multi-horizon gauge | Replay mode | CSV export | Mobile PWA        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Data I/O | `astropy`, `sunpy` | FITS reading, GOES download |
| Preprocessing | `pandas`, `scipy`, `sklearn` | Calibration, resampling, features |
| Augmentation | `tsaug` | Minority class time-series augmentation |
| Nowcasting ML | `pytorch` (TCN), `xgboost` | Core nowcast pipeline |
| Forecasting ML | `pytorch` (BiLSTM + TCN) | Multi-horizon forecast |
| Physics ML | Custom PINN loss (PyTorch) | Neupert constraint during training |
| AutoML | `flaml` | XGBoost hyperparameter search |
| Uncertainty | `mapie` | Conformal prediction intervals |
| Calibration | `sklearn.calibration` | Platt scaling |
| Explainability | `shap` | Per-prediction feature attribution |
| Drift Detection | `alibi-detect` | Distribution shift monitoring |
| Orchestration | `langgraph`, `langchain-core` | Stateful agent pipeline |
| LLM Runtime | `ollama` (Phi-3-mini / Mistral 7B) | Local LLM inference |
| RAG | `chromadb`, `langchain-community` | Grounded alert generation |
| Experiment Tracking | `mlflow` | Model versioning + metrics |
| Backend API | `FastAPI`, `uvicorn` | REST + WebSocket API |
| Frontend | `React + TypeScript` / `Streamlit` | Dashboard UI |
| Fault Tolerance | `circuitbreaker` | PRADAN outage protection |
| Testing | `pytest`, `hypothesis` | Unit + property-based tests |
| CI/CD | `GitHub Actions` | Auto-test + retrain workflows |
| Containerization | `Docker`, `docker-compose` | Full stack deployment |
| Monitoring | `Prometheus + Grafana` | Latency, FAR drift, uptime |

---

## Accuracy Strategy

Every lever, in order of impact on TSS:

```
ACCURACY LEVERS
├── 1. GOES 40yr historical data    → More labeled flares to learn from
├── 2. Physics features (Neupert)   → Domain signal models can't invent
├── 3. PINN physics loss            → Constrain model to solar reality
├── 4. Multi-horizon output head    → One encoder, 3 horizons, less overfit
├── 5. Conformal + Platt scaling    → Honest calibrated probabilities
├── 6. FLAML auto-tuning            → Optimal XGBoost config, not guessed
├── 7. Time-series augmentation     → Synthetic minority class samples
│
PRODUCTION LEVERS
├── 8.  Drift detection (alibi)     → Know when model degrades in prod
├── 9.  Auto-retraining (Actions)   → Self-healing on schedule
├── 10. Model registry (MLflow)     → Safe promotion of better models
├── 11. Circuit breaker             → Survives PRADAN/GOES outages
├── 12. Async parallel inference    → Real-time sub-second response
├── 13. Property-based tests        → Catches edge cases humans miss
│
DIFFERENTIATION LEVERS
├── 14. RAG-grounded LLM reports    → Accurate bulletins, not hallucinated
├── 15. SHAP explanations           → "Why did it alert?" answered
├── 16. Anomaly explanation (LLM)   → Physics-grounded narrative
├── 17. Daily scheduled report      → LangGraph scheduled job
└── 18. PWA mobile dashboard        → Installable, works offline
```

---

## Milestone Map

```
M0──►M1──►M2──►M3──►M3.5──►M3.6──►M3.7──►M3.8──►M3.9──►M3.10──►M4──►M5──►M6
Set  Data Now  Fore Lang   FLAML  LLM    PINN   Conf   SHAP    Dash Eval Prod
up   Pipe cast cast Graph  AutoML +RAG   Loss   ormal  Expl    board &Opt Ready
```

| Milestone | Deliverable | Day |
|---|---|---|
| M0 | Env, repo, data access | 1–2 |
| M1 | FITS + GOES pipeline, labeled dataset | 3–5 |
| M2 | Nowcasting TCN+XGBoost, evaluated | 6–9 |
| M3 | Forecasting BiLSTM+TCN, evaluated | 10–12 |
| M3.5 | LangGraph agent orchestration | 13 |
| M3.6 | FLAML auto-tuning | 14 |
| M3.7 | RAG + LLM alert intelligence | 15 |
| M3.8 | PINN physics loss integration | 16 |
| M3.9 | Conformal prediction + Platt calibration | 17 |
| M3.10 | SHAP explainability layer | 18 |
| M4 | Dashboard live with all features | 19–20 |
| M5 | Benchmarked, optimized, documented | 21–22 |
| M6 | Docker, CI/CD, drift monitor, deployed | 23–25 |

---

## M0 — Environment & Setup

### 0.1 Repository Structure

```bash
git init solar-sentinel
cd solar-sentinel
```

```
solar-sentinel/
├── data/
│   ├── raw/
│   │   ├── solexs/              # SoLEXS FITS from PRADAN
│   │   ├── hel1os/              # HEL1OS FITS from PRADAN
│   │   └── goes/                # GOES XRS historical
│   ├── processed/
│   │   └── dataset.parquet
│   ├── labels/
│   │   └── noaa_flare_catalog.csv
│   ├── knowledge/               # RAG knowledge base
│   │   ├── noaa_event_reports/
│   │   ├── goes_user_guide.pdf
│   │   └── space_weather_scales.pdf
│   └── vectorstore/             # ChromaDB RAG index
├── src/
│   ├── ingestion/
│   │   ├── fits_reader.py
│   │   ├── goes_downloader.py
│   │   └── pradan_downloader.py
│   ├── preprocessing/
│   │   ├── pipeline.py
│   │   ├── physics_features.py
│   │   ├── augmentation.py
│   │   └── windows.py
│   ├── nowcasting/
│   │   ├── tcn_encoder.py
│   │   ├── train.py
│   │   └── flaml_tuner.py
│   ├── forecasting/
│   │   ├── bilstm.py
│   │   ├── ensemble.py
│   │   ├── multi_horizon.py
│   │   └── lead_time.py
│   ├── physics/
│   │   └── pinn_loss.py
│   ├── uncertainty/
│   │   ├── conformal.py
│   │   └── calibration.py
│   ├── explainability/
│   │   └── shap_explainer.py
│   ├── orchestration/
│   │   ├── state.py
│   │   ├── agents.py
│   │   └── graph.py
│   ├── intelligence/
│   │   ├── llm_reporter.py
│   │   ├── rag_builder.py
│   │   └── anomaly_explainer.py
│   ├── monitoring/
│   │   ├── drift_detector.py
│   │   └── model_registry.py
│   ├── evaluation/
│   │   └── metrics.py
│   └── api/
│       └── main.py
├── dashboard/
│   ├── src/
│   │   ├── components/
│   │   │   ├── LightCurve.tsx
│   │   │   ├── AlertBanner.tsx
│   │   │   ├── FlareCatalogue.tsx
│   │   │   ├── ForecastGauge.tsx
│   │   │   ├── ShapWaterfall.tsx
│   │   │   ├── ConformalBands.tsx
│   │   │   └── LLMReport.tsx
│   │   └── App.tsx
│   └── package.json
├── notebooks/
│   ├── 01_eda_fits_exploration.ipynb
│   ├── 02_physics_features.ipynb
│   ├── 03_nowcasting_prototype.ipynb
│   ├── 04_forecasting_prototype.ipynb
│   ├── 05_pinn_experiments.ipynb
│   ├── 06_conformal_calibration.ipynb
│   └── 07_langgraph_pipeline_test.ipynb
├── models/
│   ├── tcn_encoder.pt
│   ├── xgb_nowcast.json
│   ├── bilstm_forecaster.pt
│   ├── tcn_forecaster.pt
│   └── conformal_mapie.pkl
├── logs/
│   └── flaml_nowcast.log
├── configs/
│   ├── nowcasting.yaml
│   ├── forecasting.yaml
│   └── llm.yaml
├── tests/
│   ├── test_pipeline.py
│   └── test_properties.py
├── docker/
│   ├── Dockerfile.api
│   └── prometheus.yml
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── retrain.yml
├── docker-compose.yml
├── requirements.txt
└── README.md
```

### 0.2 Python Environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install \
  astropy sunpy numpy pandas scipy scikit-learn \
  torch xgboost flaml mlflow \
  langgraph langchain-core langchain-community \
  chromadb \
  mapie \
  shap \
  alibi-detect \
  tsaug \
  circuitbreaker \
  fastapi uvicorn \
  requests pytest hypothesis httpx \
  python-dotenv pyyaml \
  plotly matplotlib seaborn
```

### 0.3 Data Access — PRADAN + GOES

```bash
# 1. Register at https://pradan.issdc.gov.in (instant, no approval)
# 2. Download SoLEXS Level-1 + HEL1OS Level-1 FITS for known flare dates

# Known high-value dates (download these first):
# 2024-02-22  X6.3  — First jointly imaged by SoLEXS + HEL1OS
# 2024-05-10  X5.8  — Major CME event
# 2024-10-03  M-series — Multi-flare sequence

# Pull 10 years of GOES XRS (labels + supplementary features)
python src/ingestion/goes_downloader.py --start 2014-01-01 --end 2024-01-01

# Download NOAA flare event catalog
wget https://ftp.swpc.noaa.gov/pub/indices/events/ -r -np -nd \
     -P data/labels/ --accept "*.txt"
```

---

## M1 — Data Pipeline

### 1.1 FITS Reader

```python
# src/ingestion/fits_reader.py

from astropy.io import fits
import pandas as pd
import numpy as np

def read_solexs(filepath: str) -> pd.DataFrame:
    """
    SoLEXS Level-1: soft X-ray flux across 3 energy channels.
    Energy bands: ~1–3 keV (low), 3–6 keV (mid), 6–15 keV (high)
    """
    with fits.open(filepath) as hdul:
        data = hdul[1].data
        df = pd.DataFrame({
            'time':      data['TIME'],
            'flux_low':  data['FLUX_1'],
            'flux_mid':  data['FLUX_2'],
            'flux_high': data['FLUX_3'],
        })
    df['time'] = pd.to_datetime(df['time'], unit='s', origin='unix')
    return df.set_index('time').sort_index()


def read_hel1os(filepath: str) -> pd.DataFrame:
    """
    HEL1OS Level-1: hard X-ray counts in 2 energy channels.
    Energy bands: ~10–100 keV (low), 100 keV–1 MeV (high)
    """
    with fits.open(filepath) as hdul:
        data = hdul[1].data
        df = pd.DataFrame({
            'time':         data['TIME'],
            'counts_low':   data['COUNTS_1'],
            'counts_high':  data['COUNTS_2'],
        })
    df['time'] = pd.to_datetime(df['time'], unit='s', origin='unix')
    return df.set_index('time').sort_index()


def merge_instruments(solexs_df, hel1os_df, cadence='1min') -> pd.DataFrame:
    """Resample both instruments to common cadence and merge."""
    return pd.concat([
        solexs_df.resample(cadence).mean(),
        hel1os_df.resample(cadence).mean()
    ], axis=1).dropna()
```

### 1.2 Physics-Informed Feature Engineering

```python
# src/preprocessing/physics_features.py

import numpy as np
import pandas as pd
from scipy.signal import correlate

def engineer_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Time derivative — rate of flux change
    for col in ['flux_low', 'flux_mid', 'flux_high', 'counts_low', 'counts_high']:
        df[f'd_{col}_dt'] = df[col].diff()

    # 2. Neupert Effect Ratio
    # Hard X-ray integral ~ soft X-ray time derivative before flares
    # Violation of this ratio is a key precursor signal
    df['neupert_ratio'] = (
        df['counts_low'].diff() / (df['flux_high'] + 1e-12)
    )

    # 3. Thermal index (spectral slope)
    # Steepening = plasma heating = pre-flare signal
    df['thermal_index'] = (
        np.log10(df['flux_mid'] + 1e-12) -
        np.log10(df['flux_low'] + 1e-12)
    )

    # 4. Flux doubling time
    # Fast doubling = impulsive rise = active flare
    flux_ratio = df['flux_high'] / df['flux_high'].shift(5)
    df['doubling_time'] = 5.0 / (np.log2(flux_ratio.clip(lower=1e-6)))

    # 5. Spectral hardness ratio (soft/hard energy proxy)
    df['channel_ratio'] = df['flux_high'] / (df['counts_low'] + 1e-10)

    # 6. Rolling statistics
    for col in ['flux_low', 'flux_mid', 'flux_high']:
        df[f'{col}_rollstd_5']  = df[col].rolling(5).std()
        df[f'{col}_rollstd_15'] = df[col].rolling(15).std()
        df[f'{col}_rollmax_5']  = df[col].rolling(5).max()

    # 7. Normalised flux (local quiet-sun baseline ratio)
    df['normalised_flux'] = (
        df['flux_high'] /
        df['flux_high'].rolling(120).quantile(0.10).clip(lower=1e-12)
    )

    # 8. Cross-instrument temporal lag
    # SoLEXS peaks before HEL1OS — lag indicates event phase
    window = min(60, len(df))
    corr = correlate(
        df['flux_mid'].fillna(0).values[-window:],
        df['counts_low'].fillna(0).values[-window:]
    )
    df['instrument_lag'] = corr.argmax() - window + 1

    return df.dropna()


def subtract_background(df: pd.DataFrame, window_minutes=10) -> pd.DataFrame:
    """Rolling median background subtraction. Output clipped to >= 0."""
    df_clean = df.copy()
    for col in df.columns:
        baseline = df[col].rolling(window_minutes, min_periods=1).median()
        df_clean[col] = (df[col] - baseline).clip(lower=0)
    return df_clean
```

### 1.3 Time-Series Augmentation

```python
# src/preprocessing/augmentation.py

import numpy as np
from tsaug import TimeWarp, Drift, Quantize

augmenter = (
    TimeWarp(n_speed_change=3, max_speed_ratio=3.0) * 2
    + Drift(max_drift=0.1, n_drift_points=5)
    + Quantize(n_levels=20)
)

def augment_minority_class(X_train: np.ndarray,
                            y_train: np.ndarray,
                            target_ratio: float = 0.3) -> tuple:
    """
    Augment only M/X-class flare windows until they are
    `target_ratio` fraction of the total dataset.
    """
    X_flare = X_train[y_train == 1]
    X_quiet = X_train[y_train == 0]

    n_needed = int(len(X_quiet) * target_ratio) - len(X_flare)
    if n_needed <= 0:
        return X_train, y_train

    # Augment flare windows
    reps = (n_needed // len(X_flare)) + 1
    X_aug_pool = np.vstack([
        augmenter.augment(X_flare) for _ in range(reps)
    ])[:n_needed]

    X_final = np.vstack([X_train, X_aug_pool])
    y_final = np.hstack([y_train, np.ones(len(X_aug_pool), dtype=int)])
    return X_final, y_final
```

### 1.4 Dataset Builder

```python
# src/preprocessing/pipeline.py

import pandas as pd
import numpy as np

def build_labeled_dataset(merged_df: pd.DataFrame,
                           flare_catalog: pd.DataFrame) -> pd.DataFrame:
    """Cross-reference NOAA catalog to assign flare labels."""
    merged_df = merged_df.copy()
    merged_df['label'] = 0
    merged_df['flare_class'] = 'N'

    for _, flare in flare_catalog.iterrows():
        mask = (
            (merged_df.index >= flare['start_time']) &
            (merged_df.index <= flare['end_time'])
        )
        merged_df.loc[mask, 'label'] = 1
        merged_df.loc[mask, 'flare_class'] = flare['class'][0]

    return merged_df


def create_windows(df, feature_cols, window_size=60,
                   horizon=15, step=1):
    """
    Sliding window generator.
    Returns X (N, T, F), y_now (N,), y_fore (N,)
    """
    X, y_now, y_fore = [], [], []
    data   = df[feature_cols].values
    labels = df['label'].values

    for i in range(0, len(data) - window_size - horizon, step):
        X.append(data[i:i + window_size])
        y_now.append(labels[i + window_size - 1])
        y_fore.append(labels[i + window_size + horizon - 1])

    return (np.array(X, dtype=np.float32),
            np.array(y_now, dtype=np.int64),
            np.array(y_fore, dtype=np.int64))
```

**M1 Checkpoint:** `data/processed/dataset.parquet` exists. Class distribution logged. Augmented training set has ~30% positive class ratio.

---

## M2 — Nowcasting Pipeline (TCN + XGBoost)

### 2.1 TCN Encoder

```python
# src/nowcasting/tcn_encoder.py

import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              dilation=dilation, padding=self.pad)

    def forward(self, x):
        return self.conv(x)[:, :, :-self.pad]


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1    = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.conv2    = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.norm1    = nn.LayerNorm(out_ch)
        self.norm2    = nn.LayerNorm(out_ch)
        self.drop     = nn.Dropout(dropout)
        self.relu     = nn.ReLU()
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        res = x
        out = self.relu(self.norm1(self.conv1(x).transpose(1,2)).transpose(1,2))
        out = self.drop(out)
        out = self.norm2(self.conv2(out).transpose(1,2)).transpose(1,2)
        out = self.drop(out)
        if self.downsample: res = self.downsample(res)
        return self.relu(out + res)


class TCNEncoder(nn.Module):
    """
    Encodes a light-curve window into a 128-dim feature vector.
    No future leakage (causal convolutions only).
    """
    def __init__(self, n_features, embed_dim=128, n_layers=4, kernel_size=3):
        super().__init__()
        channels  = [n_features] + [embed_dim] * n_layers
        dilations = [2 ** i for i in range(n_layers)]
        self.blocks = nn.ModuleList([
            TCNBlock(channels[i], channels[i+1], kernel_size, dilations[i])
            for i in range(n_layers)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.transpose(1, 2)          # (B, T, F) → (B, F, T)
        for block in self.blocks:
            x = block(x)
        return self.pool(x).squeeze(-1) # (B, embed_dim)
```

### 2.2 XGBoost Classifier

```python
# src/nowcasting/train.py

import numpy as np
import torch
import xgboost as xgb
import mlflow
from src.nowcasting.tcn_encoder import TCNEncoder

def extract_tcn_features(encoder, X, device='cpu', batch_size=256):
    encoder.eval().to(device)
    feats = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            b = torch.tensor(X[i:i+batch_size]).to(device)
            feats.append(encoder(b).cpu().numpy())
    return np.vstack(feats)


def train_nowcasting(X, y, handcrafted_feats, device='cpu'):
    with mlflow.start_run(run_name="nowcast_tcn_xgb"):

        encoder = TCNEncoder(n_features=X.shape[2])
        tcn_feats = extract_tcn_features(encoder, X, device)
        combined  = np.concatenate([tcn_feats, handcrafted_feats], axis=1)

        # Time-aware split — never shuffle time-series
        split = int(0.8 * len(combined))
        X_tr, X_val = combined[:split], combined[split:]
        y_tr, y_val = y[:split], y[split:]

        neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()

        model = xgb.XGBClassifier(
            n_estimators=500, max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=neg / pos,
            eval_metric='logloss',
            early_stopping_rounds=50,
            tree_method='hist'
        )
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)], verbose=100)

        mlflow.xgboost.log_model(model, "nowcast_xgb")
        return encoder, model
```

**M2 Checkpoint:** Nowcasting runs end-to-end. TSS > 0.5 on validation. Feature importance chart saved.

---

## M3 — Forecasting Pipeline (BiLSTM + TCN Ensemble)

### 3.1 BiLSTM

```python
# src/forecasting/bilstm.py

import torch.nn as nn

class BiLSTMForecaster(nn.Module):
    def __init__(self, n_features, hidden_dim=128, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_dim, n_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.norm(out[:, -1, :])).squeeze(-1)
```

### 3.2 Multi-Horizon Forecasting Head

```python
# src/forecasting/multi_horizon.py

import torch.nn as nn
from src.nowcasting.tcn_encoder import TCNEncoder

class MultiHorizonForecaster(nn.Module):
    """Single encoder, three forecast horizons (15 / 30 / 60 min)."""
    def __init__(self, n_features, horizons=[15, 30, 60]):
        super().__init__()
        self.horizons = horizons
        self.encoder  = TCNEncoder(n_features, embed_dim=256)
        self.heads = nn.ModuleDict({
            f"h{h}": nn.Sequential(
                nn.Linear(256, 64), nn.ReLU(),
                nn.Dropout(0.3), nn.Linear(64, 1), nn.Sigmoid()
            ) for h in horizons
        })

    def forward(self, x):
        feat = self.encoder(x)
        return {f"h{h}": self.heads[f"h{h}"](feat).squeeze(-1)
                for h in self.horizons}
```

### 3.3 Weighted Ensemble + Focal Loss

```python
# src/forecasting/ensemble.py

import torch, torch.nn as nn, mlflow

class WeightedEnsemble(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.tensor([0.5, 0.5]))

    def forward(self, p_bi, p_tc):
        w = torch.softmax(self.weights, dim=0)
        return w[0] * p_bi + w[1] * p_tc


def focal_loss(pred, target, gamma=2.0, alpha=0.75):
    bce = nn.functional.binary_cross_entropy(pred, target.float(), reduction='none')
    pt  = torch.exp(-bce)
    return (alpha * (1 - pt) ** gamma * bce).mean()
```

**M3 Checkpoint:** Ensemble trained for all three horizons. Multi-horizon outputs logged to MLflow.

---

## M3.5 — Agent Orchestration (LangGraph)

### 3.5.1 Pipeline State

```python
# src/orchestration/state.py

from typing import TypedDict, Optional
import pandas as pd, numpy as np

class SolarPipelineState(TypedDict):
    solexs_path:         str
    hel1os_path:         str
    raw_df:              Optional[pd.DataFrame]
    processed_df:        Optional[pd.DataFrame]
    nowcast_class:       Optional[str]
    nowcast_confidence:  Optional[float]
    nowcast_shap:        Optional[dict]
    forecast_proba:      Optional[float]
    forecast_horizon:    Optional[int]
    lead_time_minutes:   Optional[float]
    conformal_interval:  Optional[tuple]
    alert_triggered:     bool
    llm_report:          Optional[str]
    shap_explanation:    Optional[str]
    errors:              list
    timestamp:           str
```

### 3.5.2 Agent Nodes + Graph

```python
# src/orchestration/graph.py

from langgraph.graph import StateGraph, END
from src.orchestration.agents import (
    ingestion_agent, preprocessing_agent,
    nowcast_agent, forecast_agent,
    shap_agent, llm_report_agent, alert_router
)
from src.orchestration.state import SolarPipelineState

def build_pipeline():
    g = StateGraph(SolarPipelineState)

    g.add_node("ingestion",     ingestion_agent)
    g.add_node("preprocessing", preprocessing_agent)
    g.add_node("nowcast",       nowcast_agent)
    g.add_node("forecast",      forecast_agent)
    g.add_node("shap",          shap_agent)
    g.add_node("llm_report",    llm_report_agent)

    g.set_entry_point("ingestion")
    g.add_edge("ingestion",     "preprocessing")
    g.add_edge("preprocessing", "nowcast")
    g.add_edge("nowcast",       "forecast")
    g.add_edge("forecast",      "shap")

    # Only call LLM if alert warranted (M/X or P > 0.75)
    g.add_conditional_edges(
        "shap", alert_router,
        {"llm_report_agent": "llm_report", "end": END}
    )
    g.add_edge("llm_report", END)
    return g.compile()


if __name__ == "__main__":
    pipeline = build_pipeline()
    result = pipeline.invoke({
        "solexs_path": "data/raw/solexs/2024-02-22.fits",
        "hel1os_path": "data/raw/hel1os/2024-02-22.fits",
        "errors": [], "alert_triggered": False,
        "timestamp": "2024-02-22T10:33:00Z"
    })
    print("Nowcast:", result["nowcast_class"], result["nowcast_confidence"])
    print("SHAP:   ", result["shap_explanation"])
    print("Report: ", result["llm_report"])
```

**M3.5 Checkpoint:** `python src/orchestration/graph.py` runs end-to-end. All agents chained. Conditional routing verified.

---

## M3.6 — AutoML Tuning (FLAML)

```python
# src/nowcasting/flaml_tuner.py

from flaml import AutoML
import mlflow

def autotune_nowcast(X_tr, y_tr, X_val, y_val,
                     time_budget: int = 3600):
    """
    FLAML searches XGBoost, LightGBM, RF, ExtraTrees.
    Optimises ROC-AUC within time_budget seconds.
    """
    automl = AutoML()
    with mlflow.start_run(run_name="flaml_nowcast"):
        automl.fit(
            X_tr, y_tr,
            task="classification",
            metric="roc_auc",
            time_budget=time_budget,
            estimator_list=["xgboost", "lgbm", "rf", "extra_tree"],
            X_val=X_val, y_val=y_val,
            log_file_name="logs/flaml_nowcast.log",
            verbose=2
        )
        mlflow.log_param("best_estimator", automl.best_estimator)
        mlflow.log_params(automl.best_config)
        mlflow.log_metric("val_auc", 1 - automl.best_loss)
        print(f"Best: {automl.best_estimator} | AUC: {1-automl.best_loss:.4f}")
    return automl
```

**FLAML searches automatically over:**

| Parameter | Range |
|---|---|
| `n_estimators` | 4 – 32768 |
| `max_depth` | 4 – 12 |
| `learning_rate` | 0.001 – 1.0 |
| `subsample` | 0.5 – 1.0 |
| `reg_alpha / lambda` | 1e-10 – 1.0 |

**M3.6 Checkpoint:** Best config logged to MLflow. Retrain XGBoost with FLAML-found params. TSS should improve 0.05–0.10 over manual config.

---

## M3.7 — LLM Alert Intelligence (RAG + Phi-3-mini)

### 3.7.1 Setup Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull phi3:mini       # 3.8B — fast, accurate, runs on CPU
ollama pull mistral:7b      # 7B  — richer output, needs GPU
ollama run phi3:mini "Summarise a C-class solar flare."
```

### 3.7.2 Build RAG Knowledge Base

```python
# src/intelligence/rag_builder.py

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import (
    DirectoryLoader, PyPDFLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

def build_rag_index(knowledge_dir: str = "data/knowledge",
                    persist_dir:   str = "data/vectorstore"):
    loader  = DirectoryLoader(knowledge_dir, glob="**/*.txt")
    docs    = loader.load()

    pdf_loader = PyPDFLoader("data/knowledge/goes_user_guide.pdf")
    docs      += pdf_loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    embeddings  = OllamaEmbeddings(model="phi3:mini")
    vectorstore = Chroma.from_documents(
        chunks, embeddings, persist_directory=persist_dir
    )
    print(f"Indexed {len(chunks)} chunks into RAG store.")
    return vectorstore
```

### 3.7.3 RAG-Grounded Report Generator

```python
# src/intelligence/llm_reporter.py

import requests
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "phi3:mini"

SYSTEM_PROMPT = """You are an expert solar physicist and space weather
operator assistant. Generate concise, factual, actionable bulletins
when solar flares are detected. Ground your response in the provided
context from the solar physics knowledge base. Under 5 sentences.
Never hallucinate instrument names or flux values."""

SEVERITY = {
    "C": "minor. Low risk. HF radio may degrade briefly.",
    "M": "moderate. HF blackout likely on sunlit side. Satellite precautions advised.",
    "X": "MAJOR. Severe radio blackout. GPS degradation. Power grid GIC risk. Immediate action required."
}

def generate_flare_report(flare_data: dict,
                           vectorstore: Chroma = None) -> str:
    cls = (flare_data.get('class') or 'C')[0].upper()
    severity_ctx = SEVERITY.get(cls, "unknown severity.")

    # RAG retrieval
    rag_context = ""
    if vectorstore:
        query = f"{cls}-class solar flare space weather impact"
        docs  = vectorstore.similarity_search(query, k=3)
        rag_context = "\n".join(d.page_content for d in docs)

    prompt = f"""
Knowledge base context:
{rag_context}

Detected event:
- Class:       {flare_data.get('class', 'Unknown')}
- Time:        {flare_data.get('timestamp', 'N/A')}
- Confidence:  {flare_data.get('confidence', 0):.0%}
- Forecast P:  {flare_data.get('forecast_proba', 0):.0%} (next 15 min)
- Lead time:   {flare_data.get('lead_time', 0):.1f} min before peak
- Severity:    {severity_ctx}

Generate an operational space weather bulletin for satellite operators
and grid managers. Ground every claim in the context above.
"""
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": MODEL, "system": SYSTEM_PROMPT,
            "prompt": prompt, "stream": False,
            "options": {"temperature": 0.2, "num_predict": 200}
        }, timeout=30)
        return r.json().get("response", "LLM unavailable.")
    except Exception as e:
        return f"Alert: {cls}-class flare. Confidence: {flare_data.get('confidence',0):.0%}. LLM offline."
```

**M3.7 Checkpoint:** RAG index built. `generate_flare_report()` returns grounded bulletin in < 3s. No hallucinated instrument names.

---

## M3.8 — Physics-Informed Neural Network (PINN)

Adds a physics constraint to the training loss. Forces the model to respect the **Neupert Effect** — hard X-ray flux should correlate with the derivative of soft X-ray flux during real flares. Models that violate this are penalised.

```python
# src/physics/pinn_loss.py

import torch
import torch.nn as nn

class PINNLoss(nn.Module):
    """
    Combined classification + physics constraint loss.

    Neupert Effect (empirical solar physics law):
        HXR(t) ∝ d(SXR)/dt
    During a real flare, soft X-ray derivative must track
    hard X-ray counts. Penalise models that violate this.
    """
    def __init__(self, physics_weight: float = 0.15):
        super().__init__()
        self.physics_weight = physics_weight
        self.bce = nn.BCELoss()

    def forward(self, pred, target,
                soft_xray: torch.Tensor,
                hard_xray: torch.Tensor) -> torch.Tensor:

        # Standard classification loss
        cls_loss = self.bce(pred, target.float())

        # Physics constraint: d(SXR)/dt should scale with HXR
        dSXR_dt = torch.diff(soft_xray, dim=1)          # (B, T-1)
        hxr     = hard_xray[:, 1:]                       # align dims

        # Penalise when they move in opposite directions during flares
        violation = torch.relu(-(dSXR_dt * hxr))         # > 0 = violation
        physics_loss = violation.mean()

        return cls_loss + self.physics_weight * physics_loss


# Usage in training loop:
# criterion = PINNLoss(physics_weight=0.15)
# loss = criterion(pred, y_batch, soft_xray_batch, hard_xray_batch)
# loss.backward()
```

**M3.8 Checkpoint:** PINN loss added to BiLSTM training loop. Physics violation metric logged to MLflow. TSS should improve 0.03–0.07 over vanilla BCE.

---

## M3.9 — Conformal Prediction & Calibration

### 3.9.1 Platt Scaling (Probability Calibration)

Raw model probabilities are often overconfident. Platt scaling fixes this.

```python
# src/uncertainty/calibration.py

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt

def calibrate_model(xgb_model, X_cal, y_cal):
    """Wrap XGBoost with Platt (sigmoid) calibration."""
    calibrated = CalibratedClassifierCV(
        xgb_model, method='sigmoid', cv='prefit'
    )
    calibrated.fit(X_cal, y_cal)
    return calibrated


def plot_calibration(model, X_test, y_test, save_path="plots/calibration.png"):
    prob_true, prob_pred = calibration_curve(
        y_test,
        model.predict_proba(X_test)[:, 1],
        n_bins=10
    )
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.plot(prob_pred, prob_true, 'b-o', label='Model')
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.savefig(save_path)
```

### 3.9.2 Conformal Prediction (Guaranteed Intervals)

```python
# src/uncertainty/conformal.py

from mapie.classification import MapieClassifier
import numpy as np

def fit_conformal(calibrated_model, X_cal, y_cal):
    """
    Wraps calibrated model with conformal prediction.
    Guarantees that the prediction set contains the true
    label with probability >= 1 - alpha.
    """
    mapie = MapieClassifier(
        estimator=calibrated_model,
        method="score",
        cv="prefit"
    )
    mapie.fit(X_cal, y_cal)
    return mapie


def predict_with_interval(mapie, X_new, alpha=0.10):
    """
    Returns point prediction + guaranteed 90% coverage set.
    Dashboard renders: "73% [61% – 84%]"
    """
    y_pred, y_sets = mapie.predict(X_new, alpha=alpha)
    # y_sets shape: (N, n_classes, n_alpha)
    return {
        "prediction":        y_pred,
        "conformal_set":     y_sets,
        "coverage_guarantee": f"{(1-alpha)*100:.0f}%"
    }


def optimize_threshold(y_true, y_proba,
                        optimize_for='tss') -> tuple:
    """
    Sweep thresholds 0.1–0.9, return the one maximising TSS.
    Never use default 0.5 for imbalanced flare prediction.
    """
    from src.evaluation.metrics import compute_tss
    best_t, best_s = 0.5, 0.0
    for t in np.arange(0.1, 0.95, 0.05):
        s = compute_tss(y_true, (y_proba >= t).astype(int))
        if s > best_s:
            best_s, best_t = s, t
    return best_t, best_s
```

**M3.9 Checkpoint:** Calibration curve diagonal. Conformal intervals on dashboard. `optimize_threshold()` run — use its output, not 0.5.

---

## M3.10 — Explainability (SHAP)

```python
# src/explainability/shap_explainer.py

import shap
import numpy as np

def build_shap_explainer(xgb_model):
    return shap.TreeExplainer(xgb_model)


def explain_prediction(explainer, x: np.ndarray,
                        feature_names: list) -> dict:
    """
    Returns top-5 feature contributions for a single prediction.
    Rendered in dashboard as a waterfall chart.
    """
    sv = explainer.shap_values(x.reshape(1, -1))
    if isinstance(sv, list):
        sv = sv[1]   # class 1 (flare) SHAP values
    sv = sv[0]

    ranked = sorted(
        zip(feature_names, sv),
        key=lambda t: abs(t[1]), reverse=True
    )[:5]

    return {
        "top_drivers": [
            {"feature": f, "contribution": round(float(v), 5),
             "direction": "↑ towards flare" if v > 0 else "↓ towards quiet"}
            for f, v in ranked
        ],
        "base_value": float(explainer.expected_value
                            if not isinstance(explainer.expected_value, list)
                            else explainer.expected_value[1])
    }


def global_importance_plot(explainer, X_test, feature_names,
                            save_path="plots/shap_summary.png"):
    sv = explainer.shap_values(X_test)
    if isinstance(sv, list): sv = sv[1]
    shap.summary_plot(sv, X_test,
                      feature_names=feature_names,
                      show=False)
    import matplotlib.pyplot as plt
    plt.savefig(save_path, bbox_inches='tight')
    print(f"SHAP summary plot saved to {save_path}")
```

Dashboard renders per-alert:
```
Alert triggered because:
  → Hard X-ray spike:      +0.42  ↑ towards flare
  → Neupert ratio shift:   +0.31  ↑ towards flare
  → Thermal index:         +0.18  ↑ towards flare
  → Flux doubling time:    −0.09  ↓ towards quiet
  → Channel ratio:         +0.07  ↑ towards flare
```

**M3.10 Checkpoint:** `explain_prediction()` returns in < 200ms. SHAP summary plot saved. Dashboard renders waterfall.

---

## M4 — Dashboard & Visualization

### 4.1 FastAPI Backend

```python
# src/api/main.py

import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="SolarSentinel API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
executor = ThreadPoolExecutor(max_workers=4)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/nowcast/latest")
async def get_nowcast():
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, run_nowcast_inference,
                                        load_latest_window())
    return result


@app.get("/api/forecast")
async def get_forecast(horizon: int = 15):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, run_forecast_inference,
                                        load_latest_window(), horizon)
    return result


@app.get("/api/explain")
async def get_explanation(prediction_id: str):
    return load_shap_explanation(prediction_id)


@app.get("/api/lightcurve")
def get_lightcurve(start: str, end: str):
    return load_lightcurve(start, end).reset_index().to_dict("records")


@app.get("/api/flare-catalog")
def get_catalog():
    return {"flares": load_master_catalog()}


@app.get("/api/daily-report")
async def get_daily_report():
    stats = compute_24hr_stats()
    return {"report": generate_daily_summary(stats),
            "date": str(__import__('datetime').date.today())}


@app.get("/api/drift-status")
def get_drift():
    return check_drift(load_recent_features())


@app.websocket("/ws/live")
async def live_stream(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = get_latest_reading()
            await ws.send_json(data)
            await asyncio.sleep(60)
    except Exception:
        await ws.close()
```

### 4.2 Dashboard Layout

```
┌──────────────────────────────────────────────────────────────┐
│  ☀️ SolarSentinel            [LIVE] [REPLAY] [HISTORY] [⚙️]  │
├──────────────────────────────────────────────────────────────┤
│  SoLEXS Light Curve + Conf.Bands │ HEL1OS Light Curve        │
│  [Plotly streaming, shaded 90%CI]│ [Counts, threshold line]  │
├─────────────────────┬────────────────────────────────────────┤
│  NOWCAST            │  FORECAST                              │
│  🟠 M2.3-CLASS      │  P(15min): 73% [61–84%]               │
│  Conf: 87%          │  P(30min): 61% [49–72%]               │
│  ✅ Conformal: 90%  │  P(60min): 44% [33–56%]               │
├─────────────────────┴────────────────────────────────────────┤
│  SHAP EXPLANATION                                            │
│  → Hard X-ray spike:     +0.42 ████████░░                   │
│  → Neupert ratio shift:  +0.31 ██████░░░░                   │
│  → Thermal index:        +0.18 ████░░░░░░                   │
├──────────────────────────────────────────────────────────────┤
│  🔴 OPERATIONAL BULLETIN (Phi-3-mini + RAG)                  │
│  "An M2.3-class flare at 10:33 UTC poses moderate risk to    │
│  HF communications. Satellite operators: activate anomaly    │
│  monitoring. Grid operators: monitor GIC levels."            │
├──────────────────────────────────────────────────────────────┤
│  FLARE CATALOGUE                                             │
│  Date | Peak | Class | Conf | Lead | Source | [Export CSV]  │
└──────────────────────────────────────────────────────────────┘
```

**Alert Levels:**

| Threshold | Colour | Action |
|---|---|---|
| P > 0.50 | 🟡 Yellow banner | Monitoring mode |
| P > 0.75 | 🟠 Orange + audio ping | Advisory issued |
| P > 0.90 | 🔴 Red CRITICAL + email + Slack webhook | Immediate action |

**M4 Checkpoint:** Dashboard live on `localhost:3000`. WebSocket streaming at 1-min cadence. SHAP waterfall renders. LLM bulletin appears on alert trigger. Replay mode tested on known flare dates.

---

## M5 — Evaluation & Optimization

### 5.1 Metrics

```python
# src/evaluation/metrics.py

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

def compute_tss(y_true, y_pred) -> float:
    """True Skill Score = TPR - FPR. Range [-1, 1]. Target > 0.80."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn + 1e-10)
    fpr = fp / (fp + tn + 1e-10)
    return tpr - fpr


def compute_far(y_true, y_pred) -> float:
    """False Alarm Rate = FP / (FP + TN). Target < 0.10."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn + 1e-10)


def compute_lead_time(y_proba, y_true, timestamps,
                       threshold=0.5) -> dict:
    """Compute mean lead time across all detected flare events."""
    alerts  = timestamps[y_proba >= threshold]
    onsets  = timestamps[np.diff(y_true.astype(int), prepend=0) == 1]
    results = []
    for onset in onsets:
        prior = alerts[alerts < onset]
        if len(prior) > 0:
            lead = (onset - prior[-1]).total_seconds() / 60
            results.append({"onset": onset, "lead_min": lead, "caught": True})
        else:
            results.append({"onset": onset, "lead_min": 0, "caught": False})
    leads = [r["lead_min"] for r in results]
    return {
        "mean_lead_min":  np.mean(leads),
        "max_lead_min":   np.max(leads) if leads else 0,
        "detection_rate": np.mean([r["caught"] for r in results]),
        "per_event":      results
    }


def full_report(y_true, y_pred, y_proba, timestamps, horizon=None):
    from sklearn.metrics import classification_report
    print("=" * 45)
    print("SolarSentinel Evaluation Report")
    print("=" * 45)
    print(f"TSS:     {compute_tss(y_true, y_pred):.4f}  (target > 0.80)")
    print(f"FAR:     {compute_far(y_true, y_pred):.4f}  (target < 0.10)")
    print(f"ROC-AUC: {roc_auc_score(y_true, y_proba):.4f} (target > 0.90)")
    print(classification_report(y_true, y_pred,
                                 target_names=['Quiet', 'Flare']))
    if horizon:
        lt = compute_lead_time(y_proba, y_true, timestamps)
        print(f"Mean Lead Time: {lt['mean_lead_min']:.1f} min (target > 30)")
        print(f"Detection Rate: {lt['detection_rate']*100:.1f}%")


def evaluate_per_class(y_class_true, y_class_pred):
    """Judges specifically check C-class detection."""
    for cls in ['C', 'M', 'X']:
        mask = (y_class_true == cls) | (y_class_true == 'N')
        yt = (y_class_true[mask] == cls).astype(int)
        yp = (y_class_pred[mask] == cls).astype(int)
        print(f"Class {cls}: TSS={compute_tss(yt,yp):.3f}  "
              f"FAR={compute_far(yt,yp):.3f}")
```

### 5.2 Target Benchmarks

| Metric | Minimum to pass | Target to win |
|---|---|---|
| TSS (nowcast) | 0.50 | **> 0.80** |
| TSS (forecast 15min) | 0.40 | **> 0.65** |
| FAR | < 0.30 | **< 0.10** |
| ROC-AUC | 0.80 | **> 0.90** |
| Mean Lead Time | 10 min | **> 30 min** |
| C-class TPR | 60% | **> 85%** |

---

## M6 — Production Deployment & MLOps

### 6.1 Drift Detection

```python
# src/monitoring/drift_detector.py

from alibi_detect.cd import TabularDrift
import numpy as np

detector = TabularDrift(X_reference, p_val=0.05)

def check_drift(X_new: np.ndarray) -> dict:
    result    = detector.predict(X_new)
    is_drift  = result['data']['is_drift']
    p_val     = result['data']['p_val']
    if is_drift:
        trigger_retrain_workflow()
    return {"drift": bool(is_drift), "p_value": float(p_val)}
```

### 6.2 Model Registry

```python
# src/monitoring/model_registry.py

import mlflow

def promote_if_better(new_run_id: str, min_improvement=0.02):
    client   = mlflow.tracking.MlflowClient()
    prod     = client.get_latest_versions("nowcast_model", stages=["Production"])
    prod_tss = float(prod[0].tags.get("tss", "0")) if prod else 0.0
    new_tss  = client.get_run(new_run_id).data.metrics.get("tss", 0.0)

    if new_tss > prod_tss + min_improvement:
        client.transition_model_version_stage(
            "nowcast_model", new_run_id, "Production"
        )
        print(f"Promoted: {prod_tss:.3f} → {new_tss:.3f}")
    else:
        print(f"Kept current model. New TSS {new_tss:.3f} insufficient.")
```

### 6.3 Circuit Breaker

```python
# src/ingestion/safe_fetch.py

from circuitbreaker import circuit
import requests

@circuit(failure_threshold=3, recovery_timeout=60)
def fetch_latest_fits(instrument: str) -> bytes:
    r = requests.get(f"{PRADAN_URL}/{instrument}", timeout=10)
    r.raise_for_status()
    return r.content

def safe_fetch(instrument: str):
    try:
        return fetch_latest_fits(instrument)
    except Exception:
        return load_cached_fits(instrument)  # last known good reading
```

### 6.4 Property-Based Tests

```python
# tests/test_properties.py

from hypothesis import given, strategies as st
import numpy as np

@given(st.floats(min_value=0, max_value=1e-3))
def test_background_subtraction_nonnegative(flux_value):
    """Background subtracted flux must never be negative."""
    result = subtract_background(make_test_df(flux_value))
    assert (result >= 0).all().all()


@given(st.arrays(np.float32, shape=(60, 8),
                  elements=st.floats(0, 1e-3, allow_nan=False)))
def test_tcn_output_bounded(window):
    """TCN output probability must always be in [0, 1]."""
    prob = load_nowcast_model().predict(window[np.newaxis])
    assert 0.0 <= float(prob) <= 1.0


@given(st.arrays(np.int64, shape=(100,),
                  elements=st.integers(0, 1)))
def test_tss_range(y):
    """TSS must always be in [-1, 1]."""
    tss = compute_tss(y, np.zeros_like(y))
    assert -1.0 <= tss <= 1.0
```

### 6.5 Docker Compose (Full Stack)

```yaml
# docker-compose.yml
version: "3.9"

services:
  api:
    build: { context: ., dockerfile: docker/Dockerfile.api }
    ports: ["8000:8000"]
    volumes: [./data:/app/data, ./models:/app/models]
    environment:
      - OLLAMA_URL=http://ollama:11434
      - MLFLOW_TRACKING_URI=http://mlflow:5000
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

### 6.6 CI/CD — Test + Retrain

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
name: Scheduled Retraining
on:
  schedule: [{ cron: '0 0 * * 0' }]  # Every Sunday
  workflow_dispatch:
jobs:
  retrain:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: python src/ingestion/goes_downloader.py --days=30
      - run: python src/nowcasting/train.py --mode=incremental
      - run: python src/evaluation/metrics.py --compare
      - run: python src/monitoring/model_registry.py --promote-if-better
```

### 6.7 Grafana Monitoring — Track in Production

```
Dashboards to build in Grafana:
├── Model Performance
│   ├── TSS over time (detect degradation)
│   ├── FAR over time
│   └── Lead time distribution
├── System Health
│   ├── API inference latency (p50, p99)
│   ├── PRADAN fetch success rate
│   └── Ollama LLM response time
└── Alert Activity
    ├── Alerts per day (C / M / X)
    ├── False alarm count
    └── Drift detection events
```

---

## Evaluation Metrics

| Metric | Formula | Target |
|---|---|---|
| **TSS** | TPR − FPR | **> 0.80** |
| **FAR** | FP / (FP + TN) | **< 0.10** |
| **ROC-AUC** | Area under ROC curve | **> 0.90** |
| **Mean Lead Time** | Avg(flare onset − first alert) | **> 30 min** |
| **C-class TPR** | TP_C / (TP_C + FN_C) | **> 0.85** |
| **Conformal Coverage** | Empirical coverage @ α=0.10 | **≥ 90%** |

> **Never report raw accuracy.** 95%+ quiet-sun data means a model predicting "no flare" always hits 99% accuracy while being completely useless. TSS is the correct operational metric across all solar flare ML literature.

---

## References

- Aditya-L1 Mission — ISRO: https://www.isro.gov.in
- PRADAN Data Portal — ISSDC: https://pradan.issdc.gov.in
- NOAA SWPC Flare Catalog: https://ftp.swpc.noaa.gov/pub/indices/events/
- GOES XRS Data Archive — NCEI: https://www.ngdc.noaa.gov/stp/solar/solarflares.html
- SunPy Documentation: https://sunpy.org
- LangGraph: https://langchain-ai.github.io/langgraph/
- FLAML — Microsoft AutoML: https://microsoft.github.io/FLAML/
- MAPIE — Conformal Prediction: https://mapie.readthedocs.io
- Alibi-Detect — Drift Detection: https://docs.seldon.io/projects/alibi-detect
- SHAP: https://shap.readthedocs.io
- tsaug — Time-Series Augmentation: https://tsaug.readthedocs.io
- Phi-3-mini — Microsoft: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
- Ollama: https://ollama.com
- Baldi et al. (2021) — Solar Flare Prediction with CNNs
- Ahmadzadeh et al. (2021) — How to Train a Flare Forecasting Model
- Neupert (1968) — Comparison of solar X-ray bursts with microwave bursts

---

*Built by Quantum Crew — ISRO Aditya-L1 Hackathon PS15*
