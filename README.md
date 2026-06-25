# ☀️ SolarSentinel — Solar Flare Nowcasting & Forecasting System
### ISRO Aditya-L1 | SoLEXS + HEL1OS | PS15 | Quantum Crew

> Automated detection, classification, and prediction of solar flares using combined Soft and Hard X-ray time-series data from ISRO's Aditya-L1 mission.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Milestone Map](#milestone-map)
- [M0 — Environment & Setup](#m0--environment--setup)
- [M1 — Data Pipeline](#m1--data-pipeline)
- [M2 — Nowcasting Pipeline (TCN + XGBoost)](#m2--nowcasting-pipeline-tcn--xgboost)
- [M3 — Forecasting Pipeline (BiLSTM + TCN Ensemble)](#m3--forecasting-pipeline-bilstm--tcn-ensemble)
- [M3.5 — Agent Orchestration (LangGraph)](#m35--agent-orchestration-langgraph)
- [M3.6 — AutoML Tuning (FLAML)](#m36--automl-tuning-flaml)
- [M3.7 — LLM Alert Intelligence (Phi-3-mini / Mistral)](#m37--llm-alert-intelligence-phi-3-mini--mistral)
- [M4 — Dashboard & Visualization](#m4--dashboard--visualization)
- [M5 — Evaluation & Optimization](#m5--evaluation--optimization)
- [M6 — Production Deployment](#m6--production-deployment)
- [Project Structure](#project-structure)
- [Evaluation Metrics](#evaluation-metrics)
- [References](#references)

---

## Project Overview

Solar flares are sudden, intense bursts of radiation from magnetic energy release in the solar atmosphere. M and X class flares disrupt satellite communications, GPS navigation, and power grids.

**SolarSentinel** builds two independent ML pipelines:

| Pipeline | Target | Model Pair | Output |
|---|---|---|---|
| **Nowcasting** | Detect flare as it happens | TCN + XGBoost | Class label + confidence |
| **Forecasting** | Predict flare N minutes ahead | BiLSTM + TCN Ensemble | P(flare) + lead time |

Both pipelines consume real-time SoLEXS (soft X-ray) and HEL1OS (hard X-ray) light curves from Aditya-L1, available via ISRO's PRADAN portal.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     PRADAN Portal (ISSDC)                        │
│          SoLEXS Level-1 FITS    HEL1OS Level-1 FITS             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              LANGGRAPH AGENT ORCHESTRATION LAYER                 │
│                                                                  │
│  [Ingestion Agent] → [Preprocess Agent] → [Router Agent]        │
│                                                ↓         ↓      │
│                                         [Nowcast]  [Forecast]   │
│                                                ↓         ↓      │
│                                         [Alert Agent]           │
│                                                ↓                │
│                                         [LLM Report Agent]      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Data Ingestion Layer                          │
│   FITS Reader → Calibration → Resampling → Background Sub      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
┌──────────────────────┐    ┌──────────────────────────────┐
│  NOWCASTING PIPELINE │    │    FORECASTING PIPELINE       │
│                      │    │                               │
│  Raw Flux Window     │    │  Rolling Sequence [T-60→T]   │
│       │              │    │         │                     │
│  [TCN Encoder]       │    │  ┌──────┴──────┐             │
│  - Dilated conv      │    │  ▼             ▼             │
│  - Multi-scale       │    │ [BiLSTM]    [TCN]            │
│  - 128-dim vector    │    │  │             │             │
│       │              │    │  └──────┬──────┘             │
│  + Handcrafted feats │    │         ▼                     │
│  - dF/dt             │    │   Weighted Ensemble           │
│  - channel ratio     │    │   P(flare in next N min)     │
│  - rolling std       │    │   Lead Time Estimate          │
│  - spectral index    │    │                               │
│       │              │    └──────────────┬────────────────┘
│  [XGBoost + FLAML]   │                   │
│  - Flare / No Flare  │                   │
│  - Class: C/M/X      │                   │
└────────┬─────────────┘                   │
         │                                 │
         └──────────────┬──────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                  LLM INTELLIGENCE LAYER                          │
│         Phi-3-mini / Mistral 7B (runs locally via Ollama)       │
│   "M-class flare at 10:33 UTC. Grid operators: action advised." │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              Master Flare Catalogue + Alert Engine              │
│          Unified Dashboard (FastAPI + React/Streamlit)           │
│    Light curve | Alert overlay | Lead time | LLM report        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data I/O | `astropy`, `sunpy`, `numpy` |
| Preprocessing | `pandas`, `scipy`, `sklearn` |
| Nowcasting ML | `pytorch` (TCN), `xgboost` |
| Forecasting ML | `pytorch` (BiLSTM + TCN) |
| AutoML Tuning | `flaml` (XGBoost hyperparameter search) |
| Agent Orchestration | `langgraph` (stateful multi-agent pipeline) |
| LLM Intelligence | `Phi-3-mini` / `Mistral 7B` via `ollama` |
| Experiment Tracking | `mlflow` |
| Backend API | `FastAPI` |
| Frontend Dashboard | `React + TypeScript` or `Streamlit` |
| Containerization | `Docker`, `docker-compose` |
| Testing | `pytest` |
| CI/CD | `GitHub Actions` |
| Monitoring | `Prometheus + Grafana` |

---

## Milestone Map

```
M0 ──► M1 ──► M2 ──► M3 ──► M3.5 ──► M3.6 ──► M3.7 ──► M4 ──► M5 ──► M6
Setup  Data   Now-   Fore-  Lang-    FLAML    LLM      Dash-  Eval   Prod
       Pipe   cast   cast   Graph    AutoML   Alerts   board  &Opt   Ready
```

| Milestone | Deliverable | Est. Time |
|---|---|---|
| M0 | Env, repo, data access | Day 1–2 |
| M1 | FITS pipeline, labeled dataset | Day 3–5 |
| M2 | Nowcasting model, evaluated | Day 6–9 |
| M3 | Forecasting model, evaluated | Day 10–13 |
| M3.5 | LangGraph agent orchestration | Day 14 |
| M3.6 | FLAML XGBoost auto-tuning | Day 15 |
| M3.7 | LLM alert report generation | Day 16 |
| M4 | Dashboard live with alerts | Day 17–18 |
| M5 | Optimized, benchmarked, documented | Day 19–20 |
| M6 | Dockerized, CI/CD, monitored, deployed | Day 21 |

---

## M0 — Environment & Setup

### 0.1 Repository Structure (initialize this first)

```bash
git init solar-sentinel
cd solar-sentinel
```

```
solar-sentinel/
├── data/
│   ├── raw/              # Downloaded FITS files
│   ├── processed/        # Parquet/numpy arrays
│   └── labels/           # NOAA/GOES flare catalog CSVs
├── src/
│   ├── ingestion/        # FITS readers, PRADAN download scripts
│   ├── preprocessing/    # Calibration, resampling, feature engineering
│   ├── nowcasting/       # TCN encoder + XGBoost pipeline
│   ├── forecasting/      # BiLSTM + TCN ensemble pipeline
│   ├── evaluation/       # TSS, TPR, FAR, lead time metrics
│   └── api/              # FastAPI backend
├── dashboard/            # React frontend or Streamlit app
├── notebooks/            # EDA, prototyping
├── models/               # Saved model checkpoints
├── configs/              # YAML config files
├── tests/                # pytest test suite
├── docker/               # Dockerfiles
├── .github/workflows/    # CI/CD pipelines
├── requirements.txt
├── docker-compose.yml
└── README.md
```

### 0.2 Python Environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install astropy sunpy numpy pandas scipy scikit-learn \
            torch xgboost flaml mlflow fastapi uvicorn \
            langgraph langchain-core \
            requests pytest httpx python-dotenv pyyaml \
            plotly matplotlib seaborn
```

### 0.3 Data Access — PRADAN Portal

1. Register at `https://pradan.issdc.gov.in`
2. Navigate → Aditya-L1 → SoLEXS Level-1
3. Navigate → Aditya-L1 → HEL1OS Level-1
4. Download FITS files for known flare periods (cross-reference dates below)

**Known high-value flare dates for training data:**

| Date | Class | Notes |
|---|---|---|
| 2024-02-22 | X6.3 | First flare imaged by SUIT+SoLEXS+HEL1OS |
| 2024-05-10 | X5.8 | Major CME event, ASPEX detected |
| 2024-10-03 | M-class series | Solar wind impact study event |

### 0.4 Supplementary Data Sources

```bash
# GOES XRS light curves — gold standard labels
# https://www.ngdc.noaa.gov/stp/solar/solarflares.html

# NOAA SWPC Flare Event Catalog
wget https://ftp.swpc.noaa.gov/pub/indices/events/

# GOES-R XRS NetCDF via SunPy
python -c "from sunpy.net import Fido, attrs as a; \
           import astropy.units as u; \
           result = Fido.search(a.Time('2024-02-22','2024-02-23'), \
                               a.Instrument('XRS')); \
           Fido.fetch(result)"
```

---

## M1 — Data Pipeline

### 1.1 FITS Reader

```python
# src/ingestion/fits_reader.py

from astropy.io import fits
import numpy as np
import pandas as pd

def read_solexs(filepath: str) -> pd.DataFrame:
    """
    Read SoLEXS Level-1 FITS file.
    Returns DataFrame with columns: [time, flux_low, flux_mid, flux_high]
    Energy bands: ~1-3 keV (low), 3-6 keV (mid), 6-15 keV (high)
    """
    with fits.open(filepath) as hdul:
        # Inspect structure first time: hdul.info()
        data = hdul[1].data
        df = pd.DataFrame({
            'time': data['TIME'],         # seconds from reference epoch
            'flux_low': data['FLUX_1'],   # adjust column names per user manual
            'flux_mid': data['FLUX_2'],
            'flux_high': data['FLUX_3'],
        })
    df['time'] = pd.to_datetime(df['time'], unit='s', origin='unix')
    return df.set_index('time').sort_index()


def read_hel1os(filepath: str) -> pd.DataFrame:
    """
    Read HEL1OS Level-1 FITS file.
    Returns DataFrame with columns: [time, counts_low, counts_high]
    Energy bands: ~10-100 keV (low), 100keV-1MeV (high)
    """
    with fits.open(filepath) as hdul:
        data = hdul[1].data
        df = pd.DataFrame({
            'time': data['TIME'],
            'counts_low': data['COUNTS_1'],
            'counts_high': data['COUNTS_2'],
        })
    df['time'] = pd.to_datetime(df['time'], unit='s', origin='unix')
    return df.set_index('time').sort_index()


def merge_instruments(solexs_df: pd.DataFrame,
                      hel1os_df: pd.DataFrame,
                      cadence: str = '1min') -> pd.DataFrame:
    """
    Resample both instruments to common cadence and merge.
    """
    solexs_resampled = solexs_df.resample(cadence).mean()
    hel1os_resampled = hel1os_df.resample(cadence).mean()
    merged = pd.concat([solexs_resampled, hel1os_resampled], axis=1).dropna()
    return merged
```

### 1.2 Preprocessing

```python
# src/preprocessing/pipeline.py

import pandas as pd
import numpy as np
from scipy.ndimage import uniform_filter1d

def subtract_background(df: pd.DataFrame,
                         window_minutes: int = 10) -> pd.DataFrame:
    """
    Rolling median background subtraction.
    Background = median of pre-event quiet period.
    """
    df_clean = df.copy()
    for col in df.columns:
        rolling_median = df[col].rolling(
            window=window_minutes, min_periods=1, center=False
        ).median()
        df_clean[col] = df[col] - rolling_median
        df_clean[col] = df_clean[col].clip(lower=0)  # no negative flux
    return df_clean


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hand-crafted physics-informed features for XGBoost.
    """
    df = df.copy()

    # 1. Time derivatives — rate of flux change
    for col in df.columns:
        df[f'd_{col}_dt'] = df[col].diff() / 1.0  # per minute

    # 2. Channel ratio soft/hard — spectral hardness indicator
    df['channel_ratio'] = (
        df['flux_high'] / (df['counts_low'] + 1e-10)
    )

    # 3. Rolling statistics
    for col in ['flux_low', 'flux_mid', 'flux_high', 'counts_low', 'counts_high']:
        df[f'{col}_rollstd_5'] = df[col].rolling(5).std()
        df[f'{col}_rollstd_15'] = df[col].rolling(15).std()
        df[f'{col}_rollmax_5'] = df[col].rolling(5).max()

    # 4. Spectral index proxy
    # slope of log(flux) vs log(energy) — hardens before M/X flares
    df['spectral_index'] = np.log(df['flux_high'] + 1e-10) - \
                           np.log(df['flux_low'] + 1e-10)

    return df.dropna()


def build_labeled_dataset(merged_df: pd.DataFrame,
                            flare_catalog: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-reference NOAA flare catalog to assign labels.
    flare_catalog columns: [start_time, peak_time, end_time, class]
    """
    merged_df = merged_df.copy()
    merged_df['label'] = 0          # 0 = no flare
    merged_df['flare_class'] = 'N'  # N = None

    class_map = {'A': 1, 'B': 2, 'C': 3, 'M': 4, 'X': 5}

    for _, flare in flare_catalog.iterrows():
        mask = (
            (merged_df.index >= flare['start_time']) &
            (merged_df.index <= flare['end_time'])
        )
        merged_df.loc[mask, 'label'] = 1
        merged_df.loc[mask, 'flare_class'] = flare['class'][0]

    return merged_df
```

### 1.3 Sliding Window Generator

```python
# src/preprocessing/windows.py

import numpy as np

def create_windows(df, feature_cols: list,
                   window_size: int = 60,
                   horizon: int = 15,
                   step: int = 1):
    """
    Creates sliding windows for sequence models.

    Args:
        window_size: lookback in minutes
        horizon: forecast horizon in minutes (for forecasting pipeline)
        step: stride between windows

    Returns:
        X: (N, window_size, n_features)
        y_now: (N,) — label at current timestep (nowcasting)
        y_fore: (N,) — label at t+horizon (forecasting)
    """
    X, y_now, y_fore = [], [], []
    data = df[feature_cols].values
    labels = df['label'].values

    for i in range(0, len(data) - window_size - horizon, step):
        X.append(data[i:i + window_size])
        y_now.append(labels[i + window_size - 1])
        y_fore.append(labels[i + window_size + horizon - 1])

    return (np.array(X, dtype=np.float32),
            np.array(y_now, dtype=np.int64),
            np.array(y_fore, dtype=np.int64))
```

**M1 Checkpoint:** You should have a clean `processed/dataset.parquet` with shape `(N_timesteps, n_features+2_labels)` and class distribution logged.

---

## M2 — Nowcasting Pipeline (TCN + XGBoost)

### 2.1 TCN Encoder (PyTorch)

```python
# src/nowcasting/tcn_encoder.py

import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding
        )

    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.padding]  # remove future leakage


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(out_ch)
        self.norm2 = nn.LayerNorm(out_ch)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        # x: (B, C, T)
        residual = x
        out = self.relu(self.norm1(self.conv1(x).transpose(1,2)).transpose(1,2))
        out = self.dropout(out)
        out = self.norm2(self.conv2(out).transpose(1,2)).transpose(1,2)
        out = self.dropout(out)
        if self.downsample:
            residual = self.downsample(residual)
        return self.relu(out + residual)


class TCNEncoder(nn.Module):
    """
    TCN that encodes a window of light curve data
    into a fixed-size feature vector for XGBoost.
    """
    def __init__(self, n_features: int, embed_dim: int = 128,
                 n_layers: int = 4, kernel_size: int = 3):
        super().__init__()
        channels = [n_features] + [embed_dim] * n_layers
        dilations = [2 ** i for i in range(n_layers)]

        self.blocks = nn.ModuleList([
            TCNBlock(channels[i], channels[i+1], kernel_size, dilations[i])
            for i in range(n_layers)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (B, T, F) → (B, F, T) for Conv1d
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = self.pool(x).squeeze(-1)  # (B, embed_dim)
        return x
```

### 2.2 Feature Extraction + XGBoost Classifier

```python
# src/nowcasting/train.py

import numpy as np
import torch
import xgboost as xgb
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def extract_tcn_features(encoder: TCNEncoder,
                          X: np.ndarray,
                          device: str = 'cpu') -> np.ndarray:
    encoder.eval()
    encoder.to(device)
    features = []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i+batch_size]).to(device)
            feat = encoder(batch).cpu().numpy()
            features.append(feat)
    return np.vstack(features)


def train_nowcasting_pipeline(X: np.ndarray,
                               y: np.ndarray,
                               handcrafted_features: np.ndarray):
    """
    Full nowcasting training:
    1. Pretrain TCN encoder (autoencoder or supervised)
    2. Extract TCN embeddings
    3. Concatenate with handcrafted features
    4. Train XGBoost on combined features
    """
    with mlflow.start_run(run_name="nowcasting_tcn_xgb"):

        # 1. Train TCN encoder
        encoder = TCNEncoder(n_features=X.shape[2], embed_dim=128)
        # (train encoder — see autoencoder or supervised pretraining below)

        # 2. Extract embeddings
        tcn_feats = extract_tcn_features(encoder, X)

        # 3. Combine
        combined = np.concatenate([tcn_feats, handcrafted_features], axis=1)

        # 4. Train/val split (time-aware — NO random shuffle)
        split = int(0.8 * len(combined))
        X_train, X_val = combined[:split], combined[split:]
        y_train, y_val = y[:split], y[split:]

        # 5. Class weights for imbalance
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        scale_pos_weight = neg / pos

        # 6. XGBoost
        model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric='logloss',
            early_stopping_rounds=50,
            tree_method='hist'
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100
        )

        # 7. Log to MLflow
        mlflow.log_param("embed_dim", 128)
        mlflow.log_param("xgb_depth", 6)
        mlflow.xgboost.log_model(model, "nowcast_xgb")

        return encoder, model
```

**M2 Checkpoint:** Nowcasting pipeline runs end-to-end. TSS > 0.5 on validation set. Feature importance chart generated.

---

## M3 — Forecasting Pipeline (BiLSTM + TCN Ensemble)

### 3.1 BiLSTM Model

```python
# src/forecasting/bilstm.py

import torch
import torch.nn as nn

class BiLSTMForecaster(nn.Module):
    """
    Bidirectional LSTM for flare probability forecasting.
    Input:  (B, T, F) — window of T minutes, F features
    Output: P(flare in next N minutes)
    """
    def __init__(self, n_features: int, hidden_dim: int = 128,
                 n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]      # take last timestep
        last = self.norm(last)
        return self.head(last).squeeze(-1)


class TCNForecaster(nn.Module):
    """TCN repurposed for forecasting output."""
    def __init__(self, n_features: int, embed_dim: int = 128, n_layers: int = 4):
        super().__init__()
        self.encoder = TCNEncoder(n_features, embed_dim, n_layers)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.head(features).squeeze(-1)
```

### 3.2 Ensemble Trainer

```python
# src/forecasting/ensemble.py

import torch
import torch.nn as nn
import numpy as np
import mlflow

class WeightedEnsemble(nn.Module):
    """
    Learnable weighted average of BiLSTM and TCN outputs.
    w1 * P_bilstm + w2 * P_tcn, where w1 + w2 = 1
    """
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.tensor([0.5, 0.5]))

    def forward(self, p_bilstm, p_tcn):
        w = torch.softmax(self.weights, dim=0)
        return w[0] * p_bilstm + w[1] * p_tcn


def train_forecasting_ensemble(X_train, y_fore_train,
                                X_val, y_fore_val,
                                n_features: int,
                                epochs: int = 100,
                                horizon_minutes: int = 15):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    bilstm = BiLSTMForecaster(n_features).to(device)
    tcn_fore = TCNForecaster(n_features).to(device)
    ensemble = WeightedEnsemble().to(device)

    all_params = (
        list(bilstm.parameters()) +
        list(tcn_fore.parameters()) +
        list(ensemble.parameters())
    )
    optimizer = torch.optim.AdamW(all_params, lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Focal loss for imbalance
    def focal_loss(pred, target, gamma=2.0, alpha=0.75):
        bce = nn.functional.binary_cross_entropy(pred, target.float(), reduction='none')
        pt = torch.exp(-bce)
        return (alpha * (1 - pt) ** gamma * bce).mean()

    X_tr = torch.tensor(X_train, device=device)
    y_tr = torch.tensor(y_fore_train, device=device)

    with mlflow.start_run(run_name=f"forecasting_ensemble_h{horizon_minutes}"):
        for epoch in range(epochs):
            bilstm.train(); tcn_fore.train(); ensemble.train()

            p_bi = bilstm(X_tr)
            p_tc = tcn_fore(X_tr)
            p_ens = ensemble(p_bi, p_tc)

            loss = focal_loss(p_ens, y_tr)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            scheduler.step()

            if epoch % 10 == 0:
                mlflow.log_metric("train_loss", loss.item(), step=epoch)

        mlflow.log_param("horizon_minutes", horizon_minutes)
        mlflow.pytorch.log_model(bilstm, "bilstm")
        mlflow.pytorch.log_model(tcn_fore, "tcn_forecaster")

    return bilstm, tcn_fore, ensemble
```

### 3.3 Lead Time Computation

```python
# src/forecasting/lead_time.py

def compute_lead_time(y_pred_proba: np.ndarray,
                       y_true: np.ndarray,
                       timestamps: pd.DatetimeIndex,
                       threshold: float = 0.5) -> dict:
    """
    For each true flare event, find how many minutes
    before flare onset the model first triggered an alert.
    """
    results = []
    alert_times = timestamps[y_pred_proba >= threshold]
    flare_onsets = timestamps[np.diff(y_true.astype(int), prepend=0) == 1]

    for onset in flare_onsets:
        prior_alerts = alert_times[alert_times < onset]
        if len(prior_alerts) > 0:
            lead = (onset - prior_alerts[-1]).total_seconds() / 60
            results.append({'onset': onset, 'lead_minutes': lead, 'caught': True})
        else:
            results.append({'onset': onset, 'lead_minutes': 0, 'caught': False})

    return {
        'mean_lead_minutes': np.mean([r['lead_minutes'] for r in results]),
        'max_lead_minutes': np.max([r['lead_minutes'] for r in results]),
        'detection_rate': np.mean([r['caught'] for r in results]),
        'per_event': results
    }
```

**M3 Checkpoint:** Forecasting ensemble trained for horizons N=15, 30, 60 min. TSS and lead time logged per horizon.

---

## M3.5 — Agent Orchestration (LangGraph)

LangGraph converts your isolated scripts into a **stateful, fault-tolerant agent pipeline**. Each node is an agent. Edges define routing logic. If any agent fails, the graph retries or routes to a fallback.

### 3.5.1 Install

```bash
pip install langgraph langchain-core
```

### 3.5.2 Define Agent State

```python
# src/orchestration/state.py

from typing import TypedDict, Optional
import pandas as pd
import numpy as np

class SolarPipelineState(TypedDict):
    # Ingestion
    solexs_path: str
    hel1os_path: str
    raw_df: Optional[pd.DataFrame]

    # Preprocessing
    processed_df: Optional[pd.DataFrame]
    feature_matrix: Optional[np.ndarray]
    windows_X: Optional[np.ndarray]

    # Nowcasting
    nowcast_class: Optional[str]        # C / M / X / None
    nowcast_confidence: Optional[float]

    # Forecasting
    forecast_proba: Optional[float]
    forecast_horizon: Optional[int]
    lead_time_minutes: Optional[float]

    # Intelligence
    alert_triggered: bool
    llm_report: Optional[str]

    # Meta
    errors: list[str]
    timestamp: str
```

### 3.5.3 Define Agent Nodes

```python
# src/orchestration/agents.py

from langgraph.graph import StateGraph, END
from src.ingestion.fits_reader import read_solexs, read_hel1os, merge_instruments
from src.preprocessing.pipeline import subtract_background, engineer_features
from src.nowcasting.train import run_nowcast_inference
from src.forecasting.ensemble import run_forecast_inference
from src.intelligence.llm_reporter import generate_flare_report


def ingestion_agent(state: SolarPipelineState) -> SolarPipelineState:
    """Reads and merges SoLEXS + HEL1OS FITS files."""
    try:
        solexs = read_solexs(state["solexs_path"])
        hel1os = read_hel1os(state["hel1os_path"])
        merged = merge_instruments(solexs, hel1os)
        return {**state, "raw_df": merged}
    except Exception as e:
        return {**state, "errors": state["errors"] + [f"Ingestion: {e}"]}


def preprocessing_agent(state: SolarPipelineState) -> SolarPipelineState:
    """Background subtraction + feature engineering."""
    try:
        clean = subtract_background(state["raw_df"])
        features = engineer_features(clean)
        return {**state, "processed_df": features}
    except Exception as e:
        return {**state, "errors": state["errors"] + [f"Preprocessing: {e}"]}


def nowcast_agent(state: SolarPipelineState) -> SolarPipelineState:
    """Runs TCN + XGBoost nowcasting inference."""
    cls, conf = run_nowcast_inference(state["processed_df"])
    return {**state, "nowcast_class": cls, "nowcast_confidence": conf}


def forecast_agent(state: SolarPipelineState) -> SolarPipelineState:
    """Runs BiLSTM + TCN ensemble forecasting inference."""
    proba, lead = run_forecast_inference(state["processed_df"], horizon=15)
    return {**state, "forecast_proba": proba, "lead_time_minutes": lead}


def alert_router(state: SolarPipelineState) -> str:
    """
    Conditional edge: decides whether to trigger LLM report.
    Returns next node name.
    """
    if state["nowcast_class"] in ["M", "X"] or state["forecast_proba"] > 0.75:
        return "llm_report_agent"
    return "end"


def llm_report_agent(state: SolarPipelineState) -> SolarPipelineState:
    """Generates natural language alert using local LLM."""
    report = generate_flare_report({
        "class": state["nowcast_class"],
        "confidence": state["nowcast_confidence"],
        "forecast_proba": state["forecast_proba"],
        "lead_time": state["lead_time_minutes"],
        "timestamp": state["timestamp"]
    })
    return {**state, "alert_triggered": True, "llm_report": report}
```

### 3.5.4 Build and Compile the Graph

```python
# src/orchestration/graph.py

from langgraph.graph import StateGraph, END
from src.orchestration.agents import (
    ingestion_agent, preprocessing_agent,
    nowcast_agent, forecast_agent,
    alert_router, llm_report_agent
)
from src.orchestration.state import SolarPipelineState

def build_pipeline() -> StateGraph:
    graph = StateGraph(SolarPipelineState)

    # Add nodes
    graph.add_node("ingestion",      ingestion_agent)
    graph.add_node("preprocessing",  preprocessing_agent)
    graph.add_node("nowcast",        nowcast_agent)
    graph.add_node("forecast",       forecast_agent)
    graph.add_node("llm_report",     llm_report_agent)

    # Linear edges
    graph.set_entry_point("ingestion")
    graph.add_edge("ingestion",     "preprocessing")
    graph.add_edge("preprocessing", "nowcast")
    graph.add_edge("nowcast",       "forecast")

    # Conditional edge — only call LLM if alert warranted
    graph.add_conditional_edges(
        "forecast",
        alert_router,
        {
            "llm_report_agent": "llm_report",
            "end": END
        }
    )
    graph.add_edge("llm_report", END)

    return graph.compile()


# Run the full pipeline
if __name__ == "__main__":
    pipeline = build_pipeline()
    result = pipeline.invoke({
        "solexs_path": "data/raw/solexs/2024-02-22.fits",
        "hel1os_path": "data/raw/hel1os/2024-02-22.fits",
        "errors": [],
        "alert_triggered": False,
        "timestamp": "2024-02-22T10:33:00Z"
    })
    print("Nowcast:", result["nowcast_class"], result["nowcast_confidence"])
    print("Forecast P:", result["forecast_proba"])
    print("LLM Report:", result["llm_report"])
```

**M3.5 Checkpoint:** Full pipeline runs end-to-end via `python src/orchestration/graph.py`. Conditional alert routing verified on known flare FITS files.

---

## M3.6 — AutoML Tuning (FLAML)

FLAML replaces manual XGBoost hyperparameter search. Give it a time budget — it finds the optimal config automatically.

### 3.6.1 Install

```bash
pip install flaml
```

### 3.6.2 Auto-tune XGBoost Nowcasting Classifier

```python
# src/nowcasting/flaml_tuner.py

from flaml import AutoML
import numpy as np
import mlflow

def autotune_nowcast(X_train: np.ndarray,
                     y_train: np.ndarray,
                     X_val: np.ndarray,
                     y_val: np.ndarray,
                     time_budget_seconds: int = 3600) -> AutoML:
    """
    FLAML searches over XGBoost, LightGBM, RandomForest, ExtraTrees.
    Optimizes for ROC-AUC. Returns best model ready for inference.
    """
    automl = AutoML()

    with mlflow.start_run(run_name="flaml_autotune_nowcast"):
        automl.fit(
            X_train, y_train,
            task="classification",
            metric="roc_auc",
            time_budget=time_budget_seconds,
            estimator_list=["xgboost", "lgbm", "rf", "extra_tree"],
            eval_method="holdout",
            X_val=X_val,
            y_val=y_val,
            log_file_name="logs/flaml_nowcast.log",
            verbose=2
        )

        mlflow.log_param("best_estimator", automl.best_estimator)
        mlflow.log_params(automl.best_config)
        mlflow.log_metric("best_val_auc", 1 - automl.best_loss)

        print(f"Best model:  {automl.best_estimator}")
        print(f"Best config: {automl.best_config}")
        print(f"Val AUC:     {1 - automl.best_loss:.4f}")

    return automl


def autotune_forecast(X_train, y_train, X_val, y_val,
                      time_budget_seconds: int = 1800) -> AutoML:
    """Same FLAML search for the ensemble feature layer."""
    automl = AutoML()
    with mlflow.start_run(run_name="flaml_autotune_forecast"):
        automl.fit(
            X_train, y_train,
            task="classification",
            metric="roc_auc",
            time_budget=time_budget_seconds,
            estimator_list=["xgboost", "lgbm"],
            X_val=X_val,
            y_val=y_val
        )
    return automl
```

**What FLAML searches over automatically:**

| Parameter | Search Range |
|---|---|
| `n_estimators` | 4 – 32768 |
| `max_depth` | 4 – 12 |
| `learning_rate` | 0.001 – 1.0 |
| `subsample` | 0.5 – 1.0 |
| `colsample_bytree` | 0.5 – 1.0 |
| `reg_alpha / lambda` | 1e-10 – 1.0 |

**M3.6 Checkpoint:** FLAML log saved at `logs/flaml_nowcast.log`. Best config printed and saved to MLflow. Retrain XGBoost with found params.

---

## M3.7 — LLM Alert Intelligence (Phi-3-mini / Mistral)

This is your **differentiator**. No other PS15 team will have it. When an alert fires, a local LLM generates a plain-English operational bulletin — automatically.

### 3.7.1 Setup Ollama (Local LLM Runtime)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull models (choose one)
ollama pull phi3:mini        # Microsoft Phi-3-mini — 3.8B, fast, accurate
ollama pull mistral:7b       # Mistral 7B — larger, richer output

# Verify
ollama run phi3:mini "Hello"
```

### 3.7.2 LLM Report Generator

```python
# src/intelligence/llm_reporter.py

import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "phi3:mini"   # swap to "mistral:7b" for richer output

SYSTEM_PROMPT = """You are an expert solar physicist and space weather operator assistant.
Your job is to generate concise, actionable operational bulletins when solar flares are detected.
Always include: event classification, estimated impact severity, affected systems, and recommended actions.
Keep the bulletin under 5 sentences. Be precise and professional."""

def generate_flare_report(flare_data: dict) -> str:
    """
    Generates a natural language alert bulletin using local LLM.

    Args:
        flare_data: dict with keys: class, confidence, forecast_proba,
                    lead_time, timestamp, peak_flux (optional)
    Returns:
        str: Operational bulletin text
    """
    user_prompt = f"""
    Solar flare event detected by SolarSentinel (Aditya-L1 SoLEXS + HEL1OS):

    - Detection time:      {flare_data.get('timestamp', 'N/A')}
    - Nowcast class:       {flare_data.get('class', 'Unknown')}
    - Nowcast confidence:  {flare_data.get('confidence', 0):.0%}
    - Forecast probability:{flare_data.get('forecast_proba', 0):.0%} (next 15 min)
    - Estimated lead time: {flare_data.get('lead_time', 0):.1f} minutes before peak

    Generate a space weather operational bulletin for satellite operators and grid managers.
    """

    payload = {
        "model": MODEL,
        "system": SYSTEM_PROMPT,
        "prompt": user_prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,   # low temp = factual, consistent
            "num_predict": 200
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        result = response.json()
        return result.get("response", "LLM unavailable — manual review required.")
    except Exception as e:
        return f"Alert: {flare_data.get('class', '?')}-class flare detected. LLM offline: {e}"


def generate_quiet_summary(stats: dict) -> str:
    """Generates a daily quiet-sun summary report."""
    prompt = f"""
    No significant flares detected in the past 24 hours.
    Background flux level: {stats.get('mean_flux', 'N/A')} W/m²
    Max flux observed:     {stats.get('max_flux', 'N/A')} W/m²
    Solar activity index:  {stats.get('activity_index', 'Low')}

    Write a 2-sentence quiet-sun daily summary for space weather operators.
    """
    payload = {"model": MODEL, "prompt": prompt, "stream": False}
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=30)
        return r.json().get("response", "Quiet sun. No action required.")
    except:
        return "Quiet sun conditions. No significant activity detected."
```

### 3.7.3 Example Output

When an M-class flare fires, the dashboard renders:

```
┌─────────────────────────────────────────────────────────────────┐
│  🔴 SOLAR FLARE ALERT — M2.3 CLASS                              │
│  Detected: 2024-02-22 10:33 UTC | Lead Time: 18 min             │
├─────────────────────────────────────────────────────────────────┤
│  OPERATIONAL BULLETIN (auto-generated by Phi-3-mini)            │
│                                                                  │
│  "An M2.3-class solar flare has been detected at 10:33 UTC      │
│  via Aditya-L1 SoLEXS and HEL1OS instruments. This event        │
│  poses moderate risk to HF radio communications on the           │
│  sunlit side of Earth and may cause minor GPS degradation.       │
│  Satellite operators should activate anomaly monitoring          │
│  protocols. Power grid operators in high-latitude regions        │
│  are advised to monitor GIC levels for the next 2 hours."       │
└─────────────────────────────────────────────────────────────────┘
```

### 3.7.4 Severity-Aware Prompting

```python
SEVERITY_CONTEXT = {
    "C": "minor event. Low risk to infrastructure.",
    "M": "moderate event. HF radio disruption likely. Satellite precautions advised.",
    "X": "MAJOR EVENT. Severe radio blackouts. GPS degradation. Power grid GIC risk. IMMEDIATE action required."
}

def get_severity_context(flare_class: str) -> str:
    prefix = flare_class[0].upper() if flare_class else "C"
    return SEVERITY_CONTEXT.get(prefix, "unknown severity.")
```

**M3.7 Checkpoint:** `ollama run phi3:mini` responding locally. `generate_flare_report()` returns bulletin in < 3 seconds. Dashboard renders LLM text block on alert trigger.

---

## M4 — Dashboard & Visualization

### 4.1 FastAPI Backend

```python
# src/api/main.py

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import asyncio, json, numpy as np

app = FastAPI(title="SolarSentinel API")

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/api/flare-catalog")
def get_flare_catalog():
    """Return detected/predicted flare events."""
    # Load from database or CSV
    return {"flares": load_master_catalog()}

@app.get("/api/lightcurve")
def get_lightcurve(start: str, end: str, instrument: str = "both"):
    """Return SoLEXS + HEL1OS light curve data for time range."""
    df = load_lightcurve(start, end, instrument)
    return df.reset_index().to_dict(orient="records")

@app.get("/api/nowcast/latest")
def get_nowcast():
    """Return latest nowcast classification."""
    return {"class": "M", "confidence": 0.87, "timestamp": "2024-02-22T10:33:00Z"}

@app.get("/api/forecast")
def get_forecast(horizon: int = 15):
    """Return flare probability for next N minutes."""
    return {"horizon_minutes": horizon, "probability": 0.73,
            "alert": True, "estimated_class": "M"}

@app.websocket("/ws/live")
async def live_stream(websocket: WebSocket):
    """WebSocket for real-time light curve + alert streaming."""
    await websocket.accept()
    try:
        while True:
            data = get_latest_reading()   # from ingestion layer
            await websocket.send_json(data)
            await asyncio.sleep(60)       # 1-minute cadence
    except Exception:
        await websocket.close()
```

### 4.2 Dashboard Features (React)

```
Dashboard Layout:
┌─────────────────────────────────────────────────────┐
│  ☀️ SolarSentinel          [LIVE] [REPLAY] [HISTORY] │
├─────────────────────────────────────────────────────┤
│                                                     │
│  SoLEXS Light Curve          HEL1OS Light Curve     │
│  [Plotly time series]        [Plotly time series]   │
│  - flux bands overlaid       - counts overlaid      │
│  - flare markers (▲)         - alert threshold line  │
│                                                     │
├──────────────────────┬──────────────────────────────┤
│  NOWCAST STATUS      │  FORECAST STATUS             │
│  ┌────────────────┐  │  ┌──────────────────────┐   │
│  │ 🟡 C-CLASS     │  │  │ P(flare|15min): 73%  │   │
│  │ Conf: 87%      │  │  │ P(flare|30min): 61%  │   │
│  │ Onset: 10:33   │  │  │ Est. class: M         │   │
│  └────────────────┘  │  │ Lead time: ~18 min    │   │
│                       │  └──────────────────────┘   │
├──────────────────────┴──────────────────────────────┤
│  FLARE CATALOGUE                                    │
│  Date | Peak | Class | Conf | Lead Time | Source   │
│  2024-02-22 | 10:41 | X6.3 | 94% | 22 min | Both  │
└─────────────────────────────────────────────────────┘
```

**Alert System:**
- `P > 0.5` → Yellow warning banner
- `P > 0.75` → Orange alert + audio ping
- `P > 0.90` → Red CRITICAL alert + email webhook

**M4 Checkpoint:** Dashboard live on `localhost:3000`. WebSocket streaming working. Alert triggers verified on replay of known flare events.

---

## M5 — Evaluation & Optimization

### 5.1 Metrics Implementation

```python
# src/evaluation/metrics.py

import numpy as np
from sklearn.metrics import confusion_matrix

def compute_tss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    True Skill Score = TPR - FPR
    Range: [-1, 1]. > 0.5 = good. > 0.7 = publishable.
    Standard metric in operational solar flare forecasting.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn + 1e-10)
    fpr = fp / (fp + tn + 1e-10)
    return tpr - fpr


def compute_far(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """False Alarm Rate = FP / (FP + TN)"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn + 1e-10)


def full_report(y_true, y_pred, y_pred_proba, timestamps, horizon_minutes=None):
    from sklearn.metrics import classification_report, roc_auc_score
    print("=== SolarSentinel Evaluation Report ===")
    print(f"TSS:     {compute_tss(y_true, y_pred):.4f}")
    print(f"FAR:     {compute_far(y_true, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_true, y_pred_proba):.4f}")
    print(classification_report(y_true, y_pred,
                                  target_names=['No Flare', 'Flare']))
    if horizon_minutes:
        lt = compute_lead_time(y_pred_proba, y_true, timestamps)
        print(f"Mean Lead Time: {lt['mean_lead_minutes']:.1f} min")
        print(f"Detection Rate: {lt['detection_rate']*100:.1f}%")
```

### 5.2 Threshold Optimization

```python
def optimize_threshold(y_true, y_pred_proba,
                        optimize_for: str = 'tss') -> float:
    """
    Sweep thresholds 0.1–0.9, pick the one maximizing TSS.
    Do NOT use default 0.5 — flare prediction needs tuning.
    """
    best_thresh, best_score = 0.5, 0.0
    for t in np.arange(0.1, 0.95, 0.05):
        y_pred = (y_pred_proba >= t).astype(int)
        score = compute_tss(y_true, y_pred)
        if score > best_score:
            best_score, best_thresh = score, t
    return best_thresh, best_score
```

### 5.3 Class-wise Evaluation (Required by PS)

```python
def evaluate_per_class(y_true_class, y_pred_class):
    """
    Evaluate detection separately for A/B/C/M/X classes.
    Judges specifically check C-class detection.
    """
    classes = ['C', 'M', 'X']
    for cls in classes:
        mask = (y_true_class == cls) | (y_true_class == 'N')
        y_t = (y_true_class[mask] == cls).astype(int)
        y_p = (y_pred_class[mask] == cls).astype(int)
        print(f"Class {cls}: TSS={compute_tss(y_t,y_p):.3f}  "
              f"FAR={compute_far(y_t,y_p):.3f}")
```

**M5 Targets:**

| Metric | Minimum | Target |
|---|---|---|
| TSS (nowcasting) | 0.5 | > 0.7 |
| TSS (forecasting 15min) | 0.4 | > 0.6 |
| FAR | < 0.30 | < 0.15 |
| Mean Lead Time | 10 min | > 20 min |
| C-class detection | > 60% | > 80% |

---

## M6 — Production Deployment

### 6.1 Dockerization

```dockerfile
# docker/Dockerfile.api
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/
COPY configs/ ./configs/

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: "3.9"

services:
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - PRADAN_TOKEN=${PRADAN_TOKEN}
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - OLLAMA_URL=http://ollama:11434
    depends_on:
      - ollama

  dashboard:
    build:
      context: ./dashboard
    ports:
      - "3000:3000"
    depends_on:
      - api

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]   # remove if no GPU

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    command: mlflow server --host 0.0.0.0

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./docker/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3001:3000"
    depends_on:
      - prometheus

volumes:
  ollama_data:
```

### 6.2 CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
name: SolarSentinel CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v --tb=short

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: docker build -f docker/Dockerfile.api -t solar-sentinel-api .
      - run: docker compose config   # validate compose file
```

### 6.3 Tests

```python
# tests/test_pipeline.py

import pytest
import numpy as np
from src.preprocessing.pipeline import subtract_background, engineer_features
from src.evaluation.metrics import compute_tss, compute_far

def test_background_subtraction_no_negatives(sample_df):
    result = subtract_background(sample_df)
    assert (result >= 0).all().all()

def test_tss_perfect():
    y = np.array([0, 0, 1, 1])
    assert compute_tss(y, y) == pytest.approx(1.0)

def test_tss_random():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, 100)
    y_pred = rng.integers(0, 2, 100)
    tss = compute_tss(y_true, y_pred)
    assert -1.0 <= tss <= 1.0

def test_feature_engineering_no_nan(sample_df):
    result = engineer_features(sample_df)
    assert not result.isnull().any().any()
```

### 6.4 Monitoring

```yaml
# docker/prometheus.yml
scrape_configs:
  - job_name: 'solar-sentinel-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
    scrape_interval: 60s
```

Track in Grafana:
- Model inference latency (p50, p99)
- Alert trigger rate per hour
- FAR drift over time (model degradation signal)
- PRADAN data fetch success rate

---

## Project Structure

```
solar-sentinel/
├── data/
│   ├── raw/
│   │   ├── solexs/        # Raw FITS files from PRADAN
│   │   └── hel1os/        # Raw FITS files from PRADAN
│   ├── processed/
│   │   └── dataset.parquet
│   └── labels/
│       └── noaa_flare_catalog.csv
├── src/
│   ├── ingestion/
│   │   ├── fits_reader.py
│   │   └── pradan_downloader.py
│   ├── preprocessing/
│   │   ├── pipeline.py
│   │   └── windows.py
│   ├── nowcasting/
│   │   ├── tcn_encoder.py
│   │   ├── train.py
│   │   └── flaml_tuner.py        # FLAML auto-tuning
│   ├── forecasting/
│   │   ├── bilstm.py
│   │   ├── ensemble.py
│   │   └── lead_time.py
│   ├── orchestration/             # LangGraph agent pipeline
│   │   ├── state.py              # SolarPipelineState TypedDict
│   │   ├── agents.py             # All agent node functions
│   │   └── graph.py              # Graph builder + entry point
│   ├── intelligence/              # LLM alert layer
│   │   └── llm_reporter.py       # Phi-3-mini / Mistral via Ollama
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
│   │   │   └── LLMReport.tsx     # LLM bulletin display component
│   │   └── App.tsx
│   └── package.json
├── notebooks/
│   ├── 01_eda_fits_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_nowcasting_prototype.ipynb
│   ├── 04_forecasting_prototype.ipynb
│   └── 05_langgraph_pipeline_test.ipynb
├── models/
│   ├── tcn_encoder.pt
│   ├── xgb_nowcast.json          # FLAML-tuned XGBoost
│   ├── bilstm_forecaster.pt
│   └── tcn_forecaster.pt
├── logs/
│   └── flaml_nowcast.log         # FLAML search history
├── configs/
│   ├── nowcasting.yaml
│   ├── forecasting.yaml
│   └── llm.yaml                  # Ollama model + prompt config
├── tests/
│   └── test_pipeline.py
├── docker/
│   ├── Dockerfile.api
│   └── prometheus.yml
├── .github/
│   └── workflows/
│       └── ci.yml
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Evaluation Metrics

| Metric | Formula | Target |
|---|---|---|
| TSS | TPR − FPR | > 0.7 |
| FAR | FP / (FP + TN) | < 0.15 |
| ROC-AUC | Area under ROC | > 0.85 |
| Mean Lead Time | Avg(onset − alert) | > 20 min |
| C-class TPR | TP_C / (TP_C + FN_C) | > 0.80 |

> **Do not report accuracy.** With 95%+ quiet-sun samples, a model that predicts "no flare" always scores >95% accuracy while being completely useless.

---

## References

- Aditya-L1 Mission Overview — ISRO: https://www.isro.gov.in
- PRADAN Data Portal — ISSDC: https://pradan.issdc.gov.in
- NOAA SWPC Flare Catalog: https://ftp.swpc.noaa.gov/pub/indices/events/
- SunPy Documentation: https://sunpy.org
- LangGraph Documentation: https://langchain-ai.github.io/langgraph/
- FLAML AutoML — Microsoft: https://microsoft.github.io/FLAML/
- Phi-3-mini — Microsoft: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
- Mistral 7B: https://huggingface.co/mistralai/Mistral-7B-v0.1
- Ollama (local LLM runtime): https://ollama.com
- Baldi et al. (2021) — Solar Flare Prediction with CNNs
- Ahmadzadeh et al. (2021) — How to Train a Flare Forecasting Model

---

*Built by Quantum Crew — ISRO Hackathon PS15*
