# ☀️ JWALA — Solar Flare Nowcasting & Forecasting

JWALA is an advanced full-stack application designed for automated solar flare detection, classification, and prediction. It leverages data from instruments like ISRO's Aditya-L1 SoLEXS and HEL1OS, merging them with advanced machine learning, physics-informed models, and LLM-driven intelligence.

[![CI](https://github.com/quantum-crew/jwala/actions/workflows/ci.yml/badge.svg)](https://github.com/quantum-crew/jwala/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Current Features & Functionality

JWALA is built as a highly modular, scalable, and production-ready system with the following active capabilities:

### 1. Data Ingestion & Preprocessing (`src/ingestion/`, `src/preprocessing/`)
- **Multi-Source Data Pipelines**: Automated downloading and FITS file reading for SoLEXS, HEL1OS, and GOES XRS data.
- **Robust Preprocessing**: Cross-calibration, physics-based feature extraction, and time-series data augmentation.

### 2. Independent Detectors & Nowcasting (`src/nowcasting/`, `src/physics/`)
- **Instrument-Specific Detection**: Dedicated detection pipelines for SoLEXS (Soft X-ray) and HEL1OS (Hard X-ray).
- **Physics-Informed Neural Networks (PINN)**: Embeds solar physics constraints (e.g., Neupert effect) directly into the loss function.
- **Phase Detection**: Identifies solar flare activity phases dynamically.
- **TCN Encoders**: Utilizes Temporal Convolutional Networks for accurate feature representation.

### 3. Forecasting & Uncertainty (`src/forecasting/`, `src/uncertainty/`)
- **Three-Model Forecasting Ensemble**: Combines Causal LSTMs, TCNs, and fine-tuned TimesFM foundation models.
- **Pre-flare Anomaly Detection**: Uses MOMENT for continuous pre-flare anomaly scoring.
- **Dual Uncertainty Estimation**: Provides highly calibrated probability bands via MAPIE conformal intervals and Chronos-Bolt.

### 4. Agentic Intelligence & Explainability (`src/intelligence/`, `src/explainability/`)
- **LLM Intelligence**: DSPy-powered self-optimizing reporting and GraphRAG knowledge graph retrieval for physics-based reasoning.
- **SHAP Explainer**: Per-prediction feature attribution to explain exactly *why* a forecast or alert was triggered.

### 5. Orchestration & Monitoring (`src/orchestration/`, `src/monitoring/`)
- **LangGraph Orchestration**: Robust state-graph workflows managing agent operations and fallback states.
- **Extensive Metrics**: Real-time evaluation of prediction reliability, Service Level Objectives (SLOs), and Walk-Forward CV pipelines.

### 6. Interactive Frontend Dashboard (`dashboard/`)
A responsive and fast Vite + React dashboard featuring:
- **Live Data Visualizations**: Real-time Flux charts and observability dashboards.
- **Detailed Forecasting Views**: Forecast gauges, Lead time badges, and Earth-impact analytics.
- **Explainability View**: Frontend integration with SHAP reports to provide transparent AI decision-making to operators.
- **System Health**: Dedicated SLO health panels.

### 7. Deployment & Infrastructure
- **Containerized Workflows**: Complete `docker-compose` setups for `dev`, `staging`, and `prod` environments.
- **Grafana & Prometheus Observability**: Ready-to-deploy metrics aggregation and visualization (`grafana/`).
- **Nginx Configured Routing**: Highly optimized gateway configurations.
- **MCP Integration**: Experimental Model Context Protocol (MCP) launcher scripts (`stitch_mcp_launcher.ps1`).

---

## Getting Started

### Prerequisites
- Python 3.11+
- Node.js (for the dashboard)
- Docker & Docker Compose (for complete stack execution)
- [Ollama](https://ollama.com/) (For local LLM inference)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/quantum-crew/jwala.git
   cd jwala
   ```

2. **Backend Setup:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Frontend Setup:**
   ```bash
   cd dashboard
   npm install
   npm run dev
   ```

4. **Docker Deployment:**
   To spin up the entire application along with observability tools (Grafana/Prometheus):
   ```bash
   docker-compose -f docker-compose.dev.yml up --build
   ```

## Documentation
Additional architectural details, decision records, and API structures can be found within the `docs/` directory. For system evaluations and walk-forward validations, see `src/evaluation/`.
