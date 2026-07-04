import {
  Download, Search, GitMerge, Brain, TrendingUp,
  BarChart3, FileText, Shield, Activity
} from "lucide-react";

const STAGES = [
  {
    name: "FITS Ingestion",
    desc: "SoLEXS + HEL1OS FITS files from PRADAN, deduplicated via seen_files.db",
    icon: Download,
    color: "var(--accent-cyan)",
    tag: "src/ingestion/",
  },
  {
    name: "Independent Detection",
    desc: "solexs_detector.py and hel1os_detector.py run on separate single-instrument dataframes [RULE-1]",
    icon: Search,
    color: "var(--accent-blue)",
    tag: "src/nowcasting/",
  },
  {
    name: "Catalogue Merge",
    desc: "Temporal coincidence matching merges independent detections into a unified catalogue",
    icon: GitMerge,
    color: "var(--accent-purple)",
    tag: "src/catalogue/merger.py",
  },
  {
    name: "Nowcasting (XGBoost)",
    desc: "4-class multi:softprob [RULE-5] with per-class thresholds optimized for X-class recall",
    icon: Brain,
    color: "var(--accent-orange)",
    tag: "src/nowcasting/train.py",
  },
  {
    name: "Forecasting Ensemble",
    desc: "Causal LSTM + TimesFM + PINN — all unidirectional [RULE-2], multi-horizon (15/30/60 min)",
    icon: TrendingUp,
    color: "var(--accent-amber)",
    tag: "src/forecasting/",
  },
  {
    name: "Uncertainty Quantification",
    desc: "Conformal prediction (MAPIE) + Chronos probabilistic intervals for calibrated coverage",
    icon: BarChart3,
    color: "var(--accent-emerald)",
    tag: "src/uncertainty/",
  },
  {
    name: "Explainability",
    desc: "SHAP TreeExplainer + DSPy/GraphRAG natural-language bulletin generation",
    icon: FileText,
    color: "var(--accent-cyan)",
    tag: "src/explainability/ + src/intelligence/",
  },
  {
    name: "Deployment & SLO",
    desc: "ONNX export, shadow scoring, drift detection (PSI), Prometheus + Grafana SLO dashboard",
    icon: Shield,
    color: "var(--accent-red)",
    tag: "src/deployment/",
  },
  {
    name: "API & WebSocket",
    desc: "FastAPI serving /health, /status, /alert, /history, /explain + real-time ws/live feed",
    icon: Activity,
    color: "var(--accent-orange)",
    tag: "src/api/main.py",
  },
];

export default function Architecture() {
  return (
    <div className="page">
      <div className="page-eyebrow">Architecture</div>
      <h1 className="page-title">
        Pipeline <span className="accent">Architecture</span>
      </h1>
      <p className="page-subtitle">
        End-to-end data flow from Aditya-L1 telemetry to actionable alert,
        orchestrated by a LangGraph agent DAG completing in under 60 seconds.
      </p>

      {STAGES.map((s, i) => (
        <div key={s.name}>
          <div className="pipeline-stage">
            <div className="pipeline-stage-icon" style={{ background: s.color }}>
              <s.icon size={20} />
            </div>
            <div className="pipeline-stage-body">
              <div className="name">{s.name}</div>
              <div className="desc">{s.desc}</div>
            </div>
            <div className="pipeline-stage-tag">{s.tag}</div>
          </div>
          {i < STAGES.length - 1 && <div className="pipeline-connector" />}
        </div>
      ))}
    </div>
  );
}
