import { ExternalLink } from "lucide-react";

const TECH = [
  {
    group: "ML / Deep Learning",
    items: ["XGBoost", "PyTorch", "ONNX Runtime", "FLAML AutoML", "MAPIE"],
  },
  {
    group: "Foundation Models",
    items: ["Google TimesFM", "MOMENT", "Amazon Chronos", "DSPy", "GraphRAG"],
  },
  {
    group: "Backend & Ops",
    items: ["FastAPI", "Prometheus", "Grafana", "Docker", "LangGraph"],
  },
  {
    group: "Frontend",
    items: ["React 19", "TypeScript", "Recharts", "Vite", "Lucide Icons"],
  },
  {
    group: "Data & Science",
    items: ["Astropy FITS", "Pandera", "SHAP", "NumPy", "Pandas"],
  },
];

const TIMELINE = [
  { date: "M0–M2", text: "Project bootstrap, FITS ingestion, Pandera schemas, independent SoLEXS/HEL1OS detectors" },
  { date: "M3–M5", text: "Catalogue merger, XGBoost 4-class nowcasting, GOES cross-calibration" },
  { date: "M6–M8", text: "Causal LSTM, TimesFM forecasting, SHAP explainability, conformal prediction" },
  { date: "M9–M11", text: "LangGraph orchestration, DSPy + GraphRAG bulletins, MOMENT anomaly detection" },
  { date: "M12–M14", text: "FLAML AutoML, ONNX export, shadow scoring, drift detection, Grafana SLO dashboard" },
  { date: "M15–M16", text: "Production hardening, CI/CD, load testing, integration test gate, L6 release" },
];

export default function About() {
  return (
    <div className="page">
      <div className="page-eyebrow">About</div>
      <h1 className="page-title">
        About <span className="accent">JWALA</span>
      </h1>
      <p className="page-subtitle">
        Joint Waveband Alert &amp; Light-curve Analyzer — built by Quantum Crew
        for ISRO's Bharatiya Antariksh Hackathon 2026, Problem Statement 15.
      </p>

      {/* Tech stack */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div className="card-title">Technology Stack</div>
        {TECH.map((t) => (
          <div className="tech-pill-group" key={t.group}>
            <div className="tech-pill-group-label">{t.group}</div>
            <div className="tech-pills">
              {t.items.map((item) => (
                <span className="tech-pill" key={item}>{item}</span>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="analytics-grid-2">
        {/* Timeline */}
        <div className="card">
          <div className="card-title">Development Timeline</div>
          {TIMELINE.map((t) => (
            <div className="timeline-item" key={t.date}>
              <div className="timeline-dot" />
              <div>
                <div className="label">{t.date}</div>
                <div style={{ fontSize: "0.85rem", color: "var(--text-secondary)", lineHeight: 1.6 }}>
                  {t.text}
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Team + links */}
        <div>
          <div className="card" style={{ marginBottom: 20 }}>
            <div className="card-title">Team — Quantum Crew</div>
            <p style={{ color: "var(--text-secondary)", fontSize: "0.88rem", lineHeight: 1.7 }}>
              A multidisciplinary team of developers passionate about space
              weather science, real-time ML pipelines, and making
              astrophysics accessible through modern web interfaces.
            </p>
          </div>

          <div className="card">
            <div className="card-title">Links</div>
            <a
              href="https://github.com/Devbhavsar007/Solar-Flare-Forecasting"
              target="_blank"
              rel="noopener noreferrer"
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 8,
                color: "var(--text-primary)",
                textDecoration: "none",
                padding: "10px 16px",
                borderRadius: "var(--radius-md)",
                background: "rgba(255,255,255,0.05)",
                border: "1px solid var(--border-color)",
                fontSize: "0.85rem",
                fontWeight: 600,
              }}
            >
              <ExternalLink size={16} /> GitHub Repository
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}
