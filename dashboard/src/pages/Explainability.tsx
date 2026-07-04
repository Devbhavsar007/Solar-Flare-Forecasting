import { AlertTriangle } from "lucide-react";
import { useLiveData } from "../context/LiveDataContext";

const CLASS_LABELS = ["N / Quiet", "C-class", "M-class", "X-class"];

export default function Explainability() {
  const { alert, shap } = useLiveData();

  const bulletin = alert?.bulletin;
  const features = shap?.top_features ?? [];
  const maxAbs = Math.max(...features.map(([, v]) => Math.abs(v)), 0.01);

  return (
    <div className="page">
      <div className="page-eyebrow">Explainability</div>
      <h1 className="page-title">
        Why This <span className="accent">Classification?</span>
      </h1>
      <p className="page-subtitle">
        SHAP feature attributions and an LLM-generated natural-language
        bulletin explaining the current alert.
      </p>

      {/* SHAP Waterfall */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div className="card-title">SHAP Feature Attribution (Top 5)</div>
        {features.length > 0 ? (
          features.map(([name, value]) => {
            const pct = (Math.abs(value) / maxAbs) * 100;
            const positive = value >= 0;
            return (
              <div className="waterfall-row" key={name}>
                <div className="waterfall-row-head">
                  <span style={{ color: "var(--text-secondary)" }}>{name}</span>
                  <span style={{ color: positive ? "var(--accent-cyan)" : "var(--accent-red)" }}>
                    {positive ? "+" : ""}{value.toFixed(3)}
                  </span>
                </div>
                <div className="waterfall-bar-bg">
                  <div
                    className="waterfall-bar-fill"
                    style={{
                      width: `${pct}%`,
                      background: positive
                        ? "linear-gradient(90deg, var(--accent-cyan), var(--accent-blue))"
                        : "linear-gradient(90deg, var(--accent-orange), var(--accent-red))",
                    }}
                  />
                </div>
              </div>
            );
          })
        ) : (
          <div className="empty-state">No SHAP data available</div>
        )}
      </div>

      {/* Class importances */}
      {shap && (
        <div className="card" style={{ marginBottom: 24 }}>
          <div className="card-title">Per-Class Model Confidence</div>
          <div className="gauge-container">
            {shap.class_importances.map((v, i) => (
              <div className="gauge-item" key={i}>
                <span className="gauge-label">{CLASS_LABELS[i]}</span>
                <div className="gauge-bar-bg">
                  <div
                    className="gauge-bar-fill"
                    style={{
                      width: `${v * 100}%`,
                      background: i === shap.dominant_class ? "var(--accent-cyan)" : "var(--text-muted)",
                    }}
                  />
                </div>
                <span className="gauge-value">{(v * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* LLM Bulletin */}
      <div className="card">
        <div className="card-title">AI-Generated Bulletin</div>
        {bulletin ? (
          <div className="inference-summary">{bulletin}</div>
        ) : (
          <>
            <div className="honesty-note" style={{ marginBottom: 0 }}>
              <AlertTriangle size={16} />
              <span>
                The bulletin below is a <strong>template placeholder</strong>.
                It will be replaced automatically once the DSPy + GraphRAG
                bulletin generator runs in production and populates
                <code> alert.bulletin</code> on the WebSocket payload.
              </span>
            </div>
            <div className="inference-summary" style={{ marginTop: 16 }}>
              "A {alert?.flare_class ?? "—"}-class solar flare was detected by{" "}
              {alert?.instrument ?? "SoLEXS"} with peak flux{" "}
              {alert ? alert.peak_flux.toExponential(2) : "—"} W/m².
              Confidence: {alert ? (alert.confidence * 100).toFixed(0) : "—"}%.
              The primary driver is the 5-minute flux derivative, consistent
              with impulsive-phase hard X-ray emission."
            </div>
          </>
        )}
      </div>
    </div>
  );
}
