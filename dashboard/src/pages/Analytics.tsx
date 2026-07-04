import { useEffect, useState } from "react";
import { BarChart3 } from "lucide-react";
import type { EvaluationMetrics } from "../types/api";

const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export default function Analytics() {
  const [metrics, setMetrics] = useState<EvaluationMetrics | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_BASE}/evaluation`)
      .then((r) => r.json())
      .then((d: EvaluationMetrics) => setMetrics(d))
      .catch(() => setMetrics(null))
      .finally(() => setLoading(false));
  }, []);

  const available = metrics?.available === true;

  const stats = available
    ? [
        { label: "TSS", value: metrics!.tss!.toFixed(3), color: "var(--accent-cyan)" },
        { label: "FAR", value: metrics!.far!.toFixed(3), color: "var(--accent-orange)" },
        { label: "TPR", value: metrics!.tpr!.toFixed(3), color: "var(--accent-emerald)" },
        { label: "Lead Time", value: `${metrics!.mean_lead_min!.toFixed(1)} min`, color: "var(--accent-purple)" },
      ]
    : [
        { label: "TSS", value: "—", color: "var(--text-muted)" },
        { label: "FAR", value: "—", color: "var(--text-muted)" },
        { label: "TPR", value: "—", color: "var(--text-muted)" },
        { label: "Lead Time", value: "—", color: "var(--text-muted)" },
      ];

  return (
    <div className="page">
      <div className="page-eyebrow">Analytics</div>
      <h1 className="page-title">
        Model <span className="accent">Performance</span>
      </h1>
      <p className="page-subtitle">
        Walk-forward cross-validation metrics from the latest evaluation run.
        {!available && !loading && " No evaluation data found — run the pipeline first."}
      </p>

      {/* Stat cards */}
      <div className="stat-cards-row">
        {stats.map((s) => (
          <div className="stat-card" key={s.label}>
            <div className="stat-card-label">{s.label}</div>
            <div className="stat-card-value" style={{ color: s.color }}>{s.value}</div>
          </div>
        ))}
      </div>

      <div className="analytics-grid-2">
        {/* Additional detail cards */}
        <div className="card">
          <div className="card-title">
            <BarChart3 size={14} /> Evaluation Details
          </div>
          {available ? (
            <div style={{ fontSize: "0.85rem", color: "var(--text-secondary)", lineHeight: 1.8 }}>
              <div>
                <span className="mono" style={{ color: "var(--text-muted)", width: 80, display: "inline-block" }}>PPV</span>
                <span className="mono">{metrics!.ppv!.toFixed(4)}</span>
              </div>
              <div>
                <span className="mono" style={{ color: "var(--text-muted)", width: 80, display: "inline-block" }}>TNR</span>
                <span className="mono">{metrics!.tnr!.toFixed(4)}</span>
              </div>
              <div>
                <span className="mono" style={{ color: "var(--text-muted)", width: 80, display: "inline-block" }}>FAR</span>
                <span className="mono">{metrics!.far!.toFixed(4)}</span>
                {metrics!.far! <= 0.1 ? (
                  <span style={{ color: "var(--slo-pass)", marginLeft: 8, fontSize: "0.75rem" }}>✓ SLO-5 PASS</span>
                ) : (
                  <span style={{ color: "var(--slo-fail)", marginLeft: 8, fontSize: "0.75rem" }}>✗ SLO-5 FAIL</span>
                )}
              </div>
            </div>
          ) : (
            <div className="empty-state">
              {loading ? "Loading…" : "Run walk_forward.py to generate eval_metrics.json"}
            </div>
          )}
        </div>

        <div className="card">
          <div className="card-title">Correlation Matrix</div>
          <div className="empty-state">
            Feature correlation analysis requires the <code>/features</code> endpoint,
            which is not yet wired. This section will populate automatically once
            engineered feature values are exposed by the backend.
          </div>
        </div>
      </div>
    </div>
  );
}
