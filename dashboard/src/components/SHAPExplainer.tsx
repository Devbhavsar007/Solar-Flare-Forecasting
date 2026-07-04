import type { SHAPData } from "../types/api";
import { Lightbulb } from "lucide-react";

interface Props {
  shap: SHAPData | null;
}

export default function SHAPExplainer({ shap }: Props) {
  if (!shap || !shap.top_features || shap.top_features.length === 0) {
    return (
      <div className="card">
        <div className="card-title">
          <Lightbulb size={14} />
          SHAP Feature Importance
        </div>
        <p style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>
          No explanation data available yet.
        </p>
      </div>
    );
  }

  const maxAbs = Math.max(
    ...shap.top_features.map(([, v]) => Math.abs(v)),
    1e-9
  );

  return (
    <div className="card">
      <div className="card-title">
        <Lightbulb size={14} />
        SHAP Feature Importance
      </div>
      <div className="shap-bar-container">
        {shap.top_features.map(([name, value]) => {
          const pct = (Math.abs(value) / maxAbs) * 100;
          const isPositive = value >= 0;
          return (
            <div className="shap-row" key={name}>
              <span className="shap-feature">{name}</span>
              <div className="shap-bar-wrapper">
                <div
                  className={`shap-bar ${isPositive ? "positive" : "negative"}`}
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
