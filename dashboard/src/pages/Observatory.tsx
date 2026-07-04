import { Shield, AlertTriangle } from "lucide-react";
import { useLiveData } from "../context/LiveDataContext";
import SunHero from "../components/SunHero";
import FluxChart from "../components/FluxChart";

const CLASS_COLORS: Record<string, string> = {
  A: "var(--flare-a)",
  B: "var(--flare-b)",
  C: "var(--flare-c)",
  M: "var(--flare-m)",
  X: "var(--flare-x)",
};

function alertLevel(fc: string): { level: string; label: string } {
  if (fc === "X") return { level: "HIGH", label: "X-CLASS ALERT" };
  if (fc === "M") return { level: "MODERATE", label: "M-CLASS WARNING" };
  return { level: "LOW", label: "NOMINAL" };
}

export default function Observatory() {
  const { alert, fluxData, forecast } = useLiveData();
  const fc = alert?.flare_class ?? "N";
  const { level, label } = alertLevel(fc);

  const probs = forecast
    ? [
        { cls: "C-class", p: forecast.prob_15min, color: CLASS_COLORS.C },
        { cls: "M-class", p: forecast.prob_30min, color: CLASS_COLORS.M },
        { cls: "X-class", p: forecast.prob_60min, color: CLASS_COLORS.X },
      ]
    : [];

  return (
    <div className="page">
      <div className="page-eyebrow">Observatory</div>
      <h1 className="page-title">
        Solar <span className="accent">Activity Monitor</span>
      </h1>
      <p className="page-subtitle">
        Real-time flare classification from Aditya-L1's SoLEXS and HEL1OS
        instruments, with multi-horizon probability forecasts.
      </p>

      <div className="honesty-note">
        <AlertTriangle size={16} />
        <span>
          The animated sun below is <strong>illustrative only</strong>. SoLEXS
          and HEL1OS are disk-integrated spectrometers — they measure total
          flux, not spatially resolved imagery. No active-region boxes are
          drawn because no real data supports them.
        </span>
      </div>

      <div className="observatory-grid">
        {/* Left: Sun + alert level */}
        <div>
          <SunHero size={340} />
          <div className={`alert-level-card ${level}`} style={{ marginTop: 24 }}>
            <div className="alert-level-label">
              <Shield size={22} /> {label}
            </div>
            <div className="alert-level-sub">
              Current classification: {fc}-class
              {alert ? ` · Confidence ${(alert.confidence * 100).toFixed(0)}%` : ""}
            </div>
          </div>
        </div>

        {/* Right: Flux chart + probability bars */}
        <div>
          <div className="card" style={{ marginBottom: 20 }}>
            <div className="card-title">X-Ray Flux (120 min window)</div>
            <div style={{ height: 260 }}>
              <FluxChart data={fluxData} />
            </div>
          </div>

          <div className="card">
            <div className="card-title">Flare Probability Forecast</div>
            {probs.map((p) => (
              <div className="prob-row" key={p.cls}>
                <span className="prob-class-label" style={{ color: p.color }}>
                  {p.cls}
                </span>
                <div className="prob-bar-bg">
                  <div
                    className="prob-bar-fill"
                    style={{ width: `${p.p * 100}%`, background: p.color }}
                  />
                </div>
                <span className="prob-value">{(p.p * 100).toFixed(1)}%</span>
              </div>
            ))}
            {probs.length === 0 && (
              <div className="empty-state">No forecast data available</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
