import { AlertTriangle, Satellite, Wifi, Navigation, Radio, Zap } from "lucide-react";
import { useLiveData } from "../context/LiveDataContext";

/** Heuristic risk mapping — see honesty note below. */
function riskFromClass(fc: string): number {
  if (fc === "X") return 0.92;
  if (fc === "M") return 0.55;
  if (fc === "C") return 0.18;
  return 0.03;
}

const SECTORS = [
  { name: "Satellite Operations", icon: Satellite, weight: 1.0 },
  { name: "HF Radio Communications", icon: Radio, weight: 0.9 },
  { name: "GNSS / Navigation", icon: Navigation, weight: 0.75 },
  { name: "Power Grid (GIC)", icon: Zap, weight: 0.6 },
  { name: "Aviation (Polar Routes)", icon: Wifi, weight: 0.5 },
];

function riskColor(v: number): string {
  if (v > 0.7) return "var(--accent-red)";
  if (v > 0.4) return "var(--accent-orange)";
  return "var(--accent-emerald)";
}

export default function EarthImpact() {
  const { alert } = useLiveData();
  const fc = alert?.flare_class ?? "N";
  const baseRisk = riskFromClass(fc);

  return (
    <div className="page">
      <div className="page-eyebrow">Earth Impact</div>
      <h1 className="page-title">
        Space Weather <span className="accent">Impact Assessment</span>
      </h1>
      <p className="page-subtitle">
        Estimated downstream effects of the current solar flare classification
        on critical Earth-side infrastructure.
      </p>

      <div className="honesty-note">
        <AlertTriangle size={16} />
        <span>
          These risk scores are <strong>heuristic estimates</strong> derived
          from the current flare class, not from a calibrated geomagnetic
          impact model. JWALA does not ingest CME propagation data, Dst/Kp
          indices, or magnetometer readings — so the numbers here are
          illustrative only and should not be used for operational decisions.
        </span>
      </div>

      {/* L1 Diagram */}
      <div className="l1-diagram" style={{ marginBottom: 28 }}>
        <div className="body-sun" />
        <div className="axis-line">
          <div className="satellite-marker">
            <Satellite size={18} />
            <span className="tag">Aditya-L1 @ Sun–Earth L1</span>
          </div>
        </div>
        <div className="body-earth" />
      </div>

      {/* Sectoral Risk */}
      <div className="card" style={{ marginBottom: 28 }}>
        <div className="card-title">Sectoral Risk Breakdown</div>
        {SECTORS.map((s) => {
          const risk = Math.min(baseRisk * s.weight, 1);
          return (
            <div className="sectoral-risk-row" key={s.name}>
              <div className="sectoral-risk-icon" style={{ color: riskColor(risk) }}>
                <s.icon size={18} />
              </div>
              <span className="sectoral-risk-name">{s.name}</span>
              <div className="sectoral-risk-bar-bg">
                <div
                  className="sectoral-risk-bar-fill"
                  style={{ width: `${risk * 100}%`, background: riskColor(risk) }}
                />
              </div>
              <span className="sectoral-risk-pct" style={{ color: riskColor(risk) }}>
                {(risk * 100).toFixed(0)}%
              </span>
            </div>
          );
        })}
      </div>

      <div className="card">
        <div className="card-title">What This Means</div>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.88rem", lineHeight: 1.7 }}>
          {fc === "X" && (
            <>
              <strong style={{ color: "var(--accent-red)" }}>CRITICAL:</strong> X-class
              flares can disrupt HF radio on the sunlit side within minutes. Satellite
              drag increases, GNSS accuracy degrades, and induced currents may stress
              high-latitude power grids. Airlines may reroute polar flights.
            </>
          )}
          {fc === "M" && (
            <>
              <strong style={{ color: "var(--accent-orange)" }}>ELEVATED:</strong> M-class
              flares cause moderate HF radio fadeouts and minor GNSS perturbations.
              Satellite operators should monitor charging levels.
            </>
          )}
          {(fc === "C" || fc === "N" || fc === "B" || fc === "A") && (
            <>
              <strong style={{ color: "var(--accent-emerald)" }}>NOMINAL:</strong> Current
              solar activity poses no significant risk to Earth-side systems. Standard
              monitoring continues.
            </>
          )}
        </p>
      </div>
    </div>
  );
}
