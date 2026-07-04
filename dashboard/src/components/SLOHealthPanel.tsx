import { useEffect, useState } from "react";
import type { StatusResponse } from "../types/api";
import { Activity, CheckCircle, XCircle } from "lucide-react";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";
const BUDGET_SECONDS = 15; // per-component budget

export default function SLOHealthPanel() {
  const [status, setStatus] = useState<StatusResponse | null>(null);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch(`${API_BASE}/status`);
        if (res.ok) {
          setStatus(await res.json());
        }
      } catch {
        // Backend might not be running yet
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 10_000); // Poll every 10s
    return () => clearInterval(interval);
  }, []);

  const timing = status?.timing ?? {};
  const components = Object.entries(timing);

  function getBarClass(seconds: number): string {
    if (seconds <= BUDGET_SECONDS * 0.7) return "ok";
    if (seconds <= BUDGET_SECONDS) return "warn";
    return "over";
  }

  return (
    <div className="card slo-panel">
      <div className="card-title">
        <Activity size={14} />
        SLO Health Monitor
      </div>

      {status ? (
        <>
          <div
            className={`slo-status-badge ${
              status.slo_status === "PASS" ? "pass" : "fail"
            }`}
          >
            {status.slo_status === "PASS" ? (
              <CheckCircle size={14} />
            ) : (
              <XCircle size={14} />
            )}
            {status.slo_status} — {status.total_seconds.toFixed(1)}s
          </div>

          {status.offending_components.length > 0 && (
            <p
              style={{
                fontSize: "0.75rem",
                color: "var(--accent-red)",
                marginBottom: 12,
              }}
            >
              Offending: {status.offending_components.join(", ")}
            </p>
          )}

          {components.length === 0 ? (
            <p style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>
              Awaiting first pipeline run…
            </p>
          ) : (
            components.map(([name, seconds]) => {
              const pct = Math.min((seconds / BUDGET_SECONDS) * 100, 100);
              return (
                <div className="slo-component" key={name}>
                  <span className="slo-component-name">{name}</span>
                  <div className="slo-bar-bg">
                    <div
                      className={`slo-bar-fill ${getBarClass(seconds)}`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <span className="slo-time">{seconds.toFixed(1)}s</span>
                </div>
              );
            })
          )}
        </>
      ) : (
        <p style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>
          Connecting to API…
        </p>
      )}
    </div>
  );
}
