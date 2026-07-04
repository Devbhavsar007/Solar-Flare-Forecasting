import type { FlareAlert } from "../types/api";
import { AlertTriangle, Shield, Zap, Sun } from "lucide-react";

interface Props {
  alert: FlareAlert | null;
}

const classIcons: Record<string, React.ReactNode> = {
  N: <Shield size={24} />,
  C: <Sun size={24} />,
  M: <Zap size={24} />,
  X: <AlertTriangle size={24} />,
};

const classLabels: Record<string, string> = {
  N: "No Active Flare",
  C: "C-Class Flare Detected",
  M: "M-Class Flare Detected — Enhanced Monitoring",
  X: "X-Class Flare Detected — Critical Alert",
};

export default function AlertBanner({ alert }: Props) {
  const flareClass = alert?.flare_class ?? "N";
  const safeClass = ["N", "C", "M", "X"].includes(flareClass) ? flareClass : "N";

  return (
    <div className={`alert-banner ${safeClass}`}>
      <div className="alert-class">
        {classIcons[safeClass]}
        <span style={{ marginLeft: 8 }}>{safeClass}</span>
      </div>
      <div className="alert-info">
        <strong>{classLabels[safeClass]}</strong>
        {alert && (
          <p>
            {alert.instrument} • Peak: {alert.peak_time} • Flux:{" "}
            {alert.peak_flux?.toExponential(2)}
          </p>
        )}
      </div>
      {alert && (
        <div className="alert-confidence">
          {(alert.confidence * 100).toFixed(0)}% confidence
        </div>
      )}
    </div>
  );
}
