import type { ForecastData } from "../types/api";
import { Clock } from "lucide-react";

interface Props {
  forecast: ForecastData | null;
}

function getBarColor(prob: number): string {
  if (prob >= 0.7) return "linear-gradient(90deg, #ef4444, #f87171)";
  if (prob >= 0.4) return "linear-gradient(90deg, #f59e0b, #fbbf24)";
  return "linear-gradient(90deg, #10b981, #34d399)";
}

export default function ForecastGauge({ forecast }: Props) {
  const items = [
    { label: "15 min", value: forecast?.prob_15min ?? 0 },
    { label: "30 min", value: forecast?.prob_30min ?? 0 },
    { label: "60 min", value: forecast?.prob_60min ?? 0 },
  ];

  return (
    <div className="card">
      <div className="card-title">
        <Clock size={14} />
        Flare Probability Forecast
      </div>
      <div className="gauge-container">
        {items.map((item) => (
          <div className="gauge-item" key={item.label}>
            <span className="gauge-label">{item.label}</span>
            <div className="gauge-bar-bg">
              <div
                className="gauge-bar-fill"
                style={{
                  width: `${Math.round(item.value * 100)}%`,
                  background: getBarColor(item.value),
                }}
              />
            </div>
            <span className="gauge-value">
              {(item.value * 100).toFixed(0)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
