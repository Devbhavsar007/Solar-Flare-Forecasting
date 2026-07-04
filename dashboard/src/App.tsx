import { useState, useEffect } from "react";
import { useWebSocket } from "./hooks/useWebSocket";
import AlertBanner from "./components/AlertBanner";
import FluxChart from "./components/FluxChart";
import ForecastGauge from "./components/ForecastGauge";
import SHAPExplainer from "./components/SHAPExplainer";
import LeadTimeBadge from "./components/LeadTimeBadge";
import SLOHealthPanel from "./components/SLOHealthPanel";
import type {
  FlareAlert,
  FluxDataPoint,
  ForecastData,
  SHAPData,
} from "./types/api";
import "./index.css";

const WS_URL = import.meta.env.VITE_WS_URL ?? "ws://localhost:8000/ws/live";

/**
 * Generate synthetic demo data for visualisation when no live backend is running.
 */
function generateDemoFluxData(): FluxDataPoint[] {
  const now = Date.now();
  const points: FluxDataPoint[] = [];
  for (let i = 119; i >= 0; i--) {
    const t = new Date(now - i * 60_000);
    const base = 1e-7;
    // Simulate a flare around minute 60
    const flareContrib =
      i < 80 && i > 40
        ? (5e-5 - base) * Math.exp(-((i - 55) ** 2) / 100)
        : 0;
    const sxr = base + flareContrib + Math.random() * 1e-8;
    const hxr =
      50 +
      (i < 80 && i > 40
        ? 2500 * Math.exp(-((i - 53) ** 2) / 80)
        : 0) +
      Math.random() * 10;

    points.push({
      timestamp: t.toLocaleTimeString("en-GB", {
        hour: "2-digit",
        minute: "2-digit",
      }),
      sxr,
      hxr,
      q10: sxr * 0.7,
      q90: sxr * 1.4,
    });
  }
  return points;
}

const demoAlert: FlareAlert = {
  flare_class: "M",
  instrument: "SoLEXS",
  peak_flux: 5e-5,
  peak_time: new Date().toISOString(),
  confidence: 0.87,
  start_time: new Date(Date.now() - 600_000).toISOString(),
  end_time: new Date(Date.now() + 600_000).toISOString(),
};

const demoForecast: ForecastData = {
  prob_15min: 0.72,
  prob_30min: 0.58,
  prob_60min: 0.34,
  predicted_class: 2,
};

const demoShap: SHAPData = {
  top_features: [
    ["flux_derivative_5m", 0.42],
    ["hxr_sxr_ratio", 0.31],
    ["background_sigma", -0.18],
    ["peak_frac_current", 0.15],
    ["counts_high_slope", -0.09],
  ],
  class_importances: [0.1, 0.15, 0.55, 0.2],
  dominant_class: 2,
};

export default function App() {
  const { lastMessage, isConnected } = useWebSocket(WS_URL);
  const [alert, setAlert] = useState<FlareAlert | null>(demoAlert);
  const [fluxData, setFluxData] = useState<FluxDataPoint[]>(
    generateDemoFluxData()
  );
  const [forecast, setForecast] = useState<ForecastData | null>(demoForecast);
  const [shap, setShap] = useState<SHAPData | null>(demoShap);
  const [leadTime] = useState<number | null>(12);

  // Process WebSocket messages
  useEffect(() => {
    if (lastMessage) {
      try {
        const data = JSON.parse(lastMessage);
        if (data.flare_class) setAlert(data);
        if (data.forecast) setForecast(data.forecast);
        if (data.shap) setShap(data.shap);
      } catch {
        // Not valid JSON — ignore
      }
    }
  }, [lastMessage]);

  // Simulate live flux updates (demo mode)
  useEffect(() => {
    const interval = setInterval(() => {
      setFluxData((prev) => {
        const newPoint: FluxDataPoint = {
          timestamp: new Date().toLocaleTimeString("en-GB", {
            hour: "2-digit",
            minute: "2-digit",
          }),
          sxr: 1e-7 + Math.random() * 1e-8,
          hxr: 50 + Math.random() * 10,
          q10: 1e-7 * 0.7,
          q90: 1e-7 * 1.4,
        };
        return [...prev.slice(1), newPoint];
      });
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="dashboard-layout">
      {/* Top Header */}
      <header className="header">
        <div className="flex items-center gap-md">
          <h1 className="title">
            ☀️ JWALA Solar Flare Dashboard
          </h1>
        </div>
        <div className="flex gap-lg" style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <LeadTimeBadge minutes={leadTime} />
          <div className="glass-panel" style={{ padding: '8px 16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ 
              width: '8px', 
              height: '8px', 
              borderRadius: '50%', 
              backgroundColor: isConnected ? 'var(--primary)' : 'var(--error)',
              boxShadow: isConnected ? '0 0 10px var(--primary)' : '0 0 10px var(--error)'
            }}></span>
            <span style={{ fontSize: '12px', fontFamily: 'var(--font-mono)' }}>{isConnected ? 'LIVE' : 'OFFLINE'}</span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="content-grid">
        {/* Left Column: Alerts & Flux Chart */}
        <div className="left-column">
          <AlertBanner alert={alert} />
          <div className="glass-panel" style={{ flex: 1, minHeight: '500px' }}>
            <h2 style={{ marginBottom: '16px', color: 'var(--text-primary)' }}>Real-Time Flux Monitor</h2>
            <FluxChart data={fluxData} />
          </div>
        </div>

        {/* Right Column: Gauges, SHAP, and SLO */}
        <div className="right-column">
          <div className="glass-panel">
            <h2 style={{ marginBottom: '16px', color: 'var(--text-primary)' }}>Forecast Probability</h2>
            <ForecastGauge forecast={forecast} />
          </div>

          <div className="glass-panel">
            <h2 style={{ marginBottom: '16px', color: 'var(--text-primary)' }}>SHAP Feature Importance</h2>
            <SHAPExplainer shap={shap} />
          </div>

          <div className="glass-panel" style={{ marginTop: 'auto' }}>
            <h2 style={{ marginBottom: '16px', color: 'var(--text-primary)' }}>SLO Health Monitor</h2>
            <SLOHealthPanel />
          </div>
        </div>
      </main>
    </div>
  );
}
