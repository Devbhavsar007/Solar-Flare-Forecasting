import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { useWebSocket } from "../hooks/useWebSocket";
import type { FlareAlert, FluxDataPoint, ForecastData, SHAPData } from "../types/api";

const WS_URL = import.meta.env.VITE_WS_URL ?? "ws://localhost:8000/ws/live";

/**
 * Generate synthetic demo data for visualisation when no live backend is running.
 * Kept identical to the original App.tsx generator so existing behaviour doesn't change.
 */
function generateDemoFluxData(): FluxDataPoint[] {
  const now = Date.now();
  const points: FluxDataPoint[] = [];
  for (let i = 119; i >= 0; i--) {
    const t = new Date(now - i * 60_000);
    const base = 1e-7;
    const flareContrib =
      i < 80 && i > 40 ? (5e-5 - base) * Math.exp(-((i - 55) ** 2) / 100) : 0;
    const sxr = base + flareContrib + Math.random() * 1e-8;
    const hxr =
      50 +
      (i < 80 && i > 40 ? 2500 * Math.exp(-((i - 53) ** 2) / 80) : 0) +
      Math.random() * 10;

    points.push({
      timestamp: t.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" }),
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

interface LiveDataState {
  isConnected: boolean;
  isLive: boolean; // true once a real WS message has been received
  alert: FlareAlert | null;
  fluxData: FluxDataPoint[];
  forecast: ForecastData | null;
  shap: SHAPData | null;
  leadTime: number | null;
}

const LiveDataContext = createContext<LiveDataState | null>(null);

export function LiveDataProvider({ children }: { children: ReactNode }) {
  const { lastMessage, isConnected } = useWebSocket(WS_URL);
  const [isLive, setIsLive] = useState(false);
  const [alert, setAlert] = useState<FlareAlert | null>(demoAlert);
  const [fluxData, setFluxData] = useState<FluxDataPoint[]>(generateDemoFluxData());
  const [forecast, setForecast] = useState<ForecastData | null>(demoForecast);
  const [shap, setShap] = useState<SHAPData | null>(demoShap);
  const [leadTime] = useState<number | null>(12);

  useEffect(() => {
    if (lastMessage) {
      try {
        const data = JSON.parse(lastMessage);
        if (data.flare_class) {
          setAlert(data);
          setIsLive(true);
        }
        if (data.forecast) setForecast(data.forecast);
        if (data.shap) setShap(data.shap);
      } catch {
        // Not valid JSON — ignore
      }
    }
  }, [lastMessage]);

  // Demo-mode flux ticking — only runs while no real telemetry has arrived,
  // so it never overwrites genuine pipeline output.
  useEffect(() => {
    if (isLive) return;
    const interval = setInterval(() => {
      setFluxData((prev) => {
        const newPoint: FluxDataPoint = {
          timestamp: new Date().toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" }),
          sxr: 1e-7 + Math.random() * 1e-8,
          hxr: 50 + Math.random() * 10,
          q10: 1e-7 * 0.7,
          q90: 1e-7 * 1.4,
        };
        return [...prev.slice(1), newPoint];
      });
    }, 5000);
    return () => clearInterval(interval);
  }, [isLive]);

  return (
    <LiveDataContext.Provider
      value={{ isConnected, isLive, alert, fluxData, forecast, shap, leadTime }}
    >
      {children}
    </LiveDataContext.Provider>
  );
}

export function useLiveData(): LiveDataState {
  const ctx = useContext(LiveDataContext);
  if (!ctx) throw new Error("useLiveData must be used within LiveDataProvider");
  return ctx;
}