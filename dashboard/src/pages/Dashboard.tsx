import { useLiveData } from "../context/LiveDataContext";
import AlertBanner from "../components/AlertBanner";
import FluxChart from "../components/FluxChart";
import ForecastGauge from "../components/ForecastGauge";
import SHAPExplainer from "../components/SHAPExplainer";
import LeadTimeBadge from "../components/LeadTimeBadge";
import SLOHealthPanel from "../components/SLOHealthPanel";

export default function Dashboard() {
  const { isConnected, isLive, alert, fluxData, forecast, shap, leadTime } = useLiveData();

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <h1>Mission Control</h1>
        <div className="header-status">
          <LeadTimeBadge minutes={leadTime} />
          <span className={`connection-dot ${isConnected ? "connected" : "disconnected"}`} />
          <span className="mono" style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
            {isConnected ? (isLive ? "LIVE" : "DEMO") : "OFFLINE"}
          </span>
        </div>
      </header>

      <div className="dashboard-grid">
        <AlertBanner alert={alert} />

        <div className="card chart-card">
          <div className="card-title">Real-Time X-Ray Flux</div>
          <FluxChart data={fluxData} />
        </div>

        <div className="card">
          <div className="card-title">Forecast Probability</div>
          <ForecastGauge forecast={forecast} />
        </div>

        <div className="card">
          <div className="card-title">SHAP Feature Importance</div>
          <SHAPExplainer shap={shap} />
        </div>

        <div className="card slo-panel">
          <div className="card-title">SLO Health</div>
          <SLOHealthPanel />
        </div>
      </div>
    </div>
  );
}
