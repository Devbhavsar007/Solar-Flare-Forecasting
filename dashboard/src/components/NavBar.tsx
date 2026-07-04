import { NavLink } from "react-router-dom";
import { Activity } from "lucide-react";
import { useLiveData } from "../context/LiveDataContext";

const LINKS = [
  { to: "/", label: "Home", end: true },
  { to: "/dashboard", label: "Mission Control" },
  { to: "/observatory", label: "Observatory" },
  { to: "/earth-impact", label: "Earth Impact" },
  { to: "/analytics", label: "Analytics" },
  { to: "/explainability", label: "Explainability" },
  { to: "/architecture", label: "Architecture" },
  { to: "/about", label: "About" },
];

export default function NavBar() {
  const { isConnected, isLive } = useLiveData();

  return (
    <nav className="top-nav">
      <NavLink to="/" className="nav-brand">
        <span className="nav-brand-icon">
          <Activity size={18} />
        </span>
        <span className="nav-brand-text">
          <div className="name">
            JW<span className="accent">ALA</span>
          </div>
          <div className="sub">SOLAR FLARE FORECASTING</div>
        </span>
      </NavLink>

      <div className="nav-links">
        {LINKS.map((l) => (
          <NavLink
            key={l.to}
            to={l.to}
            end={l.end}
            className={({ isActive }) => `nav-link${isActive ? " active" : ""}`}
          >
            {l.label}
          </NavLink>
        ))}
      </div>

      <div className="nav-status">
        <span className={`nav-status-dot ${isConnected ? "connected" : "disconnected"}`} />
        {isConnected ? (isLive ? "LIVE" : "CONNECTED") : "OFFLINE"}
      </div>
    </nav>
  );
}