import { NavLink } from "react-router-dom";
import { useLiveData } from "../context/LiveDataContext";
import JwalaLogo from "./JwalaLogo";

const LINKS = [
  { to: "/", label: "Home", end: true },
  { to: "/jwala", label: "JWALA Dashboard" },
];

export default function NavBar() {
  const { isConnected, isLive } = useLiveData();

  return (
    <nav className="top-nav">
      <NavLink to="/" className="nav-brand">
        <span className="nav-brand-icon">
          <JwalaLogo size={22} />
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