import { Link } from "react-router-dom";
import { ArrowRight, Zap, Radio, Satellite } from "lucide-react";
import SunHero from "../components/SunHero";

export default function Home() {
  return (
    <>
      {/* ── Hero ─────────────────────────────────────────── */}
      <section className="hero-section">
        <div>
          <div className="hero-badge">
            <Zap size={14} /> ISRO Aditya-L1 Mission · PS-15
          </div>
          <h1 className="hero-title">
            Solar Flare<br />
            <span className="accent">Early Warning</span><br />
            System
          </h1>
          <p className="hero-lede">
            JWALA (Joint Waveband Alert &amp; Light-curve Analyzer) fuses
            real-time SoLEXS soft X-ray and HEL1OS hard X-ray telemetry from
            ISRO's Aditya-L1 spacecraft to nowcast and forecast solar flares
            up to 60 minutes before peak flux.
          </p>
          <Link to="/dashboard" className="hero-cta">
            Open Mission Control <ArrowRight size={16} />
          </Link>
        </div>
        <SunHero />
      </section>

      {/* ── What JWALA Does ──────────────────────────────── */}
      <section className="story-section">
        <h2 className="story-heading">How It Works</h2>
        <p className="story-sub">
          From photon detection at L1 to actionable alert on the ground —
          the entire pipeline runs in under 60 seconds.
        </p>
        <div className="story-grid-3">
          <div className="story-card">
            <div style={{ color: "var(--accent-cyan)", marginBottom: 12 }}>
              <Satellite size={28} />
            </div>
            <h3>Dual-Instrument Ingestion</h3>
            <p>
              SoLEXS and HEL1OS FITS files are ingested independently —
              never merged before detection — preserving the physics of each
              waveband.
            </p>
          </div>
          <div className="story-card">
            <div style={{ color: "var(--accent-orange)", marginBottom: 12 }}>
              <Zap size={28} />
            </div>
            <h3>Multi-Model Ensemble</h3>
            <p>
              XGBoost nowcasting, Causal LSTM, TimesFM foundation model,
              and a Physics-Informed Neural Network vote together for
              robust class predictions (N/C/M/X).
            </p>
          </div>
          <div className="story-card">
            <div style={{ color: "var(--accent-red)", marginBottom: 12 }}>
              <Radio size={28} />
            </div>
            <h3>Explainable Alerts</h3>
            <p>
              Every alert ships with SHAP feature attributions, conformal
              prediction intervals, and an LLM-generated natural-language
              bulletin via DSPy + GraphRAG.
            </p>
          </div>
        </div>
      </section>

      {/* ── Historic Flares ──────────────────────────────── */}
      <section className="story-section">
        <h2 className="story-heading">Built for Real Events</h2>
        <p className="story-sub">
          JWALA's pipeline is validated against the most powerful solar
          events of Solar Cycle 25.
        </p>

        <div className="event-row">
          <div>
            <div className="event-date">2024-02-22</div>
            <h3>X6.3 — Strongest Flare in 7 Years</h3>
            <p>
              The February 2024 X6.3 flare from AR 3590 produced a
              simultaneous hard X-ray burst detectable by HEL1OS. JWALA's
              integration test replays this event end-to-end as the
              production readiness gate.
            </p>
          </div>
          <div className="event-visual">
            <Zap size={64} />
          </div>
        </div>

        <div className="event-row reverse">
          <div>
            <div className="event-date">2024-05-10 — 2024-05-14</div>
            <h3>Geomagnetic Super-Storm</h3>
            <p>
              A sequence of X-class flares and CMEs triggered the most
              intense geomagnetic storm since 2003, with aurora visible
              at mid-latitudes. Events like this motivate JWALA's
              60-minute forecast horizon.
            </p>
          </div>
          <div className="event-visual">
            <Radio size={64} />
          </div>
        </div>
      </section>
    </>
  );
}
