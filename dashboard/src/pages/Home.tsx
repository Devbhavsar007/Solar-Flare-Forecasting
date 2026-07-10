import { Link } from "react-router-dom";
import {
  ArrowRight,
  Satellite,
  Shield,
  Activity,
  Sparkles,
  CheckCircle,
  Cpu,
  Radio,
  Zap,
  Brain,
  Globe,
  AlertTriangle,
  Eye,
} from "lucide-react";
import JwalaLogo from "../components/JwalaLogo";
import { SparklesCore } from "../components/SparklesCore";
import WebGLBurnTransition from "../components/WebGLBurnTransition";

export default function Home() {

  return (
    <>

      {/* ── Sticky Hero (locks in the background) ── */}
      <section className="home-hero">
        {/* Full-screen background video */}
        <video
          className="hero-bg-video"
          src="/Magnificent_Eruption_Mk_II-overlay.mp4"
          autoPlay
          loop
          muted
          playsInline
        />
        <div className="hero-bg-overlay" />

        <div className="hero-content">
          <div className="home-badge">
            <Sparkles size={13} /> ISRO Aditya-L1 · PS-15 Mission
          </div>
          <h1 className="hero-title-giant">
            We predict solar flares{" "}
            <span className="accent-gold">60 minutes</span>{" "}
            before they strike.
          </h1>
          <p className="hero-subtitle-large">
            JWALA (Joint Waveband Alert & Light-curve Analyzer) fuses real-time
            SoLEXS soft X-ray and HEL1OS hard X-ray telemetry from ISRO's
            Aditya-L1 spacecraft to nowcast and forecast solar flares with
            conformal safety guarantees.
          </p>
          <div className="hero-cta-group">
            <Link to="/jwala/dashboard" className="home-primary-cta">
              Launch Mission Control <ArrowRight size={16} />
            </Link>
            <Link to="/jwala/about" className="home-secondary-cta">
              Learn More
            </Link>
          </div>
        </div>

        <div className="hero-stats-container">
          <div className="hero-stats-row">
            <div className="hero-stat">
              <span className="hero-stat-value">≤60s</span>
              <span className="hero-stat-label">Pipeline Latency</span>
            </div>
            <div className="hero-stat-divider" />
            <div className="hero-stat">
              <span className="hero-stat-value">90%</span>
              <span className="hero-stat-label">Conformal Coverage</span>
            </div>
            <div className="hero-stat-divider" />
            <div className="hero-stat">
              <span className="hero-stat-value">&lt;10%</span>
              <span className="hero-stat-label">False Alarm Rate</span>
            </div>
            <div className="hero-stat-divider" />
            <div className="hero-stat">
              <span className="hero-stat-value">99%</span>
              <span className="hero-stat-label">Availability SLO</span>
            </div>
          </div>
        </div>
      </section>

      {/* ── Content Wrapper (scrolls over the sticky Hero) ── */}
      <div className="scroll-wrapper-content">
        {/* Sticky Sparkles container locked to viewport height */}
        <div className="sparkles-sticky-container">
          <SparklesCore
            id="tsparticles-home"
            background="transparent"
            minSize={0.6}
            maxSize={1.4}
            speed={3.5}
            particleColor="#F6D337"
            particleDensity={80}
            className="sparkles-bg"
          />
        </div>

        {/* Dynamic WebGL burn transition edge */}
        <div className="burn-transition-webgl">
          <WebGLBurnTransition
            color="#0d0f12"
            transitionColor="#F6D337"
            noiseScale={0.37}
            noiseIntensity={0.3}
            scrollSensitivity={0.4}
            baseAnimationSpeed={0.1}
            edgeSoftness={0.4}
            bloomIntensity={0.5}
            bloomRadius={0.1}
            parallaxEnabled={true}
            movement={{ horizontal: "center", vertical: 0.5 }}
          />
        </div>

        <div className="home-container">

      {/* ── What is JWALA? ── */}
      <section className="home-callout">
        <p className="callout-text">
          <span className="accent-gold">JWALA</span> is an advanced full-stack
          application designed for automated solar flare detection,
          classification, and prediction — built for operators who need
          explainable, calibrated alerts they can trust with mission-critical
          hardware.
        </p>
      </section>

      {/* ── The Engine: Numbered Traits ── */}
      <section className="home-traits">
        <div className="section-header">
          <span className="section-subtitle">THE ENGINE</span>
          <h2 className="section-title">Seven Layers of Intelligence</h2>
        </div>

        <div className="traits-grid">
          <div className="trait-card">
            <div className="trait-number">01</div>
            <h3 className="trait-title">Data Ingestion & Preprocessing</h3>
            <p className="trait-body">
              Multi-source automated pipelines downloading FITS files from
              SoLEXS, HEL1OS, and GOES XRS. Cross-calibration,
              physics-based feature extraction, and time-series augmentation
              power every downstream model.
            </p>
          </div>

          <div className="trait-card">
            <div className="trait-number">02</div>
            <h3 className="trait-title">Instrument-Specific Detection</h3>
            <p className="trait-body">
              Dedicated detection pipelines for soft X-ray (SoLEXS) and hard
              X-ray (HEL1OS) channels. TCN encoders extract temporal features
              while phase detection identifies solar activity dynamics in real
              time.
            </p>
          </div>

          <div className="trait-card">
            <div className="trait-number">03</div>
            <h3 className="trait-title">Physics-Informed Neural Networks</h3>
            <p className="trait-body">
              PINN models embed solar physics constraints — including the
              Neupert effect — directly into the loss function, ensuring
              predictions respect the thermodynamic relationships between
              soft and hard X-ray emissions.
            </p>
          </div>

          <div className="trait-card">
            <div className="trait-number">04</div>
            <h3 className="trait-title">Three-Model Forecasting Ensemble</h3>
            <p className="trait-body">
              Causal LSTMs, Temporal Convolutional Networks, and fine-tuned
              TimesFM foundation models vote together. Pre-flare anomaly
              detection via MOMENT provides continuous scoring before any
              alert fires.
            </p>
          </div>

          <div className="trait-card">
            <div className="trait-number">05</div>
            <h3 className="trait-title">Dual Uncertainty Estimation</h3>
            <p className="trait-body">
              MAPIE conformal prediction intervals and Chronos-Bolt calibration
              deliver mathematically rigorous 90% coverage bands. No prediction
              ships without a quantified uncertainty envelope.
            </p>
          </div>

          <div className="trait-card">
            <div className="trait-number">06</div>
            <h3 className="trait-title">Agentic Intelligence & Explainability</h3>
            <p className="trait-body">
              DSPy-powered self-optimizing LLM reports and GraphRAG knowledge
              graph retrieval provide physics-based reasoning. SHAP feature
              attribution explains exactly why each forecast was triggered.
            </p>
          </div>

          <div className="trait-card">
            <div className="trait-number">07</div>
            <h3 className="trait-title">LangGraph Orchestration</h3>
            <p className="trait-body">
              Robust state-graph workflows manage agent operations, fallback
              states, and shadow runners. Walk-Forward CV pipelines validate
              every model before deployment.
            </p>
          </div>
        </div>
      </section>

      {/* ── Callout: Space Weather Context ── */}
      <section className="home-pitch">
        <div className="pitch-container">
          <span className="section-subtitle">THE CONTEXT</span>
          <h2 className="pitch-heading">Why does this matter?</h2>
          <p className="pitch-paragraph">
            In deep space operations, seconds decide the fate of high-sensitivity
            payload equipment. Extreme UV and X-ray flux from solar flares can
            cause immediate telemetry dropout, permanent sensor degradation, and
            severe orbital drag on Low Earth Orbit satellites.
          </p>
          <p className="pitch-paragraph">
            JWALA bridges the gap between raw astrophysical observations at the
            L1 Lagrange point and actionable ground alerts — giving satellite
            operators a <strong>60-minute window</strong> to safe instruments
            and orient arrays before peak solar storm impact.
          </p>

          <div className="aditya-badge">
            <Satellite size={18} className="badge-icon" />
            <div className="badge-text">
              <span className="badge-name">ISRO Aditya-L1 Spacecraft</span>
              <span className="badge-status">SoLEXS + HEL1OS · Lagrange Point 1</span>
            </div>
          </div>
        </div>
      </section>

      {/* ── Scenario Pain Points ── */}
      <section className="home-goals">
        <div className="section-header">
          <span className="section-subtitle">THE THREATS</span>
          <h2 className="section-title">What happens without early warning?</h2>
          <p className="section-desc">
            Real operational scenarios that motivate JWALA's 60-minute
            forecast horizon.
          </p>
        </div>

        <div className="goals-grid">
          <div className="goal-card">
            <span className="goal-tag"><Radio size={12} /> Scenario 01</span>
            <h3 className="goal-heading">"Our transponders went blind."</h3>
            <p className="goal-body">
              An unpredicted X-class flare ionizes communication bands, cutting
              command feeds and blinding transponders for hours.
            </p>
          </div>

          <div className="goal-card">
            <span className="goal-tag"><Cpu size={12} /> Scenario 02</span>
            <h3 className="goal-heading">"Flight software corrupted mid-orbit."</h3>
            <p className="goal-body">
              High-energy solar protons penetrate shielding, flipping bits in
              onboard computers and causing Single Event Upsets in critical systems.
            </p>
          </div>

          <div className="goal-card">
            <span className="goal-tag"><Globe size={12} /> Scenario 03</span>
            <h3 className="goal-heading">"Our orbit decayed 12 km overnight."</h3>
            <p className="goal-body">
              Surges in EUV flux heat the upper thermosphere, increasing
              atmospheric density and rapidly decaying LEO satellite orbits.
            </p>
          </div>

          <div className="goal-card">
            <span className="goal-tag"><Zap size={12} /> Scenario 04</span>
            <h3 className="goal-heading">"Solar array output dropped 8%."</h3>
            <p className="goal-body">
              Continuous unmitigated particle events degrade silicon cell
              structures, causing cumulative lifetime power decay across the array.
            </p>
          </div>

          <div className="goal-card">
            <span className="goal-tag"><AlertTriangle size={12} /> Scenario 05</span>
            <h3 className="goal-heading">"Ground transformers saturated."</h3>
            <p className="goal-body">
              Coronal mass ejections trigger geomagnetically induced currents
              that saturate terrestrial power transformer coils, risking grid collapse.
            </p>
          </div>

          <div className="goal-card">
            <span className="goal-tag"><Eye size={12} /> Scenario 06</span>
            <h3 className="goal-heading">"The team stopped trusting our alerts."</h3>
            <p className="goal-body">
              Continuous false alarms degrade warning credibility (FAR &gt; 10%),
              leading operators to ignore genuine threat alerts. JWALA's
              conformal calibration keeps FAR under budget.
            </p>
          </div>
        </div>
      </section>

      {/* ── Three-Dimensional Protection ── */}
      <section className="home-dimensions">
        <div className="section-header">
          <span className="section-subtitle">THE METHOD</span>
          <h2 className="section-title">Three Operational Layers</h2>
          <p className="section-desc">
            From photon detection at L1 to actionable ground alert — the entire
            pipeline runs in under 60 seconds.
          </p>
        </div>

        <div className="dimensions-grid">
          <div className="dimension-col">
            <div className="dimension-number">1</div>
            <div className="dimension-icon-badge">
              <Activity size={20} />
            </div>
            <h3 className="dimension-title">INGEST</h3>
            <p className="dimension-body">
              Automated FITS file downloading from PRADAN, cross-calibration
              between SoLEXS and HEL1OS channels, physics-based feature
              engineering, and time-series augmentation.
            </p>
            <ul className="dimension-list">
              <li>PRADAN Downloader</li>
              <li>GOES XRS Cross-Reference</li>
              <li>FITS Column Parser</li>
              <li>Seen-Files Deduplication</li>
            </ul>
          </div>

          <div className="dimension-col">
            <div className="dimension-number">2</div>
            <div className="dimension-icon-badge">
              <Brain size={20} />
            </div>
            <h3 className="dimension-title">PREDICT</h3>
            <p className="dimension-body">
              Live ensemble inference combining four model families. MOMENT
              anomaly scoring detects pre-flare precursors. Conformal
              calibration guarantees coverage.
            </p>
            <ul className="dimension-list">
              <li>Causal LSTM Network</li>
              <li>TimesFM Foundation Model</li>
              <li>PINN Solar Physics</li>
              <li>MAPIE Conformal Intervals</li>
            </ul>
          </div>

          <div className="dimension-col">
            <div className="dimension-number">3</div>
            <div className="dimension-icon-badge">
              <Shield size={20} />
            </div>
            <h3 className="dimension-title">PROTECT</h3>
            <p className="dimension-body">
              Emit ground crew alerts, publish DSPy natural-language bulletins,
              trigger spacecraft safe-mode commands, and log every decision
              with SHAP attribution.
            </p>
            <ul className="dimension-list">
              <li>DSPy LLM Bulletins</li>
              <li>GraphRAG Reasoning</li>
              <li>SHAP Feature Attribution</li>
              <li>LangGraph State Workflows</li>
            </ul>
          </div>
        </div>
      </section>

      {/* ── Historic Flares: Validated Events ── */}
      <section className="home-events">
        <div className="section-header">
          <span className="section-subtitle">BATTLE-TESTED</span>
          <h2 className="section-title">Validated Against Real Events</h2>
          <p className="section-desc">
            JWALA's pipeline is tested against the most powerful solar events
            of Solar Cycle 25.
          </p>
        </div>

        <div className="events-grid">
          <div className="event-card">
            <div className="event-card-header">
              <span className="event-class flare-x">X6.3</span>
              <span className="event-date-tag">2024-02-22</span>
            </div>
            <h3 className="event-title">Strongest Flare in 7 Years</h3>
            <p className="event-body">
              The February 2024 X6.3 flare from Active Region 3590 produced a
              simultaneous hard X-ray burst detectable by HEL1OS. JWALA's
              integration test replays this event end-to-end as the production
              readiness gate.
            </p>
          </div>

          <div className="event-card">
            <div className="event-card-header">
              <span className="event-class flare-x">X-class</span>
              <span className="event-date-tag">2024-05-10 — 14</span>
            </div>
            <h3 className="event-title">Geomagnetic Super-Storm</h3>
            <p className="event-body">
              A sequence of X-class flares and CMEs triggered the most intense
              geomagnetic storm since 2003, with aurora visible at mid-latitudes.
              Events like this motivate JWALA's multi-horizon forecasting
              ensemble.
            </p>
          </div>

          <div className="event-card">
            <div className="event-card-header">
              <span className="event-class flare-m">M/X</span>
              <span className="event-date-tag">Continuous</span>
            </div>
            <h3 className="event-title">Walk-Forward Validation</h3>
            <p className="event-body">
              Every model undergoes walk-forward cross-validation against
              historical GOES XRS catalogs (2024–2025), ensuring forecast
              skill is measured on truly out-of-sample data windows.
            </p>
          </div>
        </div>
      </section>

      {/* ── SLO Targets ── */}
      <section className="home-slos">
        <div className="section-header">
          <span className="section-subtitle">OPERATIONAL GUARANTEES</span>
          <h2 className="section-title">Service Level Objectives</h2>
          <p className="section-desc">
            Hard contractual targets monitored via Prometheus + Grafana.
          </p>
        </div>

        <div className="slos-grid">
          <div className="slo-card">
            <div className="slo-value">P99 &lt; 90s</div>
            <div className="slo-label">End-to-End Latency</div>
          </div>
          <div className="slo-card">
            <div className="slo-value">≥ 30 min</div>
            <div className="slo-label">Lead Time Target</div>
          </div>
          <div className="slo-card">
            <div className="slo-value">&lt; 10%</div>
            <div className="slo-label">False Alarm Rate</div>
          </div>
          <div className="slo-card">
            <div className="slo-value">≥ 80%</div>
            <div className="slo-label">M/X-class TPR</div>
          </div>
          <div className="slo-card">
            <div className="slo-value">99%</div>
            <div className="slo-label">System Availability</div>
          </div>
          <div className="slo-card">
            <div className="slo-value">60s</div>
            <div className="slo-label">Throughput Cadence</div>
          </div>
        </div>
      </section>

      {/* ── Tech Stack ── */}
      <section className="home-credentials">
        <div className="section-header">
          <span className="section-subtitle">FOUNDATIONS</span>
          <h2 className="section-title">Built With</h2>
        </div>

        <div className="credentials-grid">
          <div className="credential-col">
            <h3 className="credential-title">Predictive Core</h3>
            <ul className="credential-list">
              <li><CheckCircle size={14} className="list-icon" /> XGBoost Nowcasting</li>
              <li><CheckCircle size={14} className="list-icon" /> Causal LSTM Networks</li>
              <li><CheckCircle size={14} className="list-icon" /> TimesFM Foundation Model</li>
              <li><CheckCircle size={14} className="list-icon" /> PINN (Neupert Effect)</li>
              <li><CheckCircle size={14} className="list-icon" /> TCN Temporal Encoders</li>
            </ul>
          </div>

          <div className="credential-col">
            <h3 className="credential-title">Intelligence & Explainability</h3>
            <ul className="credential-list">
              <li><CheckCircle size={14} className="list-icon" /> DSPy Self-Optimizing Reports</li>
              <li><CheckCircle size={14} className="list-icon" /> GraphRAG Knowledge Graphs</li>
              <li><CheckCircle size={14} className="list-icon" /> SHAP Feature Attribution</li>
              <li><CheckCircle size={14} className="list-icon" /> MAPIE Conformal Calibration</li>
              <li><CheckCircle size={14} className="list-icon" /> Chronos-Bolt Uncertainty</li>
            </ul>
          </div>

          <div className="credential-col">
            <h3 className="credential-title">Infrastructure</h3>
            <ul className="credential-list">
              <li><CheckCircle size={14} className="list-icon" /> Docker Compose (dev/staging/prod)</li>
              <li><CheckCircle size={14} className="list-icon" /> Grafana + Prometheus Observability</li>
              <li><CheckCircle size={14} className="list-icon" /> Nginx Reverse Proxy</li>
              <li><CheckCircle size={14} className="list-icon" /> LangGraph Orchestration</li>
              <li><CheckCircle size={14} className="list-icon" /> Vite + React Dashboard</li>
            </ul>
          </div>

          <div className="credential-col">
            <h3 className="credential-title">Data Sources</h3>
            <ul className="credential-list">
              <li><CheckCircle size={14} className="list-icon" /> Aditya-L1 SoLEXS (Soft X-ray)</li>
              <li><CheckCircle size={14} className="list-icon" /> Aditya-L1 HEL1OS (Hard X-ray)</li>
              <li><CheckCircle size={14} className="list-icon" /> NOAA GOES XRS Archive</li>
              <li><CheckCircle size={14} className="list-icon" /> PRADAN ISSDC Portal</li>
              <li><CheckCircle size={14} className="list-icon" /> MOMENT Anomaly Scoring</li>
            </ul>
          </div>
        </div>
      </section>

      {/* ── Newsletter ── */}
      <section className="home-newsletter">
        <div className="newsletter-card">
          <span className="section-subtitle">UPDATES</span>
          <h2 className="newsletter-title">Subscribe to Space Weather Alerts</h2>
          <p className="newsletter-desc">
            Receive automated DSPy natural-language summaries of high-energy
            solar events directly in your inbox.
          </p>
          <form className="newsletter-form" onSubmit={(e) => e.preventDefault()}>
            <input
              type="email"
              placeholder="operator@missioncontrol.gov"
              className="newsletter-input"
              required
            />
            <button type="submit" className="newsletter-btn">
              Subscribe
            </button>
          </form>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className="home-footer">
        <div className="footer-content">
          <div className="footer-brand">
            <JwalaLogo size={18} glow={false} />{" "}
            <span className="accent-gold">JWALA</span>{" "}
            SOLAR FLARE FORECASTING SYSTEM
          </div>
          <p className="footer-meta">
            Developed under ISRO Aditya-L1 telemetry requirements · PS-15
            Integration Standards · Conformal safety verification protocols
          </p>
          <div className="footer-links">
            <Link to="/jwala/dashboard">Mission Control</Link>
            <Link to="/jwala/observatory">Observatory</Link>
            <Link to="/jwala/earth-impact">Earth Impact</Link>
            <Link to="/jwala/analytics">Analytics</Link>
            <Link to="/jwala/explainability">Explainability</Link>
            <Link to="/jwala/architecture">Architecture</Link>
            <Link to="/jwala/about">About</Link>
          </div>
        </div>
      </footer>
      </div>
      </div>
    </>
  );
}
