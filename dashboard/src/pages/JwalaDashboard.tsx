import { useState, useEffect, useRef, useMemo, lazy, Suspense } from "react";
import { Link, useParams, useNavigate } from "react-router-dom";
import {
  Shield,
  Bot,
  Settings,
  HelpCircle,
  Search,
  Bell,
  FileText,
  ChevronRight,
  Send,
  Mic,
  ExternalLink,
  Sparkles,
  Clock,
  Check,
  Compass,
  Globe,
  BarChart3,
  Brain,
  Info,
  Server,
  Eye,
  EyeOff,
  Copy,
  Terminal,
  Key,
  BookOpen,
  Zap,
  Code2
} from "lucide-react";
import JwalaLogo from "../components/JwalaLogo";
import "./JwalaDashboard.css";

// Lazy-load the forecasting page subcomponents to keep bundle chunks small
const MissionControlPage = lazy(() => import("./Dashboard"));
const ObservatoryPage = lazy(() => import("./Observatory"));
const EarthImpactPage = lazy(() => import("./EarthImpact"));
const AnalyticsPage = lazy(() => import("./Analytics"));
const ExplainabilityPage = lazy(() => import("./Explainability"));
const ArchitecturePage = lazy(() => import("./Architecture"));
const AboutPage = lazy(() => import("./About"));

// AI Assistant Chat interfaces
interface ChatMessage {
  id: number;
  sender: "user" | "assistant";
  text: string;
}

export default function JwalaDashboard() {
  const { tab } = useParams();
  const navigate = useNavigate();

  // Determine active tab, defaulting to "api-section"
  const activeMenu = useMemo(() => {
    return tab ? tab.toLowerCase() : "api-section";
  }, [tab]);

  const [currentTime, setCurrentTime] = useState("");

  // Update clock every second
  useEffect(() => {
    const updateClock = () => {
      const date = new Date();
      let hours = date.getHours();
      const minutes = date.getMinutes();
      const ampm = hours >= 12 ? "PM" : "AM";
      hours = hours % 12;
      hours = hours ? hours : 12;
      const minutesStr = minutes < 10 ? "0" + minutes : minutes;
      const days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
      const dayStr = days[date.getDay()];
      setCurrentTime(`${hours}:${minutesStr} ${ampm}, ${dayStr}`);
    };
    updateClock();
    const interval = setInterval(updateClock, 1000);
    return () => clearInterval(interval);
  }, []);

  // ── API Section State ──
  const [showApiKey, setShowApiKey] = useState(false);
  const [apiKeyCopied, setApiKeyCopied] = useState(false);
  const [activeCodeTab, setActiveCodeTab] = useState<"curl" | "python" | "javascript">("curl");

  const copyApiKey = () => {
    navigator.clipboard.writeText("jwala_sk_live_7x9KmNp2QrS4TvW6YzA8BcD3EfG5HjL");
    setApiKeyCopied(true);
    setTimeout(() => setApiKeyCopied(false), 2000);
  };

  // ── AI Assistant Chat State ──
  const [messages, setMessages] = useState<ChatMessage[]>([
    { id: 1, sender: "assistant", text: "Hey! I am the JWALA AI assistant. How can I help you with solar flare forecasting today?" },
    { id: 2, sender: "user", text: "What areas should I prioritize for monitoring?" },
    { id: 3, sender: "assistant", text: "⚡ Monitoring Focus:\n• Prioritize soft X-ray derivative trends from SoLEXS for early M/X-class precursors.\n• Watch hard/soft X-ray ratios from HEL1OS for impulsive phase detection.\n• Monitor MOMENT anomaly scores above 0.7 threshold." }
  ]);
  const [chatInput, setChatInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const chatBottomRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom of chat
  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  const handleSendMessage = (textToSend: string) => {
    if (!textToSend.trim()) return;

    const userMsg: ChatMessage = {
      id: Date.now(),
      sender: "user",
      text: textToSend
    };
    setMessages(prev => [...prev, userMsg]);
    setChatInput("");
    setIsTyping(true);

    setTimeout(() => {
      setIsTyping(false);
      let replyText = "I've analyzed the current solar conditions. The ensemble models indicate low-to-moderate flare probability in the next 60-minute window.";

      const query = textToSend.toLowerCase();
      if (query.includes("forecast") || query.includes("predict")) {
        replyText = "📡 Forecast Update:\n• Current ensemble probability: 23% (M-class), 4% (X-class)\n• All three models (LSTM, TCN, TimesFM) agree on moderate activity.\n• Conformal intervals holding at 90% coverage.";
      } else if (query.includes("api") || query.includes("endpoint")) {
        replyText = "🔌 API Info:\n• Base URL: https://api.jwala.isro.gov.in/v2\n• Use GET /forecast/current for real-time predictions.\n• Authentication requires Bearer token in Authorization header.";
      } else if (query.includes("alert") || query.includes("monitor")) {
        replyText = "🔔 Alert Status:\n• Active monitoring on SoLEXS + HEL1OS channels.\n• No elevated pre-flare signatures detected in the last 30 minutes.\n• MOMENT anomaly score: 0.31 (well below 0.7 threshold).";
      } else if (query.includes("insight") || query.includes("analysis") || query.includes("summary")) {
        replyText = "🧠 AI Analysis Summary:\n• Pre-flare pattern detection is active across all channels.\n• SHAP attribution highlights soft X-ray derivative as top feature (+0.34).\n• Model consensus at 91% — all ensemble members agree on current assessment.";
      }

      const assistantMsg: ChatMessage = {
        id: Date.now() + 1,
        sender: "assistant",
        text: replyText
      };
      setMessages(prev => [...prev, assistantMsg]);
    }, 1200);
  };

  const handleChipClick = (label: string) => {
    let question = "";
    if (label === "AI Hub") question = "Give me a summary of the latest AI-powered solar analysis.";
    if (label === "Solar Analysis") question = "What does the current forecast look like?";
    if (label === "Forecast Tips") question = "What should I monitor for early flare detection?";

    if (question) {
      handleSendMessage(question);
    }
  };

  const toggleRecording = () => {
    if (isRecording) {
      setIsRecording(false);
      setChatInput("What is the current flare probability?");
    } else {
      setIsRecording(true);
      setChatInput("Listening...");
      setTimeout(() => {
        if (isRecording) {
          setIsRecording(false);
          setChatInput("What is the current flare probability?");
        }
      }, 3000);
    }
  };

  // ══════════════════════════════════════════════════════════
  // Render: API Section (Developer Portal)
  // ══════════════════════════════════════════════════════════
  const renderApiSection = () => {
    const endpoints = [
      { method: "GET", path: "/api/v2/forecast/current", desc: "Get current solar flare forecast with confidence intervals", status: "stable" },
      { method: "GET", path: "/api/v2/forecast/history", desc: "Retrieve historical forecast data and accuracy metrics", status: "stable" },
      { method: "POST", path: "/api/v2/alerts/subscribe", desc: "Subscribe to real-time solar flare alert notifications", status: "stable" },
      { method: "GET", path: "/api/v2/telemetry/solexs", desc: "Raw SoLEXS soft X-ray telemetry stream", status: "beta" },
      { method: "GET", path: "/api/v2/telemetry/hel1os", desc: "Raw HEL1OS hard X-ray telemetry stream", status: "beta" },
      { method: "POST", path: "/api/v2/models/predict", desc: "Run ensemble prediction on custom flux data", status: "stable" },
      { method: "GET", path: "/api/v2/slo/status", desc: "Current SLO compliance metrics and error budgets", status: "stable" },
      { method: "GET", path: "/api/v2/explainability/shap", desc: "SHAP feature attribution for latest prediction", status: "beta" },
    ];

    const codeExamples: Record<string, string> = {
      curl: `curl -X GET "https://api.jwala.isro.gov.in/v2/forecast/current" \\
  -H "Authorization: Bearer jwala_sk_live_•••••••" \\
  -H "Content-Type: application/json"

# Response
{
  "forecast": {
    "class": "M2.4",
    "probability": 0.87,
    "lead_time_min": 58,
    "confidence_interval": [0.82, 0.92],
    "coverage_guarantee": "90%"
  },
  "metadata": {
    "model": "ensemble-v2.1",
    "timestamp": "2026-07-11T05:15:00Z"
  }
}`,
      python: `import requests

API_KEY = "jwala_sk_live_•••••••"
BASE_URL = "https://api.jwala.isro.gov.in/v2"

# Get current forecast
response = requests.get(
    f"{BASE_URL}/forecast/current",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
)

forecast = response.json()
print(f"Predicted class: {forecast['forecast']['class']}")
print(f"Probability: {forecast['forecast']['probability']}")
print(f"Lead time: {forecast['forecast']['lead_time_min']} min")`,
      javascript: `const API_KEY = "jwala_sk_live_•••••••";
const BASE_URL = "https://api.jwala.isro.gov.in/v2";

// Get current forecast
const response = await fetch(
  \`\${BASE_URL}/forecast/current\`,
  {
    headers: {
      "Authorization": \`Bearer \${API_KEY}\`,
      "Content-Type": "application/json"
    }
  }
);

const forecast = await response.json();
console.log("Predicted class:", forecast.forecast.class);
console.log("Probability:", forecast.forecast.probability);
console.log("Lead time:", forecast.forecast.lead_time_min, "min");`
    };

    return (
      <div className="api-section">
        {/* Header */}
        <div className="api-section-header">
          <div>
            <div className="api-section-badge">
              <Zap size={12} /> Developer Portal
            </div>
            <h1 className="api-section-title">JWALA API</h1>
            <p className="api-section-desc">
              Access solar flare forecasting, real-time telemetry, and predictive analytics through our REST API.
              Built for satellite operators and space weather researchers.
            </p>
          </div>
          <div className="api-version-pill">v2.1.0 — Stable</div>
        </div>

        {/* Quick Stats */}
        <div className="api-stats-row">
          <div className="api-stat-card">
            <div className="api-stat-val">2.4M</div>
            <div className="api-stat-label">Total Requests (30d)</div>
          </div>
          <div className="api-stat-card">
            <div className="api-stat-val">99.7%</div>
            <div className="api-stat-label">Success Rate</div>
          </div>
          <div className="api-stat-card">
            <div className="api-stat-val">45ms</div>
            <div className="api-stat-label">Avg Response Time</div>
          </div>
          <div className="api-stat-card">
            <div className="api-stat-val">1,000</div>
            <div className="api-stat-label">Requests / min (limit)</div>
          </div>
        </div>

        {/* API Key */}
        <div className="api-key-card">
          <div className="api-key-header">
            <div className="api-key-title-row">
              <Key size={16} style={{ color: "var(--jwala-accent)" }} />
              <span>API Key</span>
            </div>
            <span className="api-key-active-badge">Active</span>
          </div>
          <div className="api-key-display">
            <code className="api-key-code">
              {showApiKey ? "jwala_sk_live_7x9KmNp2QrS4TvW6YzA8BcD3EfG5HjL" : "jwala_sk_live_••••••••••••••••••••••••"}
            </code>
            <div className="api-key-btns">
              <button className="api-key-btn" onClick={() => setShowApiKey(!showApiKey)} title={showApiKey ? "Hide" : "Show"}>
                {showApiKey ? <EyeOff size={14} /> : <Eye size={14} />}
              </button>
              <button className="api-key-btn" onClick={copyApiKey} title="Copy">
                {apiKeyCopied ? <Check size={14} style={{ color: "var(--jwala-green)" }} /> : <Copy size={14} />}
              </button>
            </div>
          </div>
          <div className="api-key-meta">
            <span>Created: Jul 1, 2026</span>
            <span>Last used: 2 minutes ago</span>
          </div>
        </div>

        <div className="api-two-col">
          {/* Endpoints */}
          <div className="api-endpoints-card">
            <div className="api-card-hdr">
              <h2 className="api-card-hdg">
                <BookOpen size={16} style={{ color: "var(--jwala-accent)" }} />
                API Endpoints
              </h2>
              <span className="api-endpoint-count">{endpoints.length} endpoints</span>
            </div>
            <div className="api-endpoints-list">
              {endpoints.map((ep, i) => (
                <div className="api-endpoint-row" key={i}>
                  <div className="api-endpoint-left">
                    <span className={`api-method-badge method-${ep.method.toLowerCase()}`}>{ep.method}</span>
                    <code className="api-endpoint-path">{ep.path}</code>
                  </div>
                  <div className="api-endpoint-right">
                    <span className="api-endpoint-desc">{ep.desc}</span>
                    <span className={`api-status-badge status-${ep.status}`}>{ep.status}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Code Example */}
          <div className="api-code-card">
            <div className="api-card-hdr">
              <h2 className="api-card-hdg">
                <Terminal size={16} style={{ color: "var(--jwala-accent)" }} />
                Quick Start
              </h2>
            </div>
            <div className="api-code-tabs">
              {(["curl", "python", "javascript"] as const).map((lang) => (
                <button
                  key={lang}
                  className={`api-code-tab ${activeCodeTab === lang ? "active" : ""}`}
                  onClick={() => setActiveCodeTab(lang)}
                >
                  {lang === "curl" ? "cURL" : lang.charAt(0).toUpperCase() + lang.slice(1)}
                </button>
              ))}
            </div>
            <div className="api-code-block">
              <button
                className="api-code-copy"
                onClick={() => { navigator.clipboard.writeText(codeExamples[activeCodeTab]); }}
                title="Copy code"
              >
                <Copy size={13} />
              </button>
              <pre><code>{codeExamples[activeCodeTab]}</code></pre>
            </div>
          </div>
        </div>

        {/* Authentication & Rate Limits & Base URL */}
        <div className="api-info-row">
          <div className="api-info-card">
            <h3 className="api-info-title">
              <Shield size={16} style={{ color: "var(--jwala-accent)" }} />
              Authentication
            </h3>
            <p className="api-info-desc">
              All API requests require a Bearer token in the Authorization header.
              API keys can be generated from this dashboard and rotated at any time.
            </p>
            <div className="api-info-code">
              <code>Authorization: Bearer jwala_sk_live_•••</code>
            </div>
          </div>
          <div className="api-info-card">
            <h3 className="api-info-title">
              <Clock size={16} style={{ color: "var(--jwala-accent)" }} />
              Rate Limits
            </h3>
            <div className="api-rate-limits">
              <div className="api-rate-row">
                <span>Free Tier</span>
                <span className="api-rate-val">100 req/min</span>
              </div>
              <div className="api-rate-row">
                <span>Pro Tier</span>
                <span className="api-rate-val">1,000 req/min</span>
              </div>
              <div className="api-rate-row">
                <span>Enterprise</span>
                <span className="api-rate-val">Unlimited</span>
              </div>
            </div>
          </div>
          <div className="api-info-card">
            <h3 className="api-info-title">
              <Globe size={16} style={{ color: "var(--jwala-accent)" }} />
              Base URL
            </h3>
            <p className="api-info-desc">
              All API requests should be made to the following base URL. TLS 1.3 is required for all connections.
            </p>
            <div className="api-info-code">
              <code>https://api.jwala.isro.gov.in/v2</code>
            </div>
          </div>
        </div>
      </div>
    );
  };

  // ══════════════════════════════════════════════════════════
  // Render: AI Insights Page
  // ══════════════════════════════════════════════════════════
  const renderAiInsightsPage = () => {
    const insights = [
      { title: "Pre-Flare Detection", confidence: 94, desc: "MOMENT anomaly scoring detected elevated pre-flare signatures in the soft X-ray channel. Pattern consistent with M-class precursors.", type: "warning" },
      { title: "Model Consensus", confidence: 91, desc: "All three ensemble models (LSTM, TCN, TimesFM) agree on elevated activity probability within the next 60-minute window.", type: "success" },
      { title: "Conformal Calibration", confidence: 90, desc: "MAPIE coverage bands are holding at 90% target. Current prediction intervals are well-calibrated with no drift detected.", type: "info" },
      { title: "SHAP Attribution", confidence: 88, desc: "Top contributing features: soft X-ray derivative (+0.34), hard/soft ratio (+0.22), background flux level (+0.18).", type: "info" },
      { title: "Historical Pattern Match", confidence: 85, desc: "Current flux profile matches 73% similarity with the Feb 2024 X6.3 event precursor pattern from AR 3590.", type: "warning" },
      { title: "SLO Compliance", confidence: 99, desc: "All service level objectives within budget. False alarm rate at 7.2% (budget: 10%). Latency P99 at 62s (budget: 90s).", type: "success" },
    ];

    return (
      <div className="ai-insights-page">
        <div className="api-section-header">
          <div>
            <div className="api-section-badge">
              <Sparkles size={12} /> AI-Powered Analysis
            </div>
            <h1 className="api-section-title">AI Insights</h1>
            <p className="api-section-desc">
              Real-time AI-generated analysis from JWALA's ensemble models, DSPy reports, and SHAP explainability engine.
            </p>
          </div>
        </div>

        <div className="ai-insights-grid">
          {insights.map((insight, i) => (
            <div className={`ai-insight-card insight-${insight.type}`} key={i}>
              <div className="ai-insight-header">
                <h3 className="ai-insight-title">{insight.title}</h3>
                <div className="ai-insight-confidence">
                  <span className="ai-insight-pct">{insight.confidence}%</span>
                  <span className="ai-insight-label">confidence</span>
                </div>
              </div>
              <p className="ai-insight-desc">{insight.desc}</p>
              <div className="ai-insight-bar">
                <div className="ai-insight-bar-fill" style={{ width: `${insight.confidence}%` }} />
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  // ══════════════════════════════════════════════════════════
  // Render: AI Assistant Page (full-width chat)
  // ══════════════════════════════════════════════════════════
  const renderAiAssistantPage = () => {
    return (
      <div className="ai-assistant-page">
        <div className="api-section-header" style={{ marginBottom: "24px" }}>
          <div>
            <div className="api-section-badge">
              <Bot size={12} /> Conversational AI
            </div>
            <h1 className="api-section-title">AI Assistant</h1>
            <p className="api-section-desc">
              Chat with JWALA's AI to get real-time solar analysis, forecast explanations, and API guidance.
            </p>
          </div>
        </div>

        <div className="jwala-card" style={{ maxWidth: "900px" }}>
          <div className="jwala-card-header">
            <div className="jwala-card-title">
              <Bot size={16} style={{ color: "var(--jwala-accent)" }} />
              JWALA AI Chat
            </div>
            <div style={{ display: "flex", gap: "8px" }}>
              <button className="jwala-card-action-btn">
                <FileText size={14} />
              </button>
              <button className="jwala-card-action-btn">
                <ExternalLink size={14} />
              </button>
            </div>
          </div>

          <div className="jwala-chat-container">
            <div className="jwala-chat-messages">
              {messages.map((m) => (
                <div key={m.id} className={`jwala-chat-msg ${m.sender}`}>
                  <div className="jwala-msg-header">
                    {m.sender === "user" ? "You" : "JWALA AI"}
                  </div>
                  <div className="jwala-msg-bubble">
                    {m.text.split("\n").map((line, i) => (
                      <div key={i}>{line}</div>
                    ))}
                  </div>
                </div>
              ))}
              {isTyping && (
                <div className="jwala-chat-msg assistant">
                  <div className="jwala-msg-header">JWALA AI is typing</div>
                  <div className="jwala-msg-bubble" style={{ padding: 0 }}>
                    <div className="jwala-typing-indicator">
                      <span className="jwala-typing-dot"></span>
                      <span className="jwala-typing-dot"></span>
                      <span className="jwala-typing-dot"></span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={chatBottomRef} />
            </div>

            <div>
              <div className="jwala-chat-chips">
                <button className="jwala-chat-chip" onClick={() => handleChipClick("AI Hub")}>AI Hub</button>
                <button className="jwala-chat-chip" onClick={() => handleChipClick("Solar Analysis")}>Solar Analysis</button>
                <button className="jwala-chat-chip" onClick={() => handleChipClick("Forecast Tips")}>Forecast Tips</button>
              </div>

              <div className="jwala-chat-input-wrapper">
                <button
                  type="button"
                  className={`jwala-chat-mic-btn ${isRecording ? "active" : ""}`}
                  onClick={toggleRecording}
                  title={isRecording ? "Stop Recording" : "Voice Input"}
                >
                  <Mic size={16} />
                </button>
                <input
                  type="text"
                  placeholder="Ask about solar forecasts, API usage, or monitoring..."
                  className="jwala-chat-input"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") handleSendMessage(chatInput);
                  }}
                />
                <button
                  className="jwala-chat-send-btn"
                  onClick={() => handleSendMessage(chatInput)}
                  disabled={!chatInput.trim()}
                >
                  <Send size={14} />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  // Render a simulated layout panel for minor tabs
  const renderMockPage = (title: string) => {
    const displayTitle = title
      .split("-")
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");

    return (
      <div className="jwala-card" style={{ padding: "40px", textAlign: "center" }}>
        <div style={{ color: "var(--jwala-accent)", marginBottom: "16px" }}>
          <Sparkles size={48} style={{ margin: "0 auto" }} />
        </div>
        <h2>{displayTitle} Page</h2>
        <p style={{ color: "var(--jwala-text-secondary)", marginTop: "8px" }}>
          This is a simulated panel for the <strong>{displayTitle}</strong> dashboard item. Toggle other sections in the sidebar menu.
        </p>
      </div>
    );
  };

  // Dynamic content router for rendering active tab
  const renderContent = () => {
    switch (activeMenu) {
      case "api-section":
      case "overview":
        return renderApiSection();
      case "ai-insights":
        return renderAiInsightsPage();
      case "ai-assistant":
        return renderAiAssistantPage();
      case "dashboard":
        return <MissionControlPage />;
      case "observatory":
        return <ObservatoryPage />;
      case "earth-impact":
        return <EarthImpactPage />;
      case "analytics":
        return <AnalyticsPage />;
      case "explainability":
        return <ExplainabilityPage />;
      case "architecture":
        return <ArchitecturePage />;
      case "about":
        return <AboutPage />;
      default:
        return renderMockPage(activeMenu);
    }
  };

  return (
    <div className="jwala-dashboard-page">
      {/* ── Sidebar Navigation ── */}
      <aside className="jwala-sidebar">
        <div>
          {/* Brand */}
          <div className="jwala-brand-section">
            <Link to="/" className="jwala-brand">
              <span className="jwala-brand-logo">
                <JwalaLogo size={24} />
              </span>
              <span className="jwala-brand-text">JWALA</span>
            </Link>
            <button className="jwala-sidebar-collapse">
              <ChevronRight size={16} style={{ transform: "rotate(180deg)" }} />
            </button>
          </div>

          {/* Sidebar Menu */}
          <nav className="jwala-sidebar-menu">
            {/* SOLAR FORECASTING */}
            <div className="jwala-menu-group">
              <div className="jwala-menu-title">Solar Forecasting</div>
              <div
                className={`jwala-menu-item ${activeMenu === "dashboard" ? "active" : ""}`}
                onClick={() => navigate("/jwala/dashboard")}
              >
                <Compass size={18} />
                Mission Control
              </div>
              <div
                className={`jwala-menu-item ${activeMenu === "observatory" ? "active" : ""}`}
                onClick={() => navigate("/jwala/observatory")}
              >
                <Globe size={18} />
                Observatory
              </div>
              <div
                className={`jwala-menu-item ${activeMenu === "earth-impact" ? "active" : ""}`}
                onClick={() => navigate("/jwala/earth-impact")}
              >
                <Shield size={18} />
                Earth Impact
              </div>
              <div
                className={`jwala-menu-item ${activeMenu === "analytics" ? "active" : ""}`}
                onClick={() => navigate("/jwala/analytics")}
              >
                <BarChart3 size={18} />
                Analytics
              </div>
              <div
                className={`jwala-menu-item ${activeMenu === "explainability" ? "active" : ""}`}
                onClick={() => navigate("/jwala/explainability")}
              >
                <Brain size={18} />
                Explainability
              </div>
              <div
                className={`jwala-menu-item ${activeMenu === "architecture" ? "active" : ""}`}
                onClick={() => navigate("/jwala/architecture")}
              >
                <Server size={18} />
                Architecture
              </div>
              <div
                className={`jwala-menu-item ${activeMenu === "about" ? "active" : ""}`}
                onClick={() => navigate("/jwala/about")}
              >
                <Info size={18} />
                About
              </div>
            </div>

            {/* API & AI */}
            <div className="jwala-menu-group">
              <div className="jwala-menu-title">API &amp; AI</div>
              <div
                className={`jwala-menu-item ${activeMenu === "api-section" || activeMenu === "overview" ? "active" : ""}`}
                onClick={() => navigate("/jwala/api-section")}
              >
                <Code2 size={18} />
                API Section
              </div>
              <div
                className={`jwala-menu-item ${activeMenu === "ai-insights" ? "active" : ""}`}
                onClick={() => navigate("/jwala/ai-insights")}
              >
                <Sparkles size={18} />
                AI Insights
              </div>
              <div
                className={`jwala-menu-item ${activeMenu === "ai-assistant" ? "active" : ""}`}
                onClick={() => navigate("/jwala/ai-assistant")}
              >
                <Bot size={18} />
                AI Assistant
              </div>
            </div>

            {/* TOOLS */}
            <div className="jwala-menu-group">
              <div className="jwala-menu-title">Tools</div>
              <div
                className={`jwala-menu-item ${activeMenu === "settings" ? "active" : ""}`}
                onClick={() => navigate("/jwala/settings")}
              >
                <Settings size={18} />
                Settings
              </div>
              <div
                className={`jwala-menu-item ${activeMenu === "help-center" ? "active" : ""}`}
                onClick={() => navigate("/jwala/help-center")}
              >
                <HelpCircle size={18} />
                Help center
              </div>
            </div>
          </nav>
        </div>

        {/* Upgrade Card */}
        <div className="jwala-upgrade-card">
          <div className="jwala-upgrade-badge">
            <Sparkles size={10} /> Pro Version
          </div>
          <div className="jwala-upgrade-title">Upgrade to Pro</div>
          <div className="jwala-upgrade-text">
            Get advanced API access, higher rate limits, and priority support for your space weather operations.
          </div>
          <div className="jwala-upgrade-actions">
            <button className="jwala-upgrade-btn">Upgrade</button>
            <a href="#learn" className="jwala-upgrade-link">Learn More</a>
          </div>
        </div>
      </aside>

      {/* ── Main Panel Content ── */}
      <main className="jwala-main-panel">
        <header className="jwala-topbar">
          <div className="jwala-topbar-left">
            <a href="#reports" className="jwala-topbar-pill">
              <FileText size={16} style={{ color: "var(--jwala-accent)" }} />
              Reports
            </a>
            <div className="jwala-topbar-pill">
              <Clock size={16} style={{ color: "var(--jwala-accent)" }} />
              <span>{currentTime || "12:37 PM, Wed"}</span>
            </div>
          </div>

          <div className="jwala-topbar-right">
            <div className="jwala-search-wrapper">
              <Search className="jwala-search-icon" />
              <input
                type="text"
                placeholder="Search endpoints, docs, forecasts..."
                className="jwala-search-input"
              />
            </div>
            <button className="jwala-icon-btn">
              <Bell size={18} />
            </button>
            <img
              src="/user_avatar.png"
              alt="User profile"
              className="jwala-profile-avatar"
              onError={(e) => {
                e.currentTarget.src = "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=100&h=100&fit=crop";
              }}
            />
          </div>
        </header>

        <div className="jwala-content">
          <Suspense fallback={<div style={{ padding: "40px", textAlign: "center", opacity: 0.5 }}>Loading component...</div>}>
            {renderContent()}
          </Suspense>
        </div>
      </main>
    </div>
  );
}
