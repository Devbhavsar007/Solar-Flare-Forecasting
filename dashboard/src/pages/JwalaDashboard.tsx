import React, { useState, useEffect, useRef, useMemo, lazy, Suspense } from "react";
import { Link, useParams, useNavigate } from "react-router-dom";
import {
  Activity,
  Wallet,
  CreditCard,
  Shield,
  TrendingUp,
  Bot,
  Settings,
  HelpCircle,
  Search,
  Bell,
  FileText,
  ChevronRight,
  Plus,
  Minus,
  Send,
  Mic,
  ExternalLink,
  MoreHorizontal,
  Sparkles,
  Clock,
  ArrowUpRight,
  Check,
  UserPlus,
  Compass,
  Globe,
  BarChart3,
  Brain,
  Info,
  Server
} from "lucide-react";
import JwalaLogo from "../components/JwalaLogo";
import {
  AreaChart,
  Area,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Tooltip
} from "recharts";
import "./JwalaDashboard.css";

// Lazy-load the forecasting page subcomponents to keep bundle chunks small
const MissionControlPage = lazy(() => import("./Dashboard"));
const ObservatoryPage = lazy(() => import("./Observatory"));
const EarthImpactPage = lazy(() => import("./EarthImpact"));
const AnalyticsPage = lazy(() => import("./Analytics"));
const ExplainabilityPage = lazy(() => import("./Explainability"));
const ArchitecturePage = lazy(() => import("./Architecture"));
const AboutPage = lazy(() => import("./About"));

// Member Avatars data
interface Member {
  id: number;
  name: string;
  role: string;
  avatar: string;
}

const MEMBERS: Member[] = [
  { id: 1, name: "Rahul (Primary)", role: "Subscriber", avatar: "/user_avatar.png" },
  { id: 2, name: "Neha", role: "Spouse", avatar: "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=100&h=100&fit=crop" },
  { id: 3, name: "Aarav", role: "Son (10y)", avatar: "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=100&h=100&fit=crop" },
  { id: 4, name: "Diya", role: "Daughter (7y)", avatar: "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=100&h=100&fit=crop" },
  { id: 5, name: "Ramesh", role: "Father (Senior)", avatar: "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=100&h=100&fit=crop" }
];

// Transaction Records interfaces
interface RecordItem {
  id: number;
  category: string;
  value: number;
  status: "Paid" | "Claimed" | "Pending";
  date: string;
}

// AI Assistant Chat interfaces
interface ChatMessage {
  id: number;
  sender: "user" | "assistant";
  text: string;
}

export default function JwalaDashboard() {
  const { tab } = useParams();
  const navigate = useNavigate();

  // Determine active tab, defaulting to "overview"
  const activeMenu = useMemo(() => {
    return tab ? tab.toLowerCase() : "overview";
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

  // Card 2 State: Health Savings adjusters & hover avatar
  const [healthSavings, setHealthSavings] = useState(8000);
  const [hoveredAvatar, setHoveredAvatar] = useState<number | null>(null);

  const incrementSavings = () => setHealthSavings(prev => prev + 250);
  const decrementSavings = () => setHealthSavings(prev => Math.max(0, prev - 250));

  // Compute progress for savings limit
  const savingsPct = useMemo(() => {
    return Math.min(100, Math.round((healthSavings / 10000) * 100));
  }, [healthSavings]);

  // Card 3 State: Smart Health Finance
  const [timeframe, setTimeframe] = useState("Monthly");

  const pieData = useMemo(() => {
    if (timeframe === "Weekly") {
      return [
        { name: "Hospital Visits", value: 35, color: "#F6D337" },
        { name: "Medication", value: 45, color: "#14b8a6" },
        { name: "Lab Tests", value: 20, color: "#f59e0b" }
      ];
    } else if (timeframe === "Yearly") {
      return [
        { name: "Hospital Visits", value: 50, color: "#F6D337" },
        { name: "Medication", value: 25, color: "#14b8a6" },
        { name: "Lab Tests", value: 25, color: "#f59e0b" }
      ];
    }
    // Monthly default
    return [
      { name: "Hospital Visits", value: 40, color: "#F6D337" },
      { name: "Medication", value: 35, color: "#14b8a6" },
      { name: "Lab Tests", value: 25, color: "#f59e0b" }
    ];
  }, [timeframe]);

  // Total value calculation for center of donut
  const totalSpendPct = useMemo(() => {
    return pieData.reduce((acc, curr) => acc + curr.value, 0);
  }, [pieData]);

  // Area Chart Data representing $12,500 with +6.2% growth
  const areaData = [
    { name: "W1", value: 11200 },
    { name: "W2", value: 11500 },
    { name: "W3", value: 11900 },
    { name: "W4", value: 12500 }
  ];

  // Card 4 State: Spending Limits & Verification
  const [verifiedList, setVerifiedList] = useState({
    insights: true,
    insurance: false,
    limit: false
  });

  const toggleVerify = (key: "insights" | "insurance" | "limit") => {
    setVerifiedList(prev => ({ ...prev, [key]: !prev[key] }));
  };

  // Card 5 State: Health Records Table
  const [filterTab, setFilterTab] = useState<"All" | "Pending" | "Completed">("All");
  const [tableSearch, setTableSearch] = useState("");
  const [records, setRecords] = useState<RecordItem[]>([
    { id: 1, category: "Consultation", value: 120.00, status: "Paid", date: "2026-07-08" },
    { id: 2, category: "Pharmacy", value: 265.00, status: "Claimed", date: "2026-07-06" },
    { id: 3, category: "Insurance", value: 1200.00, status: "Pending", date: "2026-07-04" },
    { id: 4, category: "Lab Test", value: 300.00, status: "Paid", date: "2026-07-01" }
  ]);

  // Toggle status of a record on click
  const cycleStatus = (id: number) => {
    setRecords(prev =>
      prev.map(r => {
        if (r.id !== id) return r;
        const nextStatus: RecordItem["status"] =
          r.status === "Pending" ? "Claimed" : r.status === "Claimed" ? "Paid" : "Pending";
        return { ...r, status: nextStatus };
      })
    );
  };

  // Filter & Search Records
  const filteredRecords = useMemo(() => {
    return records.filter(r => {
      const matchesSearch = r.category.toLowerCase().includes(tableSearch.toLowerCase());
      if (filterTab === "All") return matchesSearch;
      if (filterTab === "Pending") return matchesSearch && r.status === "Pending";
      if (filterTab === "Completed") return matchesSearch && (r.status === "Paid" || r.status === "Claimed");
      return matchesSearch;
    });
  }, [records, filterTab, tableSearch]);

  // Form states for adding a custom record
  const [showAddForm, setShowAddForm] = useState(false);
  const [newCategory, setNewCategory] = useState("");
  const [newValue, setNewValue] = useState("");
  const [newStatus, setNewStatus] = useState<RecordItem["status"]>("Pending");

  const handleAddRecord = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newCategory || !newValue) return;
    const item: RecordItem = {
      id: Date.now(),
      category: newCategory,
      value: parseFloat(newValue) || 0,
      status: newStatus,
      date: new Date().toISOString().split("T")[0]
    };
    setRecords(prev => [item, ...prev]);
    setNewCategory("");
    setNewValue("");
    setShowAddForm(false);
  };

  // Card 6 State: Interactive AI Assistant Chat
  const [messages, setMessages] = useState<ChatMessage[]>([
    { id: 1, sender: "assistant", text: "Hey! I am the JWALA AI assistant. How can I help optimize your health coverage today?" },
    { id: 2, sender: "user", text: "What areas should I prioritize for managing my insurance?" },
    { id: 3, sender: "assistant", text: "⚡ Insurance Focus:\n• Prioritize comprehensive health coverage and read policy details.\n• Avoid: Unnecessary coverage add-ons and premium high-deductible plans." }
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

    // Add user message
    const userMsg: ChatMessage = {
      id: Date.now(),
      sender: "user",
      text: textToSend
    };
    setMessages(prev => [...prev, userMsg]);
    setChatInput("");
    setIsTyping(true);

    // Simulate AI thinking and typing response
    setTimeout(() => {
      setIsTyping(false);
      let replyText = "I've analyzed your health portfolio. It's recommended to allocate 40% to Hospitals and keep co-pays below $500.";
      
      const query = textToSend.toLowerCase();
      if (query.includes("prioritize") || query.includes("insurance")) {
        replyText = "⚡ Insurance Focus:\n• Prioritize critical care coverage and review network hospital listings.\n• Ensure co-pays remain under $300 to minimize out-of-pocket expenses.";
      } else if (query.includes("finance") || query.includes("spending")) {
        replyText = "📊 Financial Tip:\n• You have spent $1,390 out of your $1,600 monthly limit. You have 2 days remaining in your cycle, which indicates excellent budget compliance!";
      } else if (query.includes("saving") || query.includes("wallet")) {
        replyText = "💰 Savings recommendation:\n• Consider increasing your Health Savings to at least $9k to benefit from additional tax exemptions and protect against critical care emergencies.";
      }

      const assistantMsg: ChatMessage = {
        id: Date.now() + 1,
        sender: "assistant",
        text: replyText
      };
      setMessages(prev => [...prev, assistantMsg]);
    }, 1200);
  };

  // Trigger quick prompt chip
  const handleChipClick = (label: string) => {
    let question = "";
    if (label === "AI Hub") question = "Provide a summary of my AI-powered health insights.";
    if (label === "Health Finance") question = "How is my spending looking this month?";
    if (label === "Insurance Tips") question = "What policy details should I double check?";
    
    if (question) {
      handleSendMessage(question);
    }
  };

  // Voice recording simulation
  const toggleRecording = () => {
    if (isRecording) {
      setIsRecording(false);
      setChatInput("How can I minimize my medication expense?");
    } else {
      setIsRecording(true);
      setChatInput("Listening...");
      setTimeout(() => {
        if (isRecording) {
          setIsRecording(false);
          setChatInput("How can I minimize my medication expense?");
        }
      }, 3000);
    }
  };

  // Render the Overview Widgets Grid (Health & Finance Dashboard)
  const renderOverviewDashboard = () => {
    return (
      <div className="jwala-grid">
        {/* CARD 1: Total Amount & AI Glass Widget */}
        <div className="jwala-card">
          <div className="jwala-card-header">
            <div style={{ fontSize: "0.75rem", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--jwala-text-secondary)" }}>
              Total Value
            </div>
            <button className="jwala-card-action-btn">
              <MoreHorizontal size={16} />
            </button>
          </div>
          <div className="jwala-balance-amount">$18,540.00</div>
          <div className="jwala-balance-change">
            <span>Yearly Avg: <strong>$15,200.00</strong></span>
            <span className="jwala-change-pct">+12.4%</span>
          </div>
          <div className="jwala-balance-footer">
            <span>Compared to last year</span>
            <a href="#explain" className="jwala-help-link">
              How is it working? <ArrowUpRight size={12} />
            </a>
          </div>

          <div className="jwala-glass-widget">
            <div className="jwala-glass-widget-title">AI Assistant</div>
            <div className="jwala-glass-widget-desc">is analyzing your insurance usage...</div>
            <img 
              src="/ai_assistant_glass.png" 
              alt="Glassmorphic waves" 
              className="jwala-glass-widget-img" 
              onError={(e) => {
                e.currentTarget.src = "https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?w=200&h=200&fit=crop";
              }}
            />
          </div>
        </div>

        {/* CARD 2: Insurance & Health Metrics */}
        <div className="jwala-card">
          <div className="jwala-card-header">
            <div className="jwala-card-title">
              <Shield size={16} style={{ color: "var(--jwala-accent)" }} />
              Insurance &amp; Health Metrics
            </div>
            <button className="jwala-card-action-btn">
              <MoreHorizontal size={16} />
            </button>
          </div>

          <div className="jwala-metric-box">
            <div>
              <div className="jwala-metric-val">$12,240.00</div>
              <div className="jwala-metric-label">Coverage Balance</div>
            </div>
            <div className="jwala-progress-circle-wrapper">
              <svg width="50" height="50" viewBox="0 0 36 36">
                <path
                  className="jwala-circle-bg"
                  d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                  fill="none"
                  strokeWidth="2.5"
                />
                <path
                  className="jwala-circle-fill"
                  strokeDasharray="18, 100"
                  d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                  fill="none"
                  strokeWidth="2.5"
                />
                <text x="18" y="21.5" className="jwala-circle-text" textAnchor="middle">18%</text>
              </svg>
            </div>
          </div>

          <p className="jwala-metric-text-desc">
            AI-powered insurance and health insights optimized for your family.
          </p>

          <div className="jwala-sub-metrics">
            <div className="jwala-sub-metric">
              <span style={{ fontSize: "0.68rem", color: "var(--jwala-text-secondary)" }}>Co-pay Balance</span>
              <span className="jwala-sub-metric-val">$265</span>
            </div>
            <div className="jwala-sub-metric">
              <span style={{ fontSize: "0.68rem", color: "var(--jwala-text-secondary)" }}>Health Savings</span>
              <div className="jwala-sub-metric-val">
                <span>${(healthSavings/1000).toFixed(0)}k</span>
                <div className="jwala-sub-metric-ctrls">
                  <button className="jwala-ctrl-btn" onClick={decrementSavings}>
                    <Minus size={10} />
                  </button>
                  <button className="jwala-ctrl-btn" onClick={incrementSavings}>
                    <Plus size={10} />
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Avatar List & Add */}
          <div className="jwala-member-row">
            <div className="jwala-avatars-overlapping">
              {MEMBERS.map((m, idx) => (
                <div 
                  key={m.id} 
                  style={{ position: "relative" }}
                  onMouseEnter={() => setHoveredAvatar(m.id)}
                  onMouseLeave={() => setHoveredAvatar(null)}
                >
                  <img 
                    src={m.avatar} 
                    alt={m.name} 
                    className="jwala-overlap-avatar"
                    style={{ zIndex: 5 - idx }}
                    onError={(e) => {
                      e.currentTarget.src = "https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?w=100&h=100&fit=crop";
                    }}
                  />
                  {hoveredAvatar === m.id && (
                    <div className="jwala-avatar-popover">
                      <strong>{m.name}</strong>
                      <div style={{ color: "var(--jwala-accent)", marginTop: "2px" }}>{m.role}</div>
                    </div>
                  )}
                </div>
              ))}
              <button className="jwala-member-add" title="Add Member">
                <UserPlus size={12} />
              </button>
            </div>
            <span style={{ fontSize: "0.7rem", color: "var(--jwala-text-secondary)", fontWeight: "600" }}>
              Limit: {savingsPct}%
            </span>
          </div>
        </div>

        {/* CARD 3: Smart Health Finance */}
        <div className="jwala-card">
          <div className="jwala-card-header">
            <div className="jwala-card-title">Smart Health Finance</div>
            <select 
              className="jwala-chart-select" 
              value={timeframe} 
              onChange={(e) => setTimeframe(e.target.value)}
            >
              <option value="Weekly">Weekly</option>
              <option value="Monthly">Monthly</option>
              <option value="Yearly">Yearly</option>
            </select>
          </div>

          <div className="jwala-chart-flex">
            <div className="jwala-radial-chart-container">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={45}
                    outerRadius={60}
                    paddingAngle={3}
                    dataKey="value"
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
              <div className="jwala-radial-center-text">
                <span className="jwala-radial-center-val">{totalSpendPct}%</span>
                <span className="jwala-radial-center-lbl">Total</span>
              </div>
            </div>

            <div className="jwala-radial-legend">
              {pieData.map((item, idx) => (
                <div className="jwala-legend-item" key={idx}>
                  <span className="jwala-legend-dot" style={{ backgroundColor: item.color }} />
                  <div className="jwala-legend-text">
                    <span className="jwala-legend-val">{item.value}%</span>
                    <span className="jwala-legend-lbl">{item.name}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Bottom line chart */}
          <div className="jwala-line-chart-wrapper">
            <div className="jwala-line-chart-header">
              <div>
                <div style={{ fontSize: "0.68rem", color: "var(--jwala-text-secondary)" }}>Growth Trend</div>
                <div className="jwala-line-chart-val">$12,500.00</div>
              </div>
              <span className="jwala-line-chart-change">+6.2%</span>
            </div>
            <div style={{ width: "100%", height: 50 }}>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={areaData} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
                  <defs>
                    <linearGradient id="colorYellow" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="var(--jwala-accent)" stopOpacity={0.4}/>
                      <stop offset="95%" stopColor="var(--jwala-accent)" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <Tooltip 
                    contentStyle={{ backgroundColor: "#1e2230", border: "1px solid var(--jwala-border)", borderRadius: "8px" }}
                    labelStyle={{ color: "var(--jwala-text-secondary)" }}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="value" 
                    stroke="var(--jwala-accent)" 
                    strokeWidth={2}
                    fillOpacity={1} 
                    fill="url(#colorYellow)" 
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* CARD 4: Healthcare Financial Tracking */}
        <div className="jwala-card">
          <div className="jwala-card-header">
            <div className="jwala-card-title">Healthcare Financial Tracking</div>
            <span className="jwala-insight-badge" style={{ backgroundColor: "rgba(16, 185, 129, 0.1)", borderColor: "rgba(16, 185, 129, 0.3)", color: "var(--jwala-green)" }}>
              +8%
            </span>
          </div>

          <div style={{ marginBottom: "16px" }}>
            <div style={{ fontSize: "0.68rem", color: "var(--jwala-text-secondary)", marginBottom: "4px" }}>Monthly Expenses</div>
            <div className="jwala-metric-val">$1,390 <span style={{ fontSize: "0.8rem", color: "var(--jwala-text-muted)" }}>/ $1,600</span></div>
            <div style={{ fontSize: "0.68rem", color: "var(--jwala-text-muted)", marginTop: "2px" }}>2 days into cycle</div>
          </div>

          <div className="jwala-tracking-limit">
            <span>Smart Spending Limits</span>
            <span className="jwala-tracking-limit-val">85% Capacity</span>
          </div>

          {/* Multibar Spending segments */}
          <div className="jwala-multibar">
            <div className="jwala-multibar-segment" style={{ width: "40%", backgroundColor: "var(--jwala-accent)" }} title="Hospitals" />
            <div className="jwala-multibar-segment" style={{ width: "35%", backgroundColor: "var(--jwala-teal)" }} title="Medication" />
            <div className="jwala-multibar-segment" style={{ width: "25%", backgroundColor: "var(--jwala-amber)" }} title="Lab Tests" />
            <div className="jwala-multibar-segment" style={{ width: "20%", backgroundColor: "rgba(255, 255, 255, 0.3)" }} title="Other" />
          </div>

          <div className="jwala-tracking-categories">
            <div className="jwala-category-pill">
              <span className="jwala-category-dot" style={{ backgroundColor: "var(--jwala-accent)" }} />
              <span>Hospitals <span className="jwala-category-percentage">40%</span></span>
            </div>
            <div className="jwala-category-pill">
              <span className="jwala-category-dot" style={{ backgroundColor: "var(--jwala-teal)" }} />
              <span>Medications <span className="jwala-category-percentage">35%</span></span>
            </div>
            <div className="jwala-category-pill">
              <span className="jwala-category-dot" style={{ backgroundColor: "var(--jwala-amber)" }} />
              <span>Lab Tests <span className="jwala-category-percentage">25%</span></span>
            </div>
            <div className="jwala-category-pill">
              <span className="jwala-category-dot" style={{ backgroundColor: "rgba(255, 255, 255, 0.3)" }} />
              <span>Other <span className="jwala-category-percentage">20%</span></span>
            </div>
          </div>

          {/* Insights List */}
          <div className="jwala-insights-panel">
            <div 
              className={`jwala-insight-pill ${verifiedList.insights ? "verified" : ""}`}
              onClick={() => toggleVerify("insights")}
            >
              <div className="jwala-insight-left">
                {verifiedList.insights ? <Check className="jwala-insight-icon" /> : <Shield className="jwala-insight-icon" />}
                <span className="jwala-insight-text">Insights</span>
              </div>
              <span style={{ fontSize: "0.68rem", fontWeight: "600" }}>70% Care, 20% Preventive</span>
            </div>

            <div 
              className={`jwala-insight-pill ${verifiedList.insurance ? "verified" : ""}`}
              onClick={() => toggleVerify("insurance")}
            >
              <div className="jwala-insight-left">
                {verifiedList.insurance ? <Check className="jwala-insight-icon" /> : <Shield className="jwala-insight-icon" />}
                <span className="jwala-insight-text">Verify</span>
              </div>
              <span style={{ fontSize: "0.68rem", fontWeight: "600" }}>60% Insurance, 40% OOP</span>
            </div>

            <div 
              className={`jwala-insight-pill ${verifiedList.limit ? "verified" : ""}`}
              onClick={() => toggleVerify("limit")}
            >
              <div className="jwala-insight-left">
                {verifiedList.limit ? <Check className="jwala-insight-icon" /> : <Shield className="jwala-insight-icon" />}
                <span className="jwala-insight-text">Verify</span>
              </div>
              <span style={{ fontSize: "0.68rem", fontWeight: "600" }}>Spent $1,390 of $1,600</span>
            </div>
          </div>
        </div>

        {/* CARD 5: Health Records Table & Search */}
        <div className="jwala-card">
          <div className="jwala-card-header">
            <div className="jwala-card-title">Health Transactions</div>
            <button 
              className="jwala-topbar-pill" 
              style={{ padding: "4px 10px", fontSize: "0.7rem", borderRadius: "6px" }}
              onClick={() => setShowAddForm(!showAddForm)}
            >
              {showAddForm ? "Cancel" : "Add Record"}
            </button>
          </div>

          {showAddForm && (
            <form onSubmit={handleAddRecord} style={{ marginBottom: "16px", padding: "12px", background: "rgba(255,255,255,0.02)", borderRadius: "8px", border: "1px solid var(--jwala-border)" }}>
              <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                <input 
                  type="text" 
                  placeholder="Category (e.g. Dental)" 
                  className="jwala-table-search"
                  style={{ paddingLeft: "10px", marginBottom: "0" }}
                  value={newCategory}
                  onChange={(e) => setNewCategory(e.target.value)}
                  required
                />
                <div style={{ display: "flex", gap: "8px" }}>
                  <input 
                    type="number" 
                    placeholder="Amount ($)" 
                    className="jwala-table-search"
                    style={{ paddingLeft: "10px", marginBottom: "0", flex: 1 }}
                    value={newValue}
                    onChange={(e) => setNewValue(e.target.value)}
                    required
                  />
                  <select 
                    className="jwala-chart-select" 
                    style={{ flex: 1 }}
                    value={newStatus}
                    onChange={(e) => setNewStatus(e.target.value as RecordItem["status"])}
                  >
                    <option value="Paid">Paid</option>
                    <option value="Claimed">Claimed</option>
                    <option value="Pending">Pending</option>
                  </select>
                </div>
                <button 
                  type="submit" 
                  className="jwala-upgrade-btn" 
                  style={{ width: "100%", padding: "6px" }}
                >
                  Insert Transaction
                </button>
              </div>
            </form>
          )}

          <div className="jwala-table-search-wrapper">
            <Search className="jwala-table-search-icon" />
            <input
              type="text"
              placeholder="Search health metrics..."
              className="jwala-table-search"
              value={tableSearch}
              onChange={(e) => setTableSearch(e.target.value)}
            />
          </div>

          <div className="jwala-filter-tabs">
            <button 
              className={`jwala-filter-tab ${filterTab === "All" ? "active" : ""}`}
              onClick={() => setFilterTab("All")}
            >
              All Records
            </button>
            <button 
              className={`jwala-filter-tab ${filterTab === "Pending" ? "active" : ""}`}
              onClick={() => setFilterTab("Pending")}
            >
              Pending
            </button>
            <button 
              className={`jwala-filter-tab ${filterTab === "Completed" ? "active" : ""}`}
              onClick={() => setFilterTab("Completed")}
            >
              Completed
            </button>
          </div>

          <div className="jwala-table-container">
            <table className="jwala-table">
              <thead>
                <tr>
                  <th>Category</th>
                  <th>Value</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {filteredRecords.length > 0 ? (
                  filteredRecords.map((record) => (
                    <tr key={record.id}>
                      <td>{record.category}</td>
                      <td style={{ fontFamily: "JetBrains Mono, monospace" }}>${record.value.toFixed(2)}</td>
                      <td>
                        <span 
                          className={`jwala-status-pill ${record.status.toLowerCase()}`}
                          onClick={() => cycleStatus(record.id)}
                          title="Click to cycle status"
                        >
                          {record.status}
                        </span>
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={3} style={{ textAlign: "center", color: "var(--jwala-text-muted)" }}>
                      No transactions found
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* CARD 6: Interactive AI Assistant Chat */}
        <div className="jwala-card">
          <div className="jwala-card-header">
            <div className="jwala-card-title">
              <Bot size={16} style={{ color: "var(--jwala-accent)" }} />
              AI Assistant
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
                <button className="jwala-chat-chip" onClick={() => handleChipClick("Health Finance")}>Health Finance</button>
                <button className="jwala-chat-chip" onClick={() => handleChipClick("Insurance Tips")}>Insurance Tips</button>
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
                  placeholder="Ask or Search..." 
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
      case "overview":
        return renderOverviewDashboard();
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

            {/* HEALTH & FINANCE */}
            <div className="jwala-menu-group">
              <div className="jwala-menu-title">Health Finance</div>
              <div 
                className={`jwala-menu-item ${activeMenu === "overview" ? "active" : ""}`}
                onClick={() => navigate("/jwala/overview")}
              >
                <Activity size={18} />
                Overview
              </div>
              <div 
                className={`jwala-menu-item ${activeMenu === "health-wallet" ? "active" : ""}`}
                onClick={() => navigate("/jwala/health-wallet")}
              >
                <Wallet size={18} />
                Health Wallet
              </div>
              <div 
                className={`jwala-menu-item ${activeMenu === "transactions" ? "active" : ""}`}
                onClick={() => navigate("/jwala/transactions")}
              >
                <CreditCard size={18} />
                Transactions
              </div>
              <div 
                className={`jwala-menu-item ${activeMenu === "insurance" ? "active" : ""}`}
                onClick={() => navigate("/jwala/insurance")}
              >
                <Shield size={18} />
                Insurance
              </div>
              <div 
                className={`jwala-menu-item ${activeMenu === "payment" ? "active" : ""}`}
                onClick={() => navigate("/jwala/payment")}
              >
                <TrendingUp size={18} />
                Payment
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
            Get insights on coverage and eligibility with AI. Simplify decisions.
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
                placeholder="Search for any health metrics..." 
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
