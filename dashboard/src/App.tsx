import { lazy, Suspense } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import NavBar from "./components/NavBar";
import { LiveDataProvider } from "./context/LiveDataContext";
import { SparklesCore } from "./components/SparklesCore";
import "./index.css";

const Home = lazy(() => import("./pages/Home"));
const JwalaDashboard = lazy(() => import("./pages/JwalaDashboard"));

export default function App() {
  return (
    <LiveDataProvider>
      <div className="app-shell">
        <SparklesCore
          id="tsparticles"
          background="transparent"
          minSize={0.6}
          maxSize={1.4}
          speed={3.5}
          particleColor="#F6D337"
          particleDensity={80}
          className="sparkles-bg"
        />
        <NavBar />
        <Suspense
          fallback={
            <div className="page" style={{ textAlign: "center", paddingTop: "20vh", opacity: 0.5 }}>
              Loading…
            </div>
          }
        >
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/jwala/:tab?" element={<JwalaDashboard />} />
            
            {/* Redirects for original standalone pages into nested JWALA Dashboard sidebar panels */}
            <Route path="/dashboard" element={<Navigate to="/jwala/dashboard" replace />} />
            <Route path="/observatory" element={<Navigate to="/jwala/observatory" replace />} />
            <Route path="/earth-impact" element={<Navigate to="/jwala/earth-impact" replace />} />
            <Route path="/analytics" element={<Navigate to="/jwala/analytics" replace />} />
            <Route path="/explainability" element={<Navigate to="/jwala/explainability" replace />} />
            <Route path="/architecture" element={<Navigate to="/jwala/architecture" replace />} />
            <Route path="/about" element={<Navigate to="/jwala/about" replace />} />
            
            {/* Fallback to home */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Suspense>
      </div>
    </LiveDataProvider>
  );
}