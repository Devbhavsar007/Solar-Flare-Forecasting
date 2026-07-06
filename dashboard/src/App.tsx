import { lazy, Suspense } from "react";
import { Routes, Route } from "react-router-dom";
import NavBar from "./components/NavBar";
import { LiveDataProvider } from "./context/LiveDataContext";
import "./index.css";

/**
 * Lazy-load every page route to resolve the 650kB chunk warning.
 * LiveDataProvider (WebSocket) wraps ABOVE the lazy boundary so
 * switching routes never tears down / reopens the socket.
 */
const Home = lazy(() => import("./pages/Home"));
const Dashboard = lazy(() => import("./pages/Dashboard"));
const Observatory = lazy(() => import("./pages/Observatory"));
const EarthImpact = lazy(() => import("./pages/EarthImpact"));
const Analytics = lazy(() => import("./pages/Analytics"));
const Explainability = lazy(() => import("./pages/Explainability"));
const Architecture = lazy(() => import("./pages/Architecture"));
const About = lazy(() => import("./pages/About"));

export default function App() {
  return (
    <LiveDataProvider>
      <div className="app-shell">
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
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/observatory" element={<Observatory />} />
            <Route path="/earth-impact" element={<EarthImpact />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/explainability" element={<Explainability />} />
            <Route path="/architecture" element={<Architecture />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </Suspense>
      </div>
    </LiveDataProvider>
  );
}