import { Routes, Route } from "react-router-dom";
import NavBar from "./components/NavBar";
import { LiveDataProvider } from "./context/LiveDataContext";
import Home from "./pages/Home";
import Dashboard from "./pages/Dashboard";
import Observatory from "./pages/Observatory";
import EarthImpact from "./pages/EarthImpact";
import Analytics from "./pages/Analytics";
import Explainability from "./pages/Explainability";
import Architecture from "./pages/Architecture";
import About from "./pages/About";
import "./index.css";

export default function App() {
  return (
    <LiveDataProvider>
      <div className="app-shell">
        <NavBar />
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
      </div>
    </LiveDataProvider>
  );
}