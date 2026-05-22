import type { ReactNode } from "react";
import { NavLink } from "react-router-dom";
import HealthDot from "./HealthDot";
import ModelInfoPanel from "./ModelInfoPanel";
import "./Layout.css";

interface LayoutProps {
  children: ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  return (
    <div className="layout">
      <nav className="navbar">
        <div className="navbar-brand">
          <HealthDot /> Forecasting
        </div>
        <div className="navbar-links">
          <NavLink to="/forecast" className={({ isActive }) => (isActive ? "active" : "")}>
            Forecast
          </NavLink>
          <NavLink to="/backtest" className={({ isActive }) => (isActive ? "active" : "")}>
            Backtest
          </NavLink>
          <NavLink to="/monitoring" className={({ isActive }) => (isActive ? "active" : "")}>
            Monitoring
          </NavLink>
          <NavLink to="/copilot" className={({ isActive }) => (isActive ? "active" : "")}>
            Copilot
          </NavLink>
          <NavLink to="/eda" className={({ isActive }) => (isActive ? "active" : "")}>
            EDA
          </NavLink>
          <NavLink to="/chat" className={({ isActive }) => (isActive ? "active" : "")}>
            Chat
          </NavLink>
        </div>
      </nav>

      <ModelInfoPanel />

      <main className="main-content">{children}</main>
    </div>
  );
}
