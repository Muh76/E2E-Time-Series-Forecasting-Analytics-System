import type { ReactNode } from "react";
import { NavLink } from "react-router-dom";
import ModelInfoPanel from "./ModelInfoPanel";
import "./Layout.css";

interface LayoutProps {
  children: ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  return (
    <div className="layout">
      <nav className="navbar">
        <div className="navbar-brand">Forecasting</div>
        <div className="navbar-links">
          <NavLink to="/forecast" className={({ isActive }) => (isActive ? "active" : "")}>
            Forecast
          </NavLink>
          <NavLink to="/backtest" className={({ isActive }) => (isActive ? "active" : "")}>
            Backtest
          </NavLink>
        </div>
      </nav>

      <ModelInfoPanel />

      <main className="main-content">{children}</main>
    </div>
  );
}
