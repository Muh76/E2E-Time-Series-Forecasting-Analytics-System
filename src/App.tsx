import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import ErrorBoundary from "./components/ErrorBoundary";
import Layout from "./components/Layout";
import BacktestPage from "./pages/BacktestPage";
import ChatPage from "./pages/ChatPage";
import CopilotPage from "./pages/CopilotPage";
import EDAPage from "./pages/EDAPage";
import ForecastPage from "./pages/ForecastPage";
import MonitoringPage from "./pages/MonitoringPage";

console.info(
  `[App] env=${import.meta.env.VITE_APP_ENV ?? import.meta.env.MODE}, ` +
  `api=${import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000"}`,
);

export default function App() {
  return (
    // Root boundary: last resort catch — keeps a blank screen from showing
    // if BrowserRouter or Layout itself fails to render.
    <ErrorBoundary label="Application error">
      <BrowserRouter>
        <Layout>
          {/* Per-route boundaries: a crash on one page keeps the nav bar
              and other routes fully functional. */}
          <Routes>
            <Route
              path="/forecast"
              element={
                <ErrorBoundary label="Forecast page error">
                  <ForecastPage />
                </ErrorBoundary>
              }
            />
            <Route
              path="/backtest"
              element={
                <ErrorBoundary label="Backtest page error">
                  <BacktestPage />
                </ErrorBoundary>
              }
            />
            <Route
              path="/monitoring"
              element={
                <ErrorBoundary label="Monitoring page error">
                  <MonitoringPage />
                </ErrorBoundary>
              }
            />
            <Route
              path="/copilot"
              element={
                <ErrorBoundary label="Copilot page error">
                  <CopilotPage />
                </ErrorBoundary>
              }
            />
            <Route
              path="/eda"
              element={
                <ErrorBoundary label="EDA page error">
                  <EDAPage />
                </ErrorBoundary>
              }
            />
            <Route
              path="/chat"
              element={
                <ErrorBoundary label="Chat page error">
                  <ChatPage />
                </ErrorBoundary>
              }
            />
            <Route path="*" element={<Navigate to="/forecast" replace />} />
          </Routes>
        </Layout>
      </BrowserRouter>
    </ErrorBoundary>
  );
}
