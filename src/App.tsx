import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import ErrorBoundary from "./components/ErrorBoundary";
import Layout from "./components/Layout";
import BacktestPage from "./pages/BacktestPage";
import ForecastPage from "./pages/ForecastPage";

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
          <Routes>
            <Route path="/forecast" element={<ForecastPage />} />
            <Route path="/backtest" element={<BacktestPage />} />
            <Route path="*" element={<Navigate to="/forecast" replace />} />
          </Routes>
        </Layout>
      </BrowserRouter>
    </ErrorBoundary>
  );
}
