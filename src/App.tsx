import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import ForecastPage from "./pages/ForecastPage";

console.info(
  `[App] env=${import.meta.env.VITE_APP_ENV ?? import.meta.env.MODE}, ` +
  `api=${import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000"}`,
);

function BacktestPage() {
  return <div><h2>Backtest</h2><p>Rolling-origin backtest UI — coming soon.</p></div>;
}

export default function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/forecast" element={<ForecastPage />} />
          <Route path="/backtest" element={<BacktestPage />} />
          <Route path="*" element={<Navigate to="/forecast" replace />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}
