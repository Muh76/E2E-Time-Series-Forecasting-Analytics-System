import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";

function ForecastPage() {
  return <div><h2>Forecast</h2><p>Store-level forecast UI — coming soon.</p></div>;
}

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
