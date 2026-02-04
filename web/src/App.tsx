import { Routes, Route } from "react-router-dom";
import { Layout } from "./components/layout/Layout";
import { LandingPage } from "./pages/LandingPage";
import { LeaderboardPage } from "./pages/LeaderboardPage";
import { MatchHistoryPage } from "./pages/MatchHistoryPage";
import { CompetePage } from "./pages/CompetePage";

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/leaderboard" element={<LeaderboardPage />} />
        <Route path="/matches" element={<MatchHistoryPage />} />
        <Route path="/compete" element={<CompetePage />} />
      </Routes>
    </Layout>
  );
}
