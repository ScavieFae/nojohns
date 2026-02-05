import { Routes, Route } from "react-router-dom";
import { Layout } from "./components/layout/Layout";
import { LandingPage } from "./pages/LandingPage";
import { LeaderboardPage } from "./pages/LeaderboardPage";
import { MatchHistoryPage } from "./pages/MatchHistoryPage";
import { CompetePage } from "./pages/CompetePage";
import { LivePage } from "./pages/LivePage";
import { ViewerDemo } from "./components/viewer/ViewerDemo";

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/leaderboard" element={<LeaderboardPage />} />
        <Route path="/matches" element={<MatchHistoryPage />} />
        <Route path="/compete" element={<CompetePage />} />
        <Route path="/live" element={<LivePage />} />
        <Route path="/live/:matchId" element={<LivePage />} />
        <Route path="/viewer" element={<ViewerDemo />} />
      </Routes>
    </Layout>
  );
}
