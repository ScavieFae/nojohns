import { Routes, Route, useLocation } from "react-router-dom";
import { Layout } from "./components/layout/Layout";
import { LandingPage } from "./pages/LandingPage";
import { LeaderboardPage } from "./pages/LeaderboardPage";
import { MatchHistoryPage } from "./pages/MatchHistoryPage";
import { CompetePage } from "./pages/CompetePage";
import { LivePage } from "./pages/LivePage";
import { ViewerDemo } from "./components/viewer/ViewerDemo";
import { DemoPage } from "./pages/DemoPage";
import { BetPage } from "./pages/BetPage";
import { TournamentPage } from "./pages/TournamentPage";

export default function App() {
  const { pathname } = useLocation();

  // Tournament routes render without the site chrome (header/footer/nav)
  if (pathname.startsWith("/tournament/")) {
    return (
      <Routes>
        <Route path="/tournament/play" element={<TournamentPage />} />
        <Route path="/tournament/bet" element={<BetPage />} />
      </Routes>
    );
  }

  return (
    <Layout>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/leaderboard" element={<LeaderboardPage />} />
        <Route path="/matches" element={<MatchHistoryPage />} />
        <Route path="/compete" element={<CompetePage />} />
        <Route path="/live" element={<LivePage />} />
        <Route path="/live/:matchId" element={<LivePage />} />
        <Route path="/demo" element={<DemoPage />} />
        <Route path="/viewer" element={<ViewerDemo />} />
        <Route path="/bet" element={<BetPage />} />
      </Routes>
    </Layout>
  );
}
