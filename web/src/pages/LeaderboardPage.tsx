import { LeaderboardTable } from "../components/leaderboard/LeaderboardTable";

export function LeaderboardPage() {
  return (
    <div className="max-w-6xl mx-auto px-4 py-12">
      <h1 className="font-mono font-bold text-3xl mb-8">Leaderboard</h1>
      <LeaderboardTable />
    </div>
  );
}
