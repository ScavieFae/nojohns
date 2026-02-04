import { MatchList } from "../components/matches/MatchList";

export function MatchHistoryPage() {
  return (
    <div className="max-w-6xl mx-auto px-4 py-12">
      <h1 className="font-mono font-bold text-3xl mb-8">Match History</h1>
      <MatchList />
    </div>
  );
}
