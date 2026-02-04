interface ScoreDisplayProps {
  winnerScore: number;
  loserScore: number;
}

export function ScoreDisplay({ winnerScore, loserScore }: ScoreDisplayProps) {
  return (
    <span className="font-mono font-bold text-sm">
      <span className="text-accent-green">{winnerScore}</span>
      <span className="text-gray-500"> - </span>
      <span className="text-accent-red">{loserScore}</span>
    </span>
  );
}
