import type { MatchRecord, AgentStats, WagerRecord } from "../types";
import { WagerStatus } from "../types";

const MOCK_AGENTS: `0x${string}`[] = [
  "0xA1b2C3d4E5f6a7B8c9D0e1F2a3B4c5D6e7F8a9B0",
  "0xB2c3D4e5F6a7B8c9D0e1F2a3B4c5D6e7F8a9B0A1",
  "0xC3d4E5f6A7b8C9d0E1f2A3b4C5d6E7f8A9b0A1B2",
  "0xD4e5F6a7B8c9D0e1F2a3B4c5D6e7F8a9B0a1B2C3",
  "0xE5f6A7b8C9d0E1f2A3b4C5d6E7f8A9b0A1b2C3D4",
  "0xF6a7B8c9D0e1F2a3B4c5D6e7F8a9B0a1B2c3D4E5",
];

const AGENT_NAMES: Record<string, string> = {
  "0xA1b2C3d4E5f6a7B8c9D0e1F2a3B4c5D6e7F8a9B0": "PhillipBot",
  "0xB2c3D4e5F6a7B8c9D0e1F2a3B4c5D6e7F8a9B0A1": "SmashBot",
  "0xC3d4E5f6A7b8C9d0E1f2A3b4C5d6E7f8A9b0A1B2": "NeuralFox",
  "0xD4e5F6a7B8c9D0e1F2a3B4c5D6e7F8a9B0a1B2C3": "WaveShine",
  "0xE5f6A7b8C9d0E1f2A3b4C5d6E7f8A9b0A1b2C3D4": "TechChaser",
  "0xF6a7B8c9D0e1F2a3B4c5D6e7F8a9B0a1B2c3D4E5": "EdgeGuard",
};

export function getAgentName(address: string): string {
  return AGENT_NAMES[address] ?? null;
}

function randomBytes32(): `0x${string}` {
  const bytes = Array.from({ length: 32 }, () =>
    Math.floor(Math.random() * 256).toString(16).padStart(2, "0"),
  ).join("");
  return `0x${bytes}`;
}

function randomMatch(index: number): MatchRecord {
  const winnerIdx = index % MOCK_AGENTS.length;
  let loserIdx = (index * 3 + 1) % MOCK_AGENTS.length;
  if (loserIdx === winnerIdx) loserIdx = (loserIdx + 1) % MOCK_AGENTS.length;

  const winnerScore = 3;
  const loserScore = Math.floor(Math.random() * 3);
  const hoursAgo = index * 2 + Math.floor(Math.random() * 3);

  return {
    matchId: randomBytes32(),
    winner: MOCK_AGENTS[winnerIdx],
    loser: MOCK_AGENTS[loserIdx],
    gameId: "melee",
    winnerScore,
    loserScore,
    replayHash: randomBytes32(),
    timestamp: BigInt(Math.floor(Date.now() / 1000) - hoursAgo * 3600),
    blockNumber: BigInt(1000000 + index * 100),
    transactionHash: randomBytes32(),
  };
}

export const MOCK_MATCHES: MatchRecord[] = Array.from({ length: 24 }, (_, i) =>
  randomMatch(i),
);

export const MOCK_LEADERBOARD: AgentStats[] = MOCK_AGENTS.map((address) => {
  const wins = MOCK_MATCHES.filter((m) => m.winner === address).length;
  const losses = MOCK_MATCHES.filter((m) => m.loser === address).length;
  const totalMatches = wins + losses;
  return {
    address,
    wins,
    losses,
    totalMatches,
    winRate: totalMatches > 0 ? wins / totalMatches : 0,
  };
}).sort((a, b) => b.wins - a.wins || a.losses - b.losses);

export const MOCK_WAGERS: WagerRecord[] = [
  {
    wagerId: 0n,
    proposer: MOCK_AGENTS[0],
    opponent: MOCK_AGENTS[1],
    gameId: "melee",
    amount: 500000000000000000n, // 0.5 MON
    status: WagerStatus.Settled,
  },
  {
    wagerId: 1n,
    proposer: MOCK_AGENTS[2],
    opponent: MOCK_AGENTS[3],
    gameId: "melee",
    amount: 1000000000000000000n, // 1 MON
    status: WagerStatus.Settled,
  },
  {
    wagerId: 2n,
    proposer: MOCK_AGENTS[4],
    opponent: "0x0000000000000000000000000000000000000000",
    gameId: "melee",
    amount: 250000000000000000n, // 0.25 MON
    status: WagerStatus.Open,
  },
];
