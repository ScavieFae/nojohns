export interface MatchRecord {
  matchId: `0x${string}`;
  winner: `0x${string}`;
  loser: `0x${string}`;
  gameId: string;
  winnerScore: number;
  loserScore: number;
  replayHash: `0x${string}`;
  timestamp: bigint;
  blockNumber?: bigint;
  transactionHash?: `0x${string}`;
}

export interface AgentStats {
  address: `0x${string}`;
  wins: number;
  losses: number;
  totalMatches: number;
  winRate: number;
  elo: number;
}

export interface WagerRecord {
  wagerId: bigint;
  proposer: `0x${string}`;
  opponent: `0x${string}`;
  gameId: string;
  amount: bigint;
  status: WagerStatus;
}

export enum WagerStatus {
  Open = 0,
  Accepted = 1,
  Settled = 2,
  Cancelled = 3,
  Voided = 4,
}

export interface ProtocolStats {
  totalMatches: number;
  totalWagered: bigint;
  uniqueAgents: number;
  predictionVolume: string;
}
