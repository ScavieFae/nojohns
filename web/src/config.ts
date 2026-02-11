export const CHAIN_ID = 143;
export const RPC_URL = "https://rpc.monad.xyz";
export const BLOCK_EXPLORER_URL = "https://monadexplorer.com";
export const ARENA_URL = import.meta.env.VITE_ARENA_URL ?? "https://nojohns-arena-production.up.railway.app";
export const USE_MOCK_DATA = import.meta.env.VITE_USE_MOCK_DATA === "true";

export const CONTRACTS = {
  matchProof: "0x1CC748475F1F666017771FB49131708446B9f3DF" as const,
  wager: "0x8d4D9FD03242410261b2F9C0e66fE2131AE0459d" as const,
  predictionPool: "0x33E65E300575D11a42a579B2675A63cb4374598D" as const,
};

// Block where contracts were deployed — start scanning from here
export const DEPLOY_BLOCK = 54717354n;

// Monad caps getLogs at 100 blocks per request
export const MAX_LOG_RANGE = 100n;

export const ERC8004 = {
  identity: "0x8004A169FB4a3325136EB29fA0ceB6D2e539a432" as const,
  reputation: "0x8004BAa17C55a88189AE136b182e5fdA19dE9b63" as const,
};
