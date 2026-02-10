export const CHAIN_ID = 10143;
export const RPC_URL = "https://testnet-rpc.monad.xyz";
export const BLOCK_EXPLORER_URL = "https://testnet.monadexplorer.com";
export const ARENA_URL = import.meta.env.VITE_ARENA_URL ?? "https://nojohns-arena-production.up.railway.app";
export const USE_MOCK_DATA = import.meta.env.VITE_USE_MOCK_DATA === "true";

export const CONTRACTS = {
  matchProof: "0x1CC748475F1F666017771FB49131708446B9f3DF" as const,
  wager: "0x8d4D9FD03242410261b2F9C0e66fE2131AE0459d" as const,
  predictionPool: "0x3455b3081B2af81443bEaa898F4Dd8F98BAc23b9" as const,
};

// Block where contracts were deployed — start scanning from here
export const DEPLOY_BLOCK = 10710000n;

// Monad testnet caps getLogs at 100 blocks per request
export const MAX_LOG_RANGE = 100n;

export const ERC8004 = {
  identity: "0x8004A818BFB912233c491871b3d84c89A494BD9e" as const,
  reputation: "0x8004B663056A597Dffe9eCcC1965A193B7388713" as const,
};
