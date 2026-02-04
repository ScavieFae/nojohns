export const CHAIN_ID = 10143;
export const RPC_URL = "https://testnet-rpc.monad.xyz";
export const BLOCK_EXPLORER_URL = "https://testnet.monadexplorer.com";
export const ARENA_URL = import.meta.env.VITE_ARENA_URL ?? "http://localhost:8000";
export const USE_MOCK_DATA = import.meta.env.VITE_USE_MOCK_DATA === "true";

export const CONTRACTS = {
  matchProof: "0x1CC748475F1F666017771FB49131708446B9f3DF" as const,
  wager: "0x8d4D9FD03242410261b2F9C0e66fE2131AE0459d" as const,
};

export const ERC8004 = {
  identity: "0x8004A818BFB912233c491871b3d84c89A494BD9e" as const,
  reputation: "0x8004B663056A597Dffe9eCcC1965A193B7388713" as const,
};
