import { createPublicClient, http, defineChain } from "viem";
import { CHAIN_ID, RPC_URL } from "./config";

export const monadTestnet = defineChain({
  id: CHAIN_ID,
  name: "Monad Testnet",
  nativeCurrency: { name: "MON", symbol: "MON", decimals: 18 },
  rpcUrls: {
    default: { http: [RPC_URL] },
  },
  blockExplorers: {
    default: { name: "Monad Explorer", url: "https://testnet.monadexplorer.com" },
  },
});

export const publicClient = createPublicClient({
  chain: monadTestnet,
  transport: http(),
});
