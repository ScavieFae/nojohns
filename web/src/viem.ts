import { createPublicClient, http, defineChain } from "viem";
import { CHAIN_ID, RPC_URL, BLOCK_EXPLORER_URL } from "./config";

export const monad = defineChain({
  id: CHAIN_ID,
  name: "Monad",
  nativeCurrency: { name: "MON", symbol: "MON", decimals: 18 },
  rpcUrls: {
    default: { http: [RPC_URL] },
  },
  blockExplorers: {
    default: { name: "Monad Explorer", url: BLOCK_EXPLORER_URL },
  },
});

export const publicClient = createPublicClient({
  chain: monad,
  transport: http(),
});
