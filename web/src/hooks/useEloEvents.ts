import { useQuery } from "@tanstack/react-query";
import { createPublicClient, http } from "viem";
import { reputationRegistryAbi } from "../abi/reputationRegistry";
import { ERC8004, RPC_URL, USE_MOCK_DATA, DEPLOY_BLOCK, MAX_LOG_RANGE } from "../config";

const client = createPublicClient({
  transport: http(RPC_URL),
});

// Minimal IdentityRegistry ABI for ownerOf
const identityRegistryAbi = [
  {
    type: "function",
    name: "ownerOf",
    inputs: [{ name: "tokenId", type: "uint256" }],
    outputs: [{ name: "", type: "address" }],
    stateMutability: "view",
  },
] as const;

const feedbackGivenEvent = reputationRegistryAbi.find(
  (e) => e.type === "event" && e.name === "FeedbackGiven"
)!;

/**
 * Fetch Elo updates from ReputationRegistry FeedbackGiven events
 * Maps agentId → wallet address via IdentityRegistry.ownerOf()
 */
export function useEloEvents() {
  return useQuery({
    queryKey: ["eloEvents"],
    queryFn: async (): Promise<Map<`0x${string}`, number>> => {
      if (USE_MOCK_DATA) return new Map();

      const currentBlock = await client.getBlockNumber();
      const eloByAgent = new Map<bigint, { elo: number; block: bigint }>();

      // Scan in batches (Monad has small getLogs limit)
      let fromBlock = DEPLOY_BLOCK;
      while (fromBlock <= currentBlock) {
        const toBlock =
          fromBlock + MAX_LOG_RANGE - 1n > currentBlock
            ? currentBlock
            : fromBlock + MAX_LOG_RANGE - 1n;

        try {
          const logs = await client.getLogs({
            address: ERC8004.reputation,
            event: feedbackGivenEvent,
            fromBlock,
            toBlock,
          });

          for (const log of logs) {
            const tag1 = log.args.tag1;
            const tag2 = log.args.tag2;

            // Only process elo/melee feedback
            if (tag1 !== "elo" || tag2 !== "melee") continue;

            const agentId = log.args.agentId!;
            const value = log.args.value!;
            const blockNumber = log.blockNumber;

            // Keep only the latest value per agent
            const existing = eloByAgent.get(agentId);
            if (!existing || blockNumber > existing.block) {
              eloByAgent.set(agentId, {
                elo: Number(value),
                block: blockNumber,
              });
            }
          }
        } catch {
          // Skip failed batches
        }

        fromBlock = toBlock + 1n;
      }

      // Map agentId → wallet address
      const eloByAddress = new Map<`0x${string}`, number>();
      const agentIds = Array.from(eloByAgent.keys());

      // Batch ownerOf calls
      for (const agentId of agentIds) {
        try {
          const owner = await client.readContract({
            address: ERC8004.identity,
            abi: identityRegistryAbi,
            functionName: "ownerOf",
            args: [agentId],
          });

          const eloData = eloByAgent.get(agentId)!;
          eloByAddress.set(owner, eloData.elo);
        } catch {
          // Agent might not exist or be burned
        }
      }

      return eloByAddress;
    },
    staleTime: 30_000, // Refetch every 30s
  });
}
