import { useQuery } from "@tanstack/react-query";
import { reputationRegistryAbi } from "../abi/reputationRegistry";
import { ERC8004, USE_MOCK_DATA } from "../config";
import { publicClient } from "../viem";
import { getBatchedLogs } from "../lib/getLogs";

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
  (e) => e.type === "event" && e.name === "FeedbackGiven",
)!;

/**
 * Fetch Elo updates from ReputationRegistry FeedbackGiven events.
 * Maps agentId → wallet address via IdentityRegistry.ownerOf().
 */
export function useEloEvents() {
  return useQuery({
    queryKey: ["eloEvents"],
    queryFn: async (): Promise<Map<`0x${string}`, number>> => {
      if (USE_MOCK_DATA) return new Map();

      const { logs } = await getBatchedLogs({
        address: ERC8004.reputation,
        event: feedbackGivenEvent,
        cacheKey: "eloFeedback",
      });

      // Keep only the latest Elo per agentId
      const eloByAgent = new Map<bigint, { elo: number; block: bigint }>();
      for (const log of logs) {
        if (log.args.tag1 !== "elo" || log.args.tag2 !== "melee") continue;

        const agentId = log.args.agentId!;
        const value = log.args.value!;
        const blockNumber = log.blockNumber;

        const existing = eloByAgent.get(agentId);
        if (!existing || blockNumber > existing.block) {
          eloByAgent.set(agentId, { elo: Number(value), block: blockNumber });
        }
      }

      // Map agentId → wallet address
      const eloByAddress = new Map<`0x${string}`, number>();
      for (const [agentId, { elo }] of eloByAgent) {
        try {
          const owner = await publicClient.readContract({
            address: ERC8004.identity,
            abi: identityRegistryAbi,
            functionName: "ownerOf",
            args: [agentId],
          });
          eloByAddress.set(owner, elo);
        } catch {
          // Agent might not exist or be burned
        }
      }

      return eloByAddress;
    },
    staleTime: 30_000,
    refetchInterval: 30_000,
  });
}
