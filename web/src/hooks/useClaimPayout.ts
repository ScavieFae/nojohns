import { useState, useCallback } from "react";
import { publicClient } from "../viem";
import { CONTRACTS } from "../config";
import { predictionPoolAbi } from "../abi/predictionPool";
import { getWalletClient } from "../lib/wallet";
import { PoolStatus } from "./usePredictionPool";

/**
 * Hook to claim payout or refund from a prediction pool.
 * Calls `claim` for resolved pools, `claimRefund` for cancelled pools.
 */
export function useClaimPayout() {
  const [isPending, setIsPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const claimPayout = useCallback(
    async (poolId: bigint, poolStatus: PoolStatus) => {
      setError(null);
      setIsPending(true);

      try {
        const walletClient = getWalletClient();
        if (!walletClient) throw new Error("No wallet connected");

        const [account] = await walletClient.getAddresses();
        if (!account) throw new Error("No account");

        const functionName =
          poolStatus === PoolStatus.Cancelled ? "claimRefund" : "claim";

        const hash = await walletClient.writeContract({
          address: CONTRACTS.predictionPool as `0x${string}`,
          abi: predictionPoolAbi,
          functionName,
          args: [poolId],
          account,
        });

        await publicClient.waitForTransactionReceipt({ hash });
        return hash;
      } catch (e) {
        const msg = e instanceof Error ? e.message : "Transaction failed";
        setError(msg);
        throw e;
      } finally {
        setIsPending(false);
      }
    },
    []
  );

  return { claimPayout, isPending, error };
}
