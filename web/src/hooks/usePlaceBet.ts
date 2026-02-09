import { useState, useCallback } from "react";
import { parseEther } from "viem";
import { publicClient } from "../viem";
import { CONTRACTS } from "../config";
import { predictionPoolAbi } from "../abi/predictionPool";
import { getWalletClient } from "../lib/wallet";

/**
 * Hook to place a bet on a prediction pool.
 * Returns a function that sends the transaction and waits for confirmation.
 */
export function usePlaceBet() {
  const [isPending, setIsPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const placeBet = useCallback(
    async (poolId: bigint, betOnA: boolean, amountMon: string) => {
      setError(null);
      setIsPending(true);

      try {
        const walletClient = getWalletClient();
        if (!walletClient) throw new Error("No wallet connected");

        const [account] = await walletClient.getAddresses();
        if (!account) throw new Error("No account");

        const value = parseEther(amountMon);
        if (value <= 0n) throw new Error("Amount must be positive");

        const hash = await walletClient.writeContract({
          address: CONTRACTS.predictionPool as `0x${string}`,
          abi: predictionPoolAbi,
          functionName: "bet",
          args: [poolId, betOnA],
          value,
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

  return { placeBet, isPending, error };
}
