import { useState } from "react";
import { truncateAddress } from "../../lib/addresses";

interface BetFormProps {
  playerA: `0x${string}`;
  playerB: `0x${string}`;
  onBet: (betOnA: boolean, amount: string) => void;
  isPending: boolean;
  error: string | null;
}

export function BetForm({
  playerA,
  playerB,
  onBet,
  isPending,
  error,
}: BetFormProps) {
  const [amount, setAmount] = useState("");

  const handleBet = (betOnA: boolean) => {
    if (!amount || parseFloat(amount) <= 0) return;
    onBet(betOnA, amount);
    setAmount("");
  };

  return (
    <div className="space-y-3">
      {/* Amount input */}
      <div>
        <label className="text-xs text-gray-400 block mb-1">Amount (MON)</label>
        <input
          type="number"
          min="0"
          step="0.01"
          placeholder="0.01"
          value={amount}
          onChange={(e) => setAmount(e.target.value)}
          disabled={isPending}
          className="w-full bg-surface-700 border border-surface-600 rounded px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-accent-green"
        />
      </div>

      {/* Bet buttons */}
      <div className="grid grid-cols-2 gap-2">
        <button
          onClick={() => handleBet(true)}
          disabled={isPending || !amount || parseFloat(amount) <= 0}
          className="px-3 py-2 rounded text-xs font-bold bg-accent-green/20 text-accent-green border border-accent-green/30 hover:bg-accent-green/30 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {isPending ? "..." : truncateAddress(playerA)}
        </button>
        <button
          onClick={() => handleBet(false)}
          disabled={isPending || !amount || parseFloat(amount) <= 0}
          className="px-3 py-2 rounded text-xs font-bold bg-red-500/20 text-red-400 border border-red-500/30 hover:bg-red-500/30 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {isPending ? "..." : truncateAddress(playerB)}
        </button>
      </div>

      {/* Quick amounts */}
      <div className="flex gap-1">
        {["0.01", "0.05", "0.1", "0.5"].map((v) => (
          <button
            key={v}
            onClick={() => setAmount(v)}
            disabled={isPending}
            className="flex-1 px-1 py-1 rounded text-xs bg-surface-700 text-gray-400 hover:text-white hover:bg-surface-600 transition-colors"
          >
            {v}
          </button>
        ))}
      </div>

      {error && (
        <p className="text-red-400 text-xs break-words">{error}</p>
      )}
    </div>
  );
}
