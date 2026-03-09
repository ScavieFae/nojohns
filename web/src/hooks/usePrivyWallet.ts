import { useCallback } from "react";
import { usePrivy, useWallets } from "@privy-io/react-auth";
import { createWalletClient, custom } from "viem";
import { monad } from "../viem";

/**
 * Unified wallet hook: Privy embedded wallet + external injected wallet (MetaMask).
 *
 * For the /bet page and spectators:
 * - Email/Google login → Privy creates an embedded wallet automatically
 * - "Connect Wallet" → MetaMask/Rabby via Privy's wallet connect flow
 *
 * Returns:
 * - account: connected address (from Privy or injected provider)
 * - login: open Privy modal (handles email + external wallet)
 * - logout: disconnect all
 * - ready: Privy SDK loaded
 * - isAuthenticated: user is signed in via Privy
 * - getWalletClient: viem wallet client for the active wallet
 */
export function usePrivyWallet() {
  const { ready, authenticated, login, logout, user } = usePrivy();
  const { wallets } = useWallets();

  // Prefer embedded wallet, fall back to first external wallet
  const activeWallet = wallets.find((w) => w.walletClientType === "privy") ?? wallets[0];
  const account = (activeWallet?.address as `0x${string}`) ?? null;

  const getWalletClient = useCallback(async () => {
    if (!activeWallet) return null;

    // For Privy embedded wallets and external wallets connected through Privy,
    // switch to the correct chain and return a viem wallet client.
    await activeWallet.switchChain(monad.id);
    const provider = await activeWallet.getEthereumProvider();
    return createWalletClient({
      chain: monad,
      transport: custom(provider),
      account: activeWallet.address as `0x${string}`,
    });
  }, [activeWallet]);

  return {
    account,
    login,
    logout,
    ready,
    isAuthenticated: authenticated,
    user,
    wallets,
    getWalletClient,
    // Convenience flags
    hasWallet: !!account,
    isEmbeddedWallet: activeWallet?.walletClientType === "privy",
  };
}
