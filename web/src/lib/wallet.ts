import { createWalletClient, custom } from "viem";
import { monad } from "../viem";

/**
 * Minimal wallet connection via window.ethereum (MetaMask/injected).
 * No wagmi dependency — just viem + the browser provider.
 */

declare global {
  interface Window {
    ethereum?: {
      request: (args: { method: string; params?: unknown[] }) => Promise<unknown>;
      on: (event: string, handler: (...args: unknown[]) => void) => void;
      removeListener: (event: string, handler: (...args: unknown[]) => void) => void;
    };
  }
}

export function hasInjectedProvider(): boolean {
  return typeof window !== "undefined" && !!window.ethereum;
}

export async function connectWallet(): Promise<`0x${string}` | null> {
  if (!window.ethereum) return null;
  const accounts = (await window.ethereum.request({
    method: "eth_requestAccounts",
  })) as string[];
  return (accounts[0] as `0x${string}`) ?? null;
}

export async function getConnectedAccount(): Promise<`0x${string}` | null> {
  if (!window.ethereum) return null;
  const accounts = (await window.ethereum.request({
    method: "eth_accounts",
  })) as string[];
  return (accounts[0] as `0x${string}`) ?? null;
}

export function getWalletClient() {
  if (!window.ethereum) return null;
  return createWalletClient({
    chain: monad,
    transport: custom(window.ethereum),
  });
}
