import { useState, useEffect, useCallback } from "react";
import {
  hasInjectedProvider,
  connectWallet,
  getConnectedAccount,
} from "../lib/wallet";

/**
 * Minimal wallet connection hook. No wagmi — just window.ethereum.
 * Returns account address, connect function, and connection state.
 */
export function useWallet() {
  const [account, setAccount] = useState<`0x${string}` | null>(null);
  const [connecting, setConnecting] = useState(false);

  // Check for existing connection on mount
  useEffect(() => {
    getConnectedAccount().then(setAccount);

    if (!window.ethereum) return;
    const handleAccountsChanged = (...args: unknown[]) => {
      const accounts = args[0] as string[];
      setAccount((accounts[0] as `0x${string}`) ?? null);
    };
    window.ethereum.on("accountsChanged", handleAccountsChanged);
    return () => {
      window.ethereum?.removeListener("accountsChanged", handleAccountsChanged);
    };
  }, []);

  const connect = useCallback(async () => {
    if (!hasInjectedProvider()) {
      window.open("https://metamask.io", "_blank");
      return;
    }
    setConnecting(true);
    try {
      const addr = await connectWallet();
      setAccount(addr);
    } finally {
      setConnecting(false);
    }
  }, []);

  return { account, connect, connecting, hasProvider: hasInjectedProvider() };
}
