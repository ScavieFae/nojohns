import { BLOCK_EXPLORER_URL } from "../config";

export function truncateAddress(address: string): string {
  return `${address.slice(0, 6)}...${address.slice(-4)}`;
}

export function explorerLink(type: "tx" | "address" | "block", value: string): string {
  return `${BLOCK_EXPLORER_URL}/${type}/${value}`;
}
