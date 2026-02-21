#!/usr/bin/env python3
"""
Fund spectator wallets from a source wallet.

Usage:
    python scripts/fund_wallets.py --amount 0.05 --source-key 0x...

    # Or set env var:
    export FUNDER_PRIVATE_KEY="0x..."
    python scripts/fund_wallets.py --amount 0.05

Distributes `amount` MON to each wallet in wallets.json that has less
than `amount` MON already (idempotent — safe to rerun).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

WALLETS_FILE = Path(__file__).parent.parent / "wallets.json"
DEFAULT_RPC = "https://rpc.monad.xyz"


def fund_wallets(source_key: str, amount_mon: float, rpc_url: str):
    try:
        from web3 import Web3
        from eth_account import Account
    except ImportError:
        print("Error: web3 and eth-account required. Run: pip install -e '.[wallet]'")
        sys.exit(1)

    if not WALLETS_FILE.exists():
        print(f"Error: {WALLETS_FILE} not found. Run generate_wallets.py first.")
        sys.exit(1)

    wallets = json.loads(WALLETS_FILE.read_text())
    if not wallets:
        print("No wallets to fund.")
        return

    w3 = Web3(Web3.HTTPProvider(rpc_url))
    funder = Account.from_key(source_key)
    amount_wei = int(amount_mon * 10**18)

    funder_balance = w3.eth.get_balance(funder.address)
    funder_mon = funder_balance / 10**18
    total_needed = amount_mon * len(wallets)

    print(f"Funder: {funder.address}")
    print(f"Balance: {funder_mon:.4f} MON")
    print(f"Funding {len(wallets)} wallets with {amount_mon} MON each ({total_needed:.4f} MON total)")
    print()

    if funder_balance < amount_wei * len(wallets):
        print(f"Warning: funder may not have enough MON for all wallets")

    funded = 0
    skipped = 0

    for wallet in wallets:
        addr = wallet["address"]
        current = w3.eth.get_balance(addr)
        current_mon = current / 10**18

        if current >= amount_wei:
            print(f"  {wallet['label']}: {current_mon:.4f} MON (already funded, skip)")
            skipped += 1
            continue

        top_up = amount_wei - current
        top_up_mon = top_up / 10**18

        try:
            nonce = w3.eth.get_transaction_count(funder.address)
            tx = {
                "to": Web3.to_checksum_address(addr),
                "value": top_up,
                "nonce": nonce,
                "chainId": w3.eth.chain_id,
                "gas": 21_000,
                "gasPrice": w3.eth.gas_price,
            }
            signed = w3.eth.account.sign_transaction(tx, funder.key)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=30)

            if receipt["status"] == 1:
                print(f"  {wallet['label']}: +{top_up_mon:.4f} MON -> {addr[:10]}... (tx: {tx_hash.hex()[:16]}...)")
                funded += 1
            else:
                print(f"  {wallet['label']}: REVERTED")

            # Small delay to avoid nonce issues
            time.sleep(0.5)

        except Exception as e:
            print(f"  {wallet['label']}: ERROR — {e}")

    print(f"\nDone. Funded: {funded}, Skipped: {skipped}, Total: {len(wallets)}")


def main():
    parser = argparse.ArgumentParser(description="Fund spectator wallets")
    parser.add_argument("--amount", type=float, default=0.05,
                        help="MON per wallet (default: 0.05)")
    parser.add_argument("--source-key", default=None,
                        help="Funder private key (or set FUNDER_PRIVATE_KEY env var)")
    parser.add_argument("--rpc", default=DEFAULT_RPC, help="RPC URL")
    args = parser.parse_args()

    source_key = args.source_key or os.environ.get("FUNDER_PRIVATE_KEY")
    if not source_key:
        print("Error: provide --source-key or set FUNDER_PRIVATE_KEY env var")
        sys.exit(1)

    fund_wallets(source_key, args.amount, args.rpc)


if __name__ == "__main__":
    main()
