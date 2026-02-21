#!/usr/bin/env python3
"""
Generate N spectator wallets and save to wallets.json.

Usage:
    python scripts/generate_wallets.py 5          # Generate 5 wallets
    python scripts/generate_wallets.py 5 --append  # Add 5 more to existing file

Output: wallets.json (gitignored) with format:
    [{"address": "0x...", "private_key": "0x...", "label": "spectator-0"}, ...]
"""

import argparse
import json
import sys
from pathlib import Path

WALLETS_FILE = Path(__file__).parent.parent / "wallets.json"


def generate_wallets(count: int, append: bool = False) -> list[dict]:
    try:
        from eth_account import Account
    except ImportError:
        print("Error: eth-account not installed. Run: pip install -e '.[wallet]'")
        sys.exit(1)

    existing = []
    if append and WALLETS_FILE.exists():
        existing = json.loads(WALLETS_FILE.read_text())

    start_idx = len(existing)
    new_wallets = []

    for i in range(count):
        acct = Account.create()
        wallet = {
            "address": acct.address,
            "private_key": f"0x{acct.key.hex()}",
            "label": f"spectator-{start_idx + i}",
        }
        new_wallets.append(wallet)
        print(f"  {wallet['label']}: {wallet['address']}")

    all_wallets = existing + new_wallets
    WALLETS_FILE.write_text(json.dumps(all_wallets, indent=2))
    print(f"\nSaved {len(all_wallets)} wallets to {WALLETS_FILE}")

    return new_wallets


def main():
    parser = argparse.ArgumentParser(description="Generate spectator wallets")
    parser.add_argument("count", type=int, help="Number of wallets to generate")
    parser.add_argument("--append", action="store_true", help="Append to existing wallets.json")
    args = parser.parse_args()

    print(f"Generating {args.count} spectator wallets...")
    generate_wallets(args.count, args.append)


if __name__ == "__main__":
    main()
