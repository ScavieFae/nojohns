#!/usr/bin/env python3
"""
Check the status of all spectator wallets — balances, arena health, and pool activity.

Usage:
    python scripts/fleet_status.py
    python scripts/fleet_status.py --rpc https://rpc.monad.xyz
"""

import argparse
import json
import sys
from pathlib import Path

WALLETS_FILE = Path(__file__).parent.parent / "wallets.json"
DEFAULT_ARENA = "https://nojohns-arena-production.up.railway.app"
DEFAULT_RPC = "https://rpc.monad.xyz"
DEFAULT_POOL = "0x33E65E300575D11a42a579B2675A63cb4374598D"

LOW_BALANCE_THRESHOLD = 0.01  # MON


def main():
    parser = argparse.ArgumentParser(description="Check fleet wallet status")
    parser.add_argument("--rpc", default=DEFAULT_RPC, help="RPC URL")
    parser.add_argument("--pool", default=DEFAULT_POOL, help="PredictionPool address")
    parser.add_argument("--arena", default=DEFAULT_ARENA, help="Arena URL")
    args = parser.parse_args()

    try:
        from rich.console import Console
        from rich.table import Table
        from rich.text import Text
    except ImportError:
        print("Missing dependency: rich. Install: pip install -e '.[spectator]'")
        sys.exit(1)

    if not WALLETS_FILE.exists():
        print(f"Error: {WALLETS_FILE} not found.")
        sys.exit(1)

    try:
        from web3 import Web3
    except ImportError:
        print("Error: web3 not installed. Run: pip install -e '.[wallet]'")
        sys.exit(1)

    console = Console()
    wallets = json.loads(WALLETS_FILE.read_text())
    w3 = Web3(Web3.HTTPProvider(args.rpc))

    # --- Arena health ---
    console.print()
    try:
        import httpx
        resp = httpx.get(f"{args.arena}/health", timeout=5)
        health = resp.json()
        active = health.get("active_matches", 0)
        queue = health.get("queue_size", 0)
        live_ids = health.get("live_match_ids", [])
        style = "green" if active > 0 else "dim"
        console.print(f"[bold]Arena:[/bold] [{style}]{active} active match(es)[/{style}], {queue} queued")
        if live_ids:
            console.print(f"  Live: {', '.join(str(m) for m in live_ids)}")
    except Exception as e:
        console.print(f"[bold]Arena:[/bold] [red]unreachable[/red] ({e})")
    console.print()

    # --- Wallet balances ---
    table = Table(title="Spectator Wallets", show_header=True, header_style="bold cyan")
    table.add_column("Label", style="bold", min_width=14)
    table.add_column("Address", min_width=14)
    table.add_column("Balance (MON)", justify="right", min_width=14)

    total_balance = 0.0

    for wallet in wallets:
        addr = wallet["address"]
        try:
            balance_wei = w3.eth.get_balance(addr)
            balance_mon = balance_wei / 10**18
        except Exception:
            balance_mon = -1.0

        total_balance += max(balance_mon, 0)

        if balance_mon < 0:
            bal_text = Text("error", style="red")
        elif balance_mon < LOW_BALANCE_THRESHOLD:
            bal_text = Text(f"{balance_mon:.4f}", style="bold red")
        else:
            bal_text = Text(f"{balance_mon:.4f}")

        table.add_row(wallet["label"], f"{addr[:12]}...", bal_text)

    table.add_section()
    table.add_row("Total", "", Text(f"{total_balance:.4f}", style="bold"))

    console.print(table)

    # --- Pool count ---
    try:
        from nojohns.contract import get_prediction_pool_contract
        _, contract = get_prediction_pool_contract(args.rpc, args.pool)
        pool_count = contract.functions.poolCount().call()
        console.print(f"\n[bold]Prediction pools on-chain:[/bold] {pool_count}")
    except Exception:
        pass

    console.print()


if __name__ == "__main__":
    main()
