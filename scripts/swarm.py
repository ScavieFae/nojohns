#!/usr/bin/env python3
"""
Spectator Swarm — single-process asyncio fleet with live Rich dashboard.

Launches multiple SpectatorAgents in one event loop. Each agent discovers
prediction pools, watches live Melee matches via WebSocket, evaluates win
probability from frame data, and bets using Kelly criterion when it finds edge.

Usage:
    python scripts/swarm.py                   # All wallets (max 6), mixed risk
    python scripts/swarm.py --count 3         # First 3 wallets
    python scripts/swarm.py --risk aggressive # All aggressive
    python scripts/swarm.py --no-dashboard    # Log-only mode (headless/CI)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

WALLETS_FILE = Path(__file__).parent.parent / "wallets.json"
DEFAULT_ARENA = "https://nojohns-arena-production.up.railway.app"
DEFAULT_RPC = "https://rpc.monad.xyz"
DEFAULT_POOL = "0x33E65E300575D11a42a579B2675A63cb4374598D"
MAX_AGENTS = 6

RISK_PROFILES = ["conservative", "moderate", "aggressive"]
RISK_PARAMS: dict[str, tuple[float, float]] = {
    "conservative": (0.25, 0.03),
    "moderate": (0.5, 0.05),
    "aggressive": (1.0, 0.10),
}

MAX_RESTARTS = 5
RESTART_BACKOFF = 10.0  # seconds between crash restarts
STAGGER_DELAY = 1.5     # seconds between agent launches
DASHBOARD_REFRESH = 1.0  # seconds between dashboard updates
EVENT_LOG_SIZE = 12      # visible events in feed

logger = logging.getLogger("swarm")


# ---------------------------------------------------------------------------
# Event system — live activity feed
# ---------------------------------------------------------------------------

@dataclass
class Event:
    timestamp: float
    label: str       # agent label or "" for system events
    message: str
    style: str = ""  # Rich style for the message


class EventLog:
    """Thread-safe-ish event log with bounded size."""

    def __init__(self, maxlen: int = EVENT_LOG_SIZE):
        self._events: deque[Event] = deque(maxlen=maxlen)

    def add(self, label: str, message: str, style: str = ""):
        self._events.append(Event(
            timestamp=time.time(),
            label=label,
            message=message,
            style=style,
        ))

    @property
    def events(self) -> list[Event]:
        return list(self._events)


# ---------------------------------------------------------------------------
# Agent handle — wraps a SpectatorAgent with supervisor metadata
# ---------------------------------------------------------------------------

@dataclass
class AgentHandle:
    label: str
    risk: str
    wallet_address: str
    private_key: str
    arena_url: str
    rpc_url: str
    pool_address: str
    multiplier: float
    max_pct: float
    agent: object | None = None          # SpectatorAgent instance
    task: asyncio.Task | None = None
    status: str = "pending"
    restarts: int = 0
    total_bets: int = 0
    total_wagered_wei: int = 0
    last_bet_time: float | None = None
    error: str | None = None
    # Snapshot of bet count before last crash (so we don't lose history)
    _bets_before_crash: int = 0
    _wagered_before_crash: int = 0
    # Balance snapshot for P&L
    starting_balance_wei: int = 0
    # State tracking for event detection
    _prev_status: str = "pending"
    _prev_bet_count: int = 0
    _prev_pools: frozenset = field(default_factory=frozenset)


def create_agents(
    wallets: list[dict],
    arena_url: str,
    rpc_url: str,
    pool_address: str,
    risk: str,
) -> list[AgentHandle]:
    """Create AgentHandles from wallet list. Snapshots starting balances."""
    from web3 import Web3
    w3 = Web3(Web3.HTTPProvider(rpc_url))

    handles = []
    for i, wallet in enumerate(wallets):
        if risk == "mixed":
            r = RISK_PROFILES[i % len(RISK_PROFILES)]
        else:
            r = risk
        multiplier, max_pct = RISK_PARAMS[r]

        try:
            balance = w3.eth.get_balance(wallet["address"])
        except Exception:
            balance = 0

        handles.append(AgentHandle(
            label=wallet["label"],
            risk=r,
            wallet_address=wallet["address"],
            private_key=wallet["private_key"],
            arena_url=arena_url,
            rpc_url=rpc_url,
            pool_address=pool_address,
            multiplier=multiplier,
            max_pct=max_pct,
            starting_balance_wei=balance,
        ))
    return handles


def _make_agent(handle: AgentHandle):
    """Instantiate a ContinuousSpectatorAgent for this handle."""
    from eth_account import Account
    from agents.continuous_spectator import ContinuousSpectatorAgent

    account = Account.from_key(handle.private_key)
    return ContinuousSpectatorAgent(
        arena_url=handle.arena_url,
        account=account,
        rpc_url=handle.rpc_url,
        pool_address=handle.pool_address,
        multiplier=handle.multiplier,
        max_pct=handle.max_pct,
        poll_interval=15,
    )


# ---------------------------------------------------------------------------
# Supervisor — crash recovery per agent
# ---------------------------------------------------------------------------

async def run_agent_supervised(handle: AgentHandle, event_log: EventLog) -> None:
    """Run one agent with crash recovery. Recreates SpectatorAgent on failure."""
    while handle.restarts <= MAX_RESTARTS:
        try:
            handle.agent = _make_agent(handle)
            handle.status = "scanning"
            handle.error = None
            logger.info(f"[{handle.label}] Starting | {handle.risk} | {handle.wallet_address[:12]}...")

            await handle.agent.run()

            # Agent exited cleanly (stopped externally)
            _sync_bet_stats(handle)
            handle.status = "stopped"
            event_log.add(handle.label, "stopped", style="dim")
            logger.info(f"[{handle.label}] Stopped cleanly. {handle.total_bets} bet(s) total.")
            return

        except asyncio.CancelledError:
            # Shutdown signal — don't restart
            if handle.agent:
                handle.agent.stop()
            _sync_bet_stats(handle)
            handle.status = "stopped"
            return

        except Exception as exc:
            _sync_bet_stats(handle)
            # Preserve bet history across crash
            handle._bets_before_crash = handle.total_bets
            handle._wagered_before_crash = handle.total_wagered_wei
            handle.restarts += 1
            handle.error = f"{type(exc).__name__}: {exc}"
            handle.status = "CRASHED"
            short_err = str(exc)[:60]
            event_log.add(handle.label, f"crashed — {short_err}", style="bold red")
            logger.error(f"[{handle.label}] Crashed ({handle.restarts}/{MAX_RESTARTS}): {exc}")

            if handle.restarts > MAX_RESTARTS:
                event_log.add(handle.label, "max restarts exceeded, giving up", style="red")
                logger.error(f"[{handle.label}] Max restarts exceeded. Giving up.")
                return

            event_log.add(handle.label, f"restarting in {int(RESTART_BACKOFF)}s...", style="yellow")
            logger.info(f"[{handle.label}] Restarting in {RESTART_BACKOFF}s...")
            await asyncio.sleep(RESTART_BACKOFF)
            event_log.add(handle.label, "restarting now", style="yellow")


def _sync_bet_stats(handle: AgentHandle) -> None:
    """Pull latest bet stats from the live agent into the handle."""
    if handle.agent is None:
        return
    agent = handle.agent
    current_bets = len(getattr(agent, "bets", []))
    current_wagered = sum(b.amount_wei for b in getattr(agent, "bets", []))
    handle.total_bets = handle._bets_before_crash + current_bets
    handle.total_wagered_wei = handle._wagered_before_crash + current_wagered
    if getattr(agent, "bets", None):
        handle.last_bet_time = agent.bets[-1].timestamp


def _read_live_status(handle: AgentHandle) -> None:
    """Read agent state for dashboard display."""
    if handle.agent is None:
        return
    agent = handle.agent
    live_bets = len(getattr(agent, "bets", []))
    live_wagered = sum(b.amount_wei for b in getattr(agent, "bets", []))
    handle.total_bets = handle._bets_before_crash + live_bets
    handle.total_wagered_wei = handle._wagered_before_crash + live_wagered
    if getattr(agent, "bets", None):
        handle.last_bet_time = agent.bets[-1].timestamp

    # Infer status from agent state
    if not getattr(agent, "_running", False):
        if handle.status not in ("CRASHED", "stopped"):
            handle.status = "stopped"
    elif getattr(agent, "watched_pools", set()):
        recent_bet = (
            handle.last_bet_time
            and (time.time() - handle.last_bet_time) < 5
        )
        handle.status = "BETTING" if recent_bet else "watching"
    else:
        if handle.status not in ("CRASHED",):
            handle.status = "scanning"


def detect_events(handles: list[AgentHandle], event_log: EventLog) -> None:
    """Diff agent state against previous tick, emit events for changes."""
    for h in handles:
        _read_live_status(h)

        # Status transitions
        if h.status != h._prev_status:
            if h._prev_status == "pending" and h.status == "scanning":
                event_log.add(h.label, "scanning for pools", style="dim")
            elif h.status == "watching" and h._prev_status in ("scanning", "pending"):
                pools = getattr(h.agent, "watched_pools", set()) if h.agent else set()
                pool_str = f"Pool #{max(pools)}" if pools else "match"
                event_log.add(h.label, f"connected to {pool_str}", style="cyan")
            elif h.status == "BETTING":
                pass  # Handled by bet detection below — more detail there
            elif h.status == "CRASHED":
                pass  # Handled in supervisor
            elif h.status == "stopped" and h._prev_status not in ("pending",):
                pass  # Handled in supervisor

        # New pools discovered
        current_pools = frozenset(getattr(h.agent, "watched_pools", set())) if h.agent else frozenset()
        new_pools = current_pools - h._prev_pools
        for pool_id in sorted(new_pools):
            event_log.add(h.label, f"found Pool #{pool_id}", style="cyan")
        h._prev_pools = current_pools

        # New bets placed
        if h.total_bets > h._prev_bet_count:
            bets = getattr(h.agent, "bets", []) if h.agent else []
            new_count = h.total_bets - h._prev_bet_count
            for bet in bets[-new_count:]:
                amount_mon = bet.amount_wei / 10**18
                side = f"Player {bet.side}"
                tx_short = bet.tx_hash[:10] if bet.tx_hash else "?"
                event_log.add(
                    h.label,
                    f"bet {amount_mon:.4f} MON on {side}  Pool #{bet.pool_id}  tx:{tx_short}",
                    style="bold green",
                )

        # Update tracking state
        h._prev_status = h.status
        h._prev_bet_count = h.total_bets


# ---------------------------------------------------------------------------
# Dashboard — Rich live display (demo mode)
# ---------------------------------------------------------------------------

RISK_STYLES = {
    "conservative": "blue",
    "moderate": "yellow",
    "aggressive": "red",
}

STATUS_RENDER = {
    "scanning":  ("scanning...", "dim italic"),
    "watching":  ("watching", "cyan"),
    "BETTING":   ("BETTING", "bold green"),
    "CRASHED":   ("CRASHED", "bold red"),
    "stopped":   ("stopped", "dim"),
    "pending":   ("pending", "dim"),
}


def _agent_row(h: AgentHandle) -> "Text":
    """Build one styled agent status line."""
    from rich.text import Text

    line = Text()
    # Indicator
    if h.status == "BETTING":
        line.append("  ██ ", style="bold green")
    elif h.status == "CRASHED":
        line.append("  !! ", style="bold red")
    else:
        line.append("  ▸  ", style="dim")

    # Label
    line.append(f"{h.label:<14}", style="bold")

    # Risk
    line.append(f"{h.risk:<14}", style=RISK_STYLES.get(h.risk, "white"))

    # Wallet
    line.append(f"{h.wallet_address[:8]}..  ", style="dim")

    # Status
    label, style = STATUS_RENDER.get(h.status, (h.status, "dim"))
    line.append(f"{label:<12}", style=style)

    # Context: pool info + bet count (only when active)
    if h.status in ("watching", "BETTING"):
        pools = getattr(h.agent, "watched_pools", set()) if h.agent else set()
        if pools:
            line.append(f"Pool #{max(pools):<4}  ", style="cyan")
        if h.total_bets > 0:
            line.append(f"{h.total_bets} bet{'s' if h.total_bets != 1 else ''}", style="green")
    elif h.status == "CRASHED" and h.error:
        line.append(h.error[:40], style="dim red")

    return line


def _event_row(event: Event) -> "Text":
    """Build one styled event feed line."""
    from rich.text import Text

    ts = time.strftime("%H:%M:%S", time.localtime(event.timestamp))
    line = Text()
    line.append(f"  {ts}  ", style="dim")
    if event.label:
        line.append(f"{event.label:<14}", style="bold")
    else:
        line.append(f"{'':14}")
    line.append(event.message, style=event.style or "")
    return line


def build_dashboard(
    handles: list[AgentHandle],
    event_log: EventLog,
    start_time: float,
) -> "Panel":
    """Build the full dashboard panel: agents + event feed + stats."""
    from rich.console import Group
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.text import Text

    parts: list = []

    # --- Agent rows ---
    parts.append(Text())  # top padding
    for h in handles:
        parts.append(_agent_row(h))
    parts.append(Text())  # spacing

    # --- Event feed ---
    parts.append(Rule(title="LIVE", style="dim cyan", align="left"))
    events = event_log.events
    if events:
        for event in events:
            parts.append(_event_row(event))
    else:
        parts.append(Text("  waiting for activity...", style="dim italic"))
    parts.append(Text())  # bottom padding

    # --- Stats footer ---
    uptime = time.time() - start_time
    hours, remainder = divmod(int(uptime), 3600)
    minutes, seconds = divmod(remainder, 60)
    uptime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    active = sum(1 for h in handles if h.status in ("scanning", "watching", "BETTING"))
    total_bets = sum(h.total_bets for h in handles)
    total_wagered = sum(h.total_wagered_wei for h in handles) / 10**18

    footer = Text()
    footer.append(f"  {uptime_str}", style="bold")
    footer.append("  uptime", style="dim")
    footer.append("  ·  ", style="dim")
    footer.append(f"{active}/{len(handles)}", style="bold")
    footer.append("  active", style="dim")
    footer.append("  ·  ", style="dim")
    footer.append(f"{total_bets}", style="bold green" if total_bets > 0 else "bold")
    footer.append("  bets", style="dim")
    footer.append("  ·  ", style="dim")
    footer.append(f"{total_wagered:.4f} MON", style="bold green" if total_wagered > 0 else "bold")
    footer.append("  wagered", style="dim")

    parts.append(footer)

    return Panel(
        Group(*parts),
        title="[bold white]NO JOHNS[/bold white] [dim]·[/dim] [bold cyan]Spectator Swarm[/bold cyan]",
        subtitle="[dim]Ctrl+C to stop[/dim]",
        border_style="cyan",
        padding=(0, 1),
    )


def build_summary(handles: list[AgentHandle]) -> None:
    """Print post-shutdown summary table with P&L."""
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text
    from web3 import Web3

    console = Console()
    w3 = Web3(Web3.HTTPProvider(handles[0].rpc_url)) if handles else None

    table = Table(title="Swarm Summary", show_header=True, header_style="bold")
    table.add_column("Agent", style="bold")
    table.add_column("Risk")
    table.add_column("Bets", justify="right")
    table.add_column("Wagered (MON)", justify="right")
    table.add_column("P&L (MON)", justify="right")
    table.add_column("Final Status")

    total_pnl = 0.0

    for h in handles:
        _sync_bet_stats(h)
        wagered = h.total_wagered_wei / 10**18

        # P&L = current balance - starting balance
        try:
            current = w3.eth.get_balance(h.wallet_address) if w3 else 0
        except Exception:
            current = 0
        pnl_wei = current - h.starting_balance_wei
        pnl_mon = pnl_wei / 10**18
        total_pnl += pnl_mon

        if pnl_mon > 0:
            pnl_text = Text(f"+{pnl_mon:.4f}", style="bold green")
        elif pnl_mon < 0:
            pnl_text = Text(f"{pnl_mon:.4f}", style="bold red")
        else:
            pnl_text = Text("0.0000", style="dim")

        table.add_row(
            h.label,
            h.risk,
            str(h.total_bets),
            f"{wagered:.4f}",
            pnl_text,
            h.status,
        )

    total_bets = sum(h.total_bets for h in handles)
    total_wagered = sum(h.total_wagered_wei for h in handles) / 10**18

    if total_pnl > 0:
        total_pnl_text = Text(f"+{total_pnl:.4f}", style="bold green")
    elif total_pnl < 0:
        total_pnl_text = Text(f"{total_pnl:.4f}", style="bold red")
    else:
        total_pnl_text = Text("0.0000", style="dim")

    table.add_section()
    table.add_row("TOTAL", "", str(total_bets), f"{total_wagered:.4f}", total_pnl_text, "")

    console.print()
    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# Swarm launcher
# ---------------------------------------------------------------------------

async def launch_swarm(handles: list[AgentHandle], dashboard: bool) -> None:
    """Launch all agents with staggered startup, run dashboard or headless loop."""
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()
    event_log = EventLog(maxlen=EVENT_LOG_SIZE)

    def _signal_handler():
        logger.info("Shutdown signal received.")
        shutdown_event.set()
        event_log.add("", "shutdown signal — stopping all agents", style="bold yellow")
        for h in handles:
            if h.agent:
                h.agent.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    event_log.add("", f"fleet launching — {len(handles)} agents", style="bold cyan")

    # Staggered launch
    tasks = []
    for i, handle in enumerate(handles):
        if i > 0:
            await asyncio.sleep(STAGGER_DELAY)
        task = asyncio.create_task(
            run_agent_supervised(handle, event_log),
            name=handle.label,
        )
        handle.task = task
        tasks.append(task)
        event_log.add(handle.label, f"launched  {handle.risk}", style="dim")
        logger.info(f"Launched {handle.label} ({handle.risk})")

    start_time = time.time()

    if dashboard:
        await _run_dashboard_loop(handles, tasks, shutdown_event, start_time, event_log)
    else:
        await _run_headless_loop(tasks, shutdown_event)

    # Cancel any still-running tasks
    for task in tasks:
        if not task.done():
            task.cancel()

    await asyncio.gather(*tasks, return_exceptions=True)


async def _run_dashboard_loop(
    handles: list[AgentHandle],
    tasks: list[asyncio.Task],
    shutdown_event: asyncio.Event,
    start_time: float,
    event_log: EventLog,
) -> None:
    """Rich Live dashboard loop — updates every second until shutdown."""
    from rich.live import Live

    with Live(
        build_dashboard(handles, event_log, start_time),
        refresh_per_second=2,
        screen=False,
    ) as live:
        while not shutdown_event.is_set():
            detect_events(handles, event_log)
            live.update(build_dashboard(handles, event_log, start_time))
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=DASHBOARD_REFRESH)
            except asyncio.TimeoutError:
                pass
            if all(t.done() for t in tasks):
                break


async def _run_headless_loop(
    tasks: list[asyncio.Task],
    shutdown_event: asyncio.Event,
) -> None:
    """Headless mode — just wait for shutdown or all tasks to finish."""
    while not shutdown_event.is_set():
        if all(t.done() for t in tasks):
            break
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Launch the spectator swarm — autonomous prediction market betting fleet",
    )
    parser.add_argument(
        "--count", type=int, default=None,
        help=f"Number of agents to launch (default: all wallets, max {MAX_AGENTS})",
    )
    parser.add_argument(
        "--risk", default="mixed",
        choices=["conservative", "moderate", "aggressive", "mixed"],
        help="Risk profile (default: mixed)",
    )
    parser.add_argument(
        "--arena", default=None,
        help=f"Arena URL (default: {DEFAULT_ARENA})",
    )
    parser.add_argument(
        "--rpc", default=None,
        help=f"Monad RPC URL (default: {DEFAULT_RPC})",
    )
    parser.add_argument(
        "--pool", default=None,
        help=f"PredictionPool contract address (default: {DEFAULT_POOL})",
    )
    parser.add_argument(
        "--no-dashboard", action="store_true",
        help="Disable Rich dashboard (log-only mode for headless/CI)",
    )
    args = parser.parse_args()

    # Resolve config: CLI > env > defaults
    arena_url = args.arena or os.environ.get("ARENA_URL", DEFAULT_ARENA)
    rpc_url = args.rpc or os.environ.get("RPC_URL", DEFAULT_RPC)
    pool_address = args.pool or os.environ.get("PREDICTION_POOL", DEFAULT_POOL)

    # Validate deps early
    try:
        from agents.continuous_spectator import ContinuousSpectatorAgent  # noqa: F401
        from eth_account import Account  # noqa: F401
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install: pip install -e '.[wallet,spectator]'")
        sys.exit(1)

    try:
        from rich.console import Console  # noqa: F401
    except ImportError:
        print("Missing dependency: rich")
        print("Install: pip install -e '.[spectator]'")
        sys.exit(1)

    # Load wallets
    if not WALLETS_FILE.exists():
        print(f"Error: {WALLETS_FILE} not found. Run: python scripts/generate_wallets.py 5")
        sys.exit(1)

    wallets = json.loads(WALLETS_FILE.read_text())
    if args.count:
        wallets = wallets[:args.count]
    wallets = wallets[:MAX_AGENTS]

    if not wallets:
        print("No wallets available.")
        sys.exit(1)

    # Configure logging
    use_dashboard = not args.no_dashboard
    log_level = logging.WARNING if use_dashboard else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Create handles
    handles = create_agents(wallets, arena_url, rpc_url, pool_address, args.risk)

    if use_dashboard:
        from rich.console import Console
        console = Console()
        console.print()
        console.print("[bold white]NO JOHNS[/bold white] [dim]·[/dim] [bold cyan]Spectator Swarm[/bold cyan]")
        console.print(f"  [dim]Arena:[/dim]  {arena_url}")
        console.print(f"  [dim]Pool:[/dim]   {pool_address}")
        console.print(f"  [dim]Agents:[/dim] {len(handles)} ({args.risk} risk)")
        console.print()
    else:
        print(f"Launching {len(handles)} spectator agent(s)")
        print(f"  Arena: {arena_url}")
        print(f"  Pool:  {pool_address}")
        print(f"  Risk:  {args.risk}")
        print()

    # Run
    asyncio.run(launch_swarm(handles, use_dashboard))

    # Post-shutdown summary
    build_summary(handles)


if __name__ == "__main__":
    main()
