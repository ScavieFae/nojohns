"""
agents/bankroll.py - Balance queries, Elo math, and Kelly criterion wager sizing.

Pure utility functions — any agent can import these.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BankrollState:
    """Snapshot of an agent's financial position."""

    balance_wei: int
    active_exposure_wei: int
    available_wei: int  # balance - exposure
    balance_mon: float
    available_mon: float


def get_mon_balance(address: str, rpc_url: str) -> int:
    """Get MON balance in wei."""
    try:
        from web3 import Web3
    except ImportError:
        logger.warning("web3 not installed")
        return 0

    w3 = Web3(Web3.HTTPProvider(rpc_url))
    return w3.eth.get_balance(address)


def get_active_wager_exposure(
    address: str, rpc_url: str, wager_contract: str
) -> int:
    """Sum of MON locked in open/accepted wagers (not yet settled)."""
    try:
        from nojohns.wallet import get_agent_wagers, get_wager_info
    except ImportError:
        logger.warning("wallet module not available")
        return 0

    try:
        wager_ids = get_agent_wagers(rpc_url, wager_contract, address)
    except Exception as e:
        logger.debug(f"Failed to get wagers: {e}")
        return 0

    total = 0
    for wid in wager_ids:
        try:
            info = get_wager_info(rpc_url, wager_contract, wid)
            # Only count Open (0) and Accepted (1) — not settled/cancelled/voided
            if info["status_code"] in (0, 1):
                total += info["amount"]
        except Exception:
            continue
    return total


def get_bankroll_state(
    address: str, rpc_url: str, wager_contract: str
) -> BankrollState:
    """Full financial snapshot: balance, exposure, available funds."""
    balance = get_mon_balance(address, rpc_url)
    exposure = get_active_wager_exposure(address, rpc_url, wager_contract)
    available = max(0, balance - exposure)

    return BankrollState(
        balance_wei=balance,
        active_exposure_wei=exposure,
        available_wei=available,
        balance_mon=balance / 10**18,
        available_mon=available / 10**18,
    )


# ============================================================================
# Elo math
# ============================================================================


def win_probability_from_elo(our_elo: int, opponent_elo: int) -> float:
    """Expected win probability from Elo ratings. Standard formula."""
    return 1 / (1 + 10 ** ((opponent_elo - our_elo) / 400))


# ============================================================================
# Kelly criterion
# ============================================================================


def kelly_fraction(win_prob: float) -> float:
    """Kelly criterion for even-money bets: f* = 2p - 1.

    Returns the fraction of bankroll to wager. Negative means no edge (don't bet).
    """
    return 2 * win_prob - 1


def kelly_wager(
    win_prob: float,
    bankroll_wei: int,
    multiplier: float = 1.0,
    max_pct: float = 0.10,
) -> int:
    """Calculate wager size using Kelly criterion.

    Args:
        win_prob: Estimated probability of winning (0-1).
        bankroll_wei: Available bankroll in wei.
        multiplier: Kelly multiplier (0.5 = half-Kelly, conservative).
        max_pct: Maximum fraction of bankroll to wager (cap).

    Returns:
        Wager amount in wei. 0 if no edge.
    """
    f = kelly_fraction(win_prob)
    if f <= 0:
        return 0

    # Apply multiplier and cap
    fraction = min(f * multiplier, max_pct)
    amount = int(bankroll_wei * fraction)

    # Floor: 0.001 MON (below this, gas costs exceed the wager)
    min_wager = int(0.001 * 10**18)
    if amount < min_wager:
        return 0

    return amount
