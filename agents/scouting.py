"""
agents/scouting.py - Opponent lookup utilities.

Wraps existing reputation.py calls into a clean ScoutReport.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ScoutReport:
    """What we know about an opponent before deciding on a wager."""

    elo: int
    peak_elo: int
    record: str  # "W-L" format
    is_unknown: bool  # True if no onchain data found
    agent_id: int | None = None


def scout_opponent(
    agent_id: int, rpc_url: str, reputation_registry: str
) -> ScoutReport:
    """Look up an opponent by their ERC-8004 agent ID."""
    try:
        from nojohns.reputation import get_current_elo, STARTING_ELO
    except ImportError:
        logger.debug("reputation module not available")
        return ScoutReport(
            elo=1500, peak_elo=1500, record="0-0", is_unknown=True, agent_id=agent_id
        )

    state = get_current_elo(agent_id, rpc_url, reputation_registry)

    is_unknown = (state.elo == 1500 and state.wins == 0 and state.losses == 0)

    return ScoutReport(
        elo=state.elo,
        peak_elo=state.peak_elo,
        record=state.record,
        is_unknown=is_unknown,
        agent_id=agent_id,
    )


def scout_by_wallet(
    wallet: str, rpc_url: str, reputation_registry: str
) -> ScoutReport:
    """Look up an opponent by wallet address.

    Currently returns unknown — requires IdentityRegistry reverse lookup
    (wallet → agent_id) which isn't implemented yet. Agents without
    registered identities are always unknown.
    """
    # TODO: reverse lookup wallet → agent_id via IdentityRegistry
    # For now, return unknown so strategy can handle it
    return ScoutReport(
        elo=1500, peak_elo=1500, record="0-0", is_unknown=True, agent_id=None
    )
