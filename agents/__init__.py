"""
agents/ - Autonomous agent toolkit for No Johns

Building blocks for strategic decision-making. Compose these however you want.

    from agents.bankroll import get_bankroll_state, kelly_wager
    from agents.scouting import scout_opponent
    from agents.strategy import KellyStrategy, WagerStrategy
"""

from agents.bankroll import (
    BankrollState,
    get_mon_balance,
    get_active_wager_exposure,
    get_bankroll_state,
    win_probability_from_elo,
    kelly_fraction,
    kelly_wager,
)
from agents.scouting import ScoutReport, scout_opponent, scout_by_wallet
from agents.strategy import (
    MatchContext,
    WagerDecision,
    SessionStats,
    WagerStrategy,
    KellyStrategy,
)

__all__ = [
    "BankrollState",
    "get_mon_balance",
    "get_active_wager_exposure",
    "get_bankroll_state",
    "win_probability_from_elo",
    "kelly_fraction",
    "kelly_wager",
    "ScoutReport",
    "scout_opponent",
    "scout_by_wallet",
    "MatchContext",
    "WagerDecision",
    "SessionStats",
    "WagerStrategy",
    "KellyStrategy",
]
