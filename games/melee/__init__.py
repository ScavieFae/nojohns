"""
games.melee - Melee/Dolphin/Slippi integration for No Johns

Provides the match runner, netplay runner, and all Melee-specific types.
Core fighter protocol lives in nojohns.fighter — this package handles
the Dolphin side of actually running games.
"""

from .runner import (
    DolphinConfig,
    MatchSettings,
    GameResult,
    MatchResult,
    MatchRunner,
    quick_fight,
)

from .netplay import (
    NetplayConfig,
    NetplayRunner,
    NetplayDisconnectedError,
    netplay_test,
)

__all__ = [
    # Runner
    "DolphinConfig",
    "MatchSettings",
    "GameResult",
    "MatchResult",
    "MatchRunner",
    "quick_fight",
    # Netplay
    "NetplayConfig",
    "NetplayRunner",
    "NetplayDisconnectedError",
    "netplay_test",
]
