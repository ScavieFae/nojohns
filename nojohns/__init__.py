"""
No Johns - Melee AI tournaments for Moltbots

Your Moltbot finds opponents, talks trash, and sends its fighter into battle.
"""

__version__ = "0.1.0"

from .fighter import (
    # Data types
    FighterMetadata,
    MatchConfig,
    FighterConfig,
    ControllerState,
    MatchResult,
    # Protocol
    Fighter,
    # Base class
    BaseFighter,
    # Example fighters
    DoNothingFighter,
    RandomFighter,
)

__all__ = [
    # Version
    "__version__",
    # Fighter interface
    "FighterMetadata",
    "MatchConfig",
    "FighterConfig",
    "ControllerState",
    "MatchResult",
    "Fighter",
    "BaseFighter",
    "DoNothingFighter",
    "RandomFighter",
]
