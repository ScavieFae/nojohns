"""
arena - Lightweight matchmaking server for No Johns

Brokers connections between fighters via HTTP. The arena never touches
the game — it just pairs players and records results.
"""

from .server import app
from .db import ArenaDB

__all__ = ["app", "ArenaDB"]
