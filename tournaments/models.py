"""
tournaments/models.py — Data models for tournament bracket system.

All models are plain dataclasses — no ORM, no magic. Serialization to/from
dict is manual (for SQLite JSON columns and HTTP responses).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

Strategy = Literal["phillip", "random", "do-nothing", "smashbot"]

MatchStatus = Literal["pending", "playing", "complete", "bye", "coinflip"]


@dataclass
class Entry:
    """A tournament participant."""

    name: str
    character: str  # e.g. "FOX", "MARTH", "JIGGLYPUFF"
    strategy: Strategy
    connect_code: str  # Slippi connect code, e.g. "NOJN#001"
    wallet_address: str | None = None
    registrant: str | None = None  # Human registrant name (who signed up)
    email: str | None = None  # Email used at registration (for Privy matching)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "character": self.character,
            "strategy": self.strategy,
            "connect_code": self.connect_code,
            "wallet_address": self.wallet_address,
            "registrant": self.registrant,
            "email": self.email,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Entry":
        return cls(
            name=d["name"],
            character=d["character"],
            strategy=d["strategy"],
            connect_code=d["connect_code"],
            wallet_address=d.get("wallet_address"),
            registrant=d.get("registrant"),
            email=d.get("email"),
        )


# ---------------------------------------------------------------------------
# Match
# ---------------------------------------------------------------------------

@dataclass
class Match:
    """A single match in the bracket.

    entry_a / entry_b are None for empty slots (future rounds) or the BYE side.
    winner is None until the match is resolved.
    """

    round: int        # 0 = first round, 1 = quarterfinals if 16-entry, etc.
    slot: int         # Position within the round (0-indexed)
    entry_a: Entry | None = None
    entry_b: Entry | None = None
    winner: Entry | None = None
    score_a: int | None = None   # stocks remaining for entry_a
    score_b: int | None = None   # stocks remaining for entry_b
    pool_id: int | None = None   # onchain prediction pool ID
    status: MatchStatus = "pending"
    arena_match_id: str | None = None  # arena match ID once queued

    def to_dict(self) -> dict:
        return {
            "round": self.round,
            "slot": self.slot,
            "entry_a": self.entry_a.to_dict() if self.entry_a else None,
            "entry_b": self.entry_b.to_dict() if self.entry_b else None,
            "winner": self.winner.to_dict() if self.winner else None,
            "score_a": self.score_a,
            "score_b": self.score_b,
            "pool_id": self.pool_id,
            "status": self.status,
            "arena_match_id": self.arena_match_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Match":
        return cls(
            round=d["round"],
            slot=d["slot"],
            entry_a=Entry.from_dict(d["entry_a"]) if d.get("entry_a") else None,
            entry_b=Entry.from_dict(d["entry_b"]) if d.get("entry_b") else None,
            winner=Entry.from_dict(d["winner"]) if d.get("winner") else None,
            score_a=d.get("score_a"),
            score_b=d.get("score_b"),
            pool_id=d.get("pool_id"),
            status=d.get("status", "pending"),
            arena_match_id=d.get("arena_match_id"),
        )


# ---------------------------------------------------------------------------
# Bracket
# ---------------------------------------------------------------------------

@dataclass
class Bracket:
    """Single-elimination bracket.

    rounds[0] = first round (all entries), rounds[-1] = final.
    Each round has len(rounds[r-1]) // 2 matches.
    """

    rounds: list[list[Match]] = field(default_factory=list)
    size: int = 0  # Power of 2 — actual bracket capacity (>= len(entries))

    def to_dict(self) -> dict:
        return {
            "size": self.size,
            "rounds": [[m.to_dict() for m in round_] for round_ in self.rounds],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Bracket":
        return cls(
            size=d["size"],
            rounds=[[Match.from_dict(m) for m in round_] for round_ in d["rounds"]],
        )

    def get_match(self, round: int, slot: int) -> Match | None:
        if round < 0 or round >= len(self.rounds):
            return None
        r = self.rounds[round]
        if slot < 0 or slot >= len(r):
            return None
        return r[slot]

    def champion(self) -> Entry | None:
        """Return the tournament winner if determined."""
        if not self.rounds:
            return None
        final = self.rounds[-1]
        if len(final) == 1 and final[0].winner:
            return final[0].winner
        return None


# ---------------------------------------------------------------------------
# Tournament
# ---------------------------------------------------------------------------

@dataclass
class Tournament:
    """Top-level tournament record."""

    id: str
    name: str
    bracket: Bracket
    entries: list[Entry] = field(default_factory=list)
    status: Literal["registration", "pending", "active", "complete"] = "registration"
    current_round: int = 0
    current_slot: int = 0  # next match to play
    featured: bool = False  # Show on homepage

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "current_round": self.current_round,
            "current_slot": self.current_slot,
            "featured": self.featured,
            "entries": [e.to_dict() for e in self.entries],
            "bracket": self.bracket.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Tournament":
        return cls(
            id=d["id"],
            name=d["name"],
            bracket=Bracket.from_dict(d["bracket"]),
            entries=[Entry.from_dict(e) for e in d.get("entries", [])],
            status=d.get("status", "pending"),
            featured=d.get("featured", False),
            current_round=d.get("current_round", 0),
            current_slot=d.get("current_slot", 0),
        )
