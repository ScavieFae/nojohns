"""
tournaments/tournament.py — Tournament lifecycle orchestration.

High-level functions for creating tournaments, queuing matches, and reporting
results. State is persisted in the arena SQLite DB via ArenaDB.

Usage::

    from arena.db import ArenaDB
    from tournaments.models import Entry
    from tournaments.tournament import create_tournament, queue_next_match, report_result

    db = ArenaDB("arena.db")
    entries = [Entry("AlphaBot", "FOX", "phillip", "ABLY#001"), ...]
    t = create_tournament(db, "Fight Night 001", entries)

    match = queue_next_match(db, t)   # Returns Match with arena_match_id set
    t = report_result(db, t, match.round, match.slot, "AlphaBot", score_a=3, score_b=0)
"""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING

from .bracket import advance as bracket_advance
from .bracket import generate_bracket, next_playable
from .models import Entry, Match, Tournament

if TYPE_CHECKING:
    from arena.db import ArenaDB


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def _save(db: "ArenaDB", tournament: Tournament) -> None:
    """Persist a tournament to the arena DB."""
    db.save_tournament(
        tournament_id=tournament.id,
        name=tournament.name,
        status=tournament.status,
        data=json.dumps(tournament.to_dict()),
    )


def _load(db: "ArenaDB", tournament_id: str) -> Tournament | None:
    """Load a tournament from the arena DB. Returns None if not found."""
    row = db.load_tournament(tournament_id)
    if row is None:
        return None
    return Tournament.from_dict(json.loads(row["data"]))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_tournament(
    db: "ArenaDB", name: str, entries: list[Entry] | None = None
) -> Tournament:
    """
    Create a new tournament.

    If entries are provided, generates a bracket immediately (status="pending").
    If no entries, creates an empty tournament in "registration" status so
    fighters can be added one at a time via register_entry().
    """
    tournament_id = str(uuid.uuid4())[:8]

    if entries:
        bracket = generate_bracket(entries)
        status = "pending"
    else:
        from .models import Bracket

        bracket = Bracket()
        entries = []
        status = "registration"

    tournament = Tournament(
        id=tournament_id,
        name=name,
        bracket=bracket,
        entries=list(entries),
        status=status,
    )
    _save(db, tournament)
    return tournament


def register_entry(db: "ArenaDB", tournament: Tournament, entry: Entry) -> Tournament:
    """
    Add a fighter to a tournament that's in registration.

    Raises ValueError if tournament is not in registration status or
    if the fighter name is already taken.
    """
    if tournament.status != "registration":
        raise ValueError(f"Tournament is '{tournament.status}', not accepting registrations")

    if any(e.name == entry.name for e in tournament.entries):
        raise ValueError(f"Fighter name '{entry.name}' is already registered")

    tournament.entries.append(entry)
    _save(db, tournament)
    return tournament


def close_registration(db: "ArenaDB", tournament: Tournament) -> Tournament:
    """
    Lock entries and generate the bracket.

    Requires at least 2 entries. Sets status to "pending" (ready to play).
    """
    if tournament.status != "registration":
        raise ValueError(f"Tournament is '{tournament.status}', can't close registration")

    if len(tournament.entries) < 2:
        raise ValueError(f"Need at least 2 entries (have {len(tournament.entries)})")

    tournament.bracket = generate_bracket(tournament.entries)
    tournament.status = "pending"
    _save(db, tournament)
    return tournament


def get_tournament(db: "ArenaDB", tournament_id: str) -> Tournament | None:
    """Load a tournament by ID. Returns None if not found."""
    return _load(db, tournament_id)


def list_tournaments(db: "ArenaDB") -> list[dict]:
    """List all tournaments (summary only — no bracket data). Newest first."""
    return db.list_tournaments()


def queue_next_match(db: "ArenaDB", tournament: Tournament) -> Match | None:
    """
    Find the next unplayed match and queue both agents into the arena.

    - Adds both entries to the arena queue
    - Immediately pairs them (creates an arena match, bypassing normal FIFO)
    - Sets match.status = "playing" and match.arena_match_id on the bracket
    - Persists updated tournament state

    Returns the Match (with arena_match_id set), or None if no matches remain.
    """
    match = next_playable(tournament.bracket)
    if match is None:
        if tournament.bracket.champion() is not None and tournament.status != "complete":
            tournament.status = "complete"
            _save(db, tournament)
        return None

    assert match.entry_a is not None
    assert match.entry_b is not None

    # Add both entries to the arena queue (cancels any stale entries for these codes)
    q1_id = db.add_to_queue(
        connect_code=match.entry_a.connect_code,
        fighter_name=match.entry_a.name,
        wallet_address=match.entry_a.wallet_address,
    )
    q2_id = db.add_to_queue(
        connect_code=match.entry_b.connect_code,
        fighter_name=match.entry_b.name,
        wallet_address=match.entry_b.wallet_address,
    )

    # Immediately pair them — bypass normal FIFO matchmaking
    p1_entry = db.get_queue_entry(q1_id)
    p2_entry = db.get_queue_entry(q2_id)
    arena_match_id = db.create_match(p1_entry, p2_entry)

    # Update bracket state
    match.status = "playing"
    match.arena_match_id = arena_match_id
    tournament.status = "active"
    tournament.current_round = match.round
    tournament.current_slot = match.slot

    _save(db, tournament)
    return match


def report_result(
    db: "ArenaDB",
    tournament: Tournament,
    round: int,
    slot: int,
    winner_name: str,
    score_a: int | None = None,
    score_b: int | None = None,
) -> Tournament:
    """
    Record a match result and advance the bracket.

    winner_name must match one of the match entries' names exactly.
    Cascading byes and coinflips are auto-resolved by bracket.advance().
    Persists updated tournament state.
    """
    match = tournament.bracket.get_match(round, slot)
    if match is None:
        raise ValueError(f"No match at round={round}, slot={slot}")

    winner: Entry | None = None
    if match.entry_a and match.entry_a.name == winner_name:
        winner = match.entry_a
    elif match.entry_b and match.entry_b.name == winner_name:
        winner = match.entry_b

    if winner is None:
        raise ValueError(f"No entry named {winner_name!r} in match ({round}, {slot})")

    bracket_advance(tournament.bracket, round, slot, winner, score_a, score_b)

    if tournament.bracket.champion() is not None:
        tournament.status = "complete"

    _save(db, tournament)
    return tournament


def force_advance(
    db: "ArenaDB",
    tournament: Tournament,
    round: int,
    slot: int,
    winner_name: str,
) -> Tournament:
    """
    Admin override: force-advance a match regardless of current status.

    Same as report_result but accepts any match status (e.g. "playing", "pending").
    Use when a match needs to be skipped or manually resolved.
    """
    match = tournament.bracket.get_match(round, slot)
    if match is None:
        raise ValueError(f"No match at round={round}, slot={slot}")

    winner: Entry | None = None
    if match.entry_a and match.entry_a.name == winner_name:
        winner = match.entry_a
    elif match.entry_b and match.entry_b.name == winner_name:
        winner = match.entry_b

    if winner is None:
        raise ValueError(f"No entry named {winner_name!r} in match ({round}, {slot})")

    # Force status to pending so bracket_advance() accepts it
    match.status = "pending"
    bracket_advance(tournament.bracket, round, slot, winner)

    if tournament.bracket.champion() is not None:
        tournament.status = "complete"

    _save(db, tournament)
    return tournament
