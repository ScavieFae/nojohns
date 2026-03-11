"""
tournaments/bracket.py — Single elimination bracket generation and advancement.

Generates brackets with byes for non-power-of-2 entry counts.
Handles auto-advancing byes and do-nothing vs do-nothing coinflip resolution.
"""

from __future__ import annotations

import random

from .models import Bracket, Entry, Match


def _next_power_of_two(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


def _is_coinflip(entry_a: Entry, entry_b: Entry) -> bool:
    """Both entries use do-nothing — resolve as coinflip instead of playing."""
    return entry_a.strategy == "do-nothing" and entry_b.strategy == "do-nothing"


def generate_bracket(entries: list[Entry], seed: int | None = None) -> Bracket:
    """
    Build a full single-elimination bracket from a list of entries.

    - Pads to nearest power of 2 with byes (None slots)
    - Shuffles entries for random seeding
    - Pre-generates empty matches for all future rounds
    - Auto-advances byes and do-nothing vs do-nothing coinflips on generation

    Returns a Bracket ready for tournament use.
    """
    if not entries:
        raise ValueError("Need at least 1 entry to generate a bracket")

    rng = random.Random(seed)
    size = _next_power_of_two(len(entries))
    num_rounds = size.bit_length() - 1  # log2(size)

    # Shuffle for random seeding
    shuffled = entries[:]
    rng.shuffle(shuffled)

    # Pad to power-of-2 with None (bye slots)
    padded: list[Entry | None] = shuffled + [None] * (size - len(shuffled))

    # Build first round
    first_round: list[Match] = []
    for slot in range(size // 2):
        entry_a = padded[slot * 2]
        entry_b = padded[slot * 2 + 1]
        match = Match(round=0, slot=slot, entry_a=entry_a, entry_b=entry_b)

        # Resolve immediately if bye or double-do-nothing
        if entry_a is None and entry_b is None:
            match.status = "bye"
        elif entry_a is None:
            match.status = "bye"
            match.winner = entry_b
        elif entry_b is None:
            match.status = "bye"
            match.winner = entry_a
        elif _is_coinflip(entry_a, entry_b):
            match.status = "coinflip"
            match.winner = rng.choice([entry_a, entry_b])

        first_round.append(match)

    # Pre-generate empty matches for all future rounds
    rounds: list[list[Match]] = [first_round]
    for r in range(1, num_rounds):
        count = size // (2 ** (r + 1))
        rounds.append([Match(round=r, slot=s) for s in range(count)])

    # Propagate resolved matches (byes/coinflips) into later rounds
    _propagate_winners(rounds, rng)

    return Bracket(rounds=rounds, size=size)


def _propagate_winners(rounds: list[list[Match]], rng: random.Random | None = None) -> None:
    """
    Push resolved match winners forward into subsequent rounds.

    Two-phase approach per iteration:
      1. Place all winners into their next-round slots
      2. Then check for byes/coinflips (only after all placements are done)

    Loops until no new advancement occurs (handles cascading byes).
    """
    if rng is None:
        rng = random.Random()

    changed = True
    while changed:
        changed = False

        # Phase 1: Place all winners into next-round slots
        for r_idx in range(len(rounds) - 1):
            current_round = rounds[r_idx]
            next_round = rounds[r_idx + 1]
            for slot, match in enumerate(current_round):
                if match.winner is None:
                    continue
                next_slot = slot // 2
                next_match = next_round[next_slot]
                if slot % 2 == 0:
                    if next_match.entry_a is not match.winner:
                        next_match.entry_a = match.winner
                        changed = True
                else:
                    if next_match.entry_b is not match.winner:
                        next_match.entry_b = match.winner
                        changed = True

        # Phase 2: Resolve byes and coinflips (after all placements)
        for r_idx in range(len(rounds) - 1):
            current_round = rounds[r_idx]
            next_round = rounds[r_idx + 1]
            for next_slot, next_match in enumerate(next_round):
                if next_match.winner is not None:
                    continue
                a, b = next_match.entry_a, next_match.entry_b
                # Check if both feeder matches are resolved
                feeder_a = current_round[next_slot * 2] if next_slot * 2 < len(current_round) else None
                feeder_b = current_round[next_slot * 2 + 1] if next_slot * 2 + 1 < len(current_round) else None
                a_resolved = feeder_a is not None and feeder_a.status in ("bye", "complete", "coinflip")
                b_resolved = feeder_b is not None and feeder_b.status in ("bye", "complete", "coinflip")

                if a is None and b is not None and b_resolved and a_resolved:
                    next_match.status = "bye"
                    next_match.winner = b
                    changed = True
                elif b is None and a is not None and a_resolved and b_resolved:
                    next_match.status = "bye"
                    next_match.winner = a
                    changed = True
                elif a is not None and b is not None and _is_coinflip(a, b):
                    next_match.status = "coinflip"
                    next_match.winner = rng.choice([a, b])
                    changed = True


def advance(
    bracket: Bracket,
    round: int,
    slot: int,
    winner: Entry,
    score_a: int | None = None,
    score_b: int | None = None,
) -> None:
    """
    Record a match result and advance the winner to the next round.

    Sets match status to "complete", stores scores, then propagates the
    winner forward (may auto-resolve downstream byes/coinflips).
    """
    match = bracket.get_match(round, slot)
    if match is None:
        raise ValueError(f"No match at round={round}, slot={slot}")
    if match.status not in ("pending", "playing"):
        raise ValueError(f"Match at round={round}, slot={slot} is already {match.status!r}")

    match.winner = winner
    match.score_a = score_a
    match.score_b = score_b
    match.status = "complete"

    _propagate_winners(bracket.rounds)


def next_playable(bracket: Bracket) -> Match | None:
    """
    Return the first match that needs a real fight: status pending/playing with both entries set.

    Byes and coinflips are auto-resolved at generation/advance time and never returned here.
    """
    for round_ in bracket.rounds:
        for match in round_:
            if match.status in ("pending", "playing") and match.entry_a and match.entry_b:
                return match
    return None
