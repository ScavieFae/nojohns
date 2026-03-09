# Brief: Tournament Bracket System

**Branch:** scav/bracket-system
**Model:** sonnet

## Goal

Build the bracket engine and live bracket viewer for Fight Night. Single elimination, flexible size (byes for non-power-of-2), random seeding. The bracket viewer is the centerpiece of the room — displayed on a secondary monitor, showing the full tournament tree with match results updating live.

Reference: `tournaments/PROGRAM.md` for full context, `tournaments/SPEC.md` for prior architecture thinking.

## Design Decisions

- **Pacing:** Manual. Mattie operates the bracket from phone/laptop. No commentators. Operator taps "Start Next Match" when ready.
- **Stage:** Random stage every match (not FD-only). Adds variety.
- **Character:** Locked for whole tournament. Picked at registration, displayed on bracket.
- **Do-nothing vs do-nothing:** Coinflip. Special case — if both entries have strategy "do-nothing", skip the match and advance a random winner. This is the risk you take.
- **Exhibitions:** Separate computer, no bracket integration. `nojohns fight` on side machine.
- **Viewer:** Self-contained HTML file. No build step. Can open in Chrome on a projector.
- **State:** SQLite in arena DB. Survives restarts, consistent with existing infra.
- **Byes:** Show "BYE" in the empty slot. Auto-advance the real entry.
- **One match at a time.** One station, one screen, one audience focus.

## Tasks

1. Create `tournaments/models.py` — dataclasses for Tournament, Bracket, Round, Match, Entry. An Entry has: name, character, strategy (phillip/random/do-nothing), connect_code, wallet_address (optional). A Match has: round, slot, entry_a, entry_b, winner, score, pool_id, status (pending/playing/complete/bye/coinflip).

2. Create `tournaments/bracket.py` — single elimination bracket generation. Takes a list of entries, pads to nearest power of 2 with byes, shuffles for random seeding, returns a Bracket with all rounds pre-generated (empty matches for future rounds). Include `advance(bracket, round, slot, winner)` that fills in the next round's match. Auto-advance byes. Handle do-nothing vs do-nothing as coinflip (random winner, status "coinflip").

3. Create `tournaments/tournament.py` — lifecycle orchestration. `create_tournament(name, entries)` → Tournament with bracket. `queue_next_match(tournament)` → finds the next unplayed match and returns both entries' connect codes + characters + strategies. `report_result(tournament, round, slot, winner_name, score)` → advances bracket. State persisted in arena DB.

4. Add arena endpoints — `POST /tournaments` (create from entry list), `GET /tournaments/{id}` (full bracket state), `GET /tournaments/{id}/bracket` (bracket JSON for viewer), `POST /tournaments/{id}/advance` (report match result, advance bracket), `POST /tournaments/{id}/next` (operator triggers next match — queues both agents into arena matchmaking). Keep these minimal — the bracket viewer polls `/bracket`.

5. Build bracket viewer — self-contained HTML page at `tournaments/viewer.html`. Polls arena `/tournaments/{id}/bracket` every 3 seconds. Renders single-elimination tree with agent names, characters, match scores. Highlights current match. Shows "LIVE" badge on in-progress match. Shows "BYE" and "COINFLIP" for those match types. Clean enough for a projector at 1920x1080. No build step — vanilla HTML/CSS/JS.

6. Build admin panel — `tournaments/admin.html`. Buttons: create tournament (entry form for names/characters/strategies), start next match, force-advance match, force-expire match, rematch, manual bye. Mattie uses this on her phone. Simple, big buttons, confirmation on destructive actions.

## Completion Criteria

- [ ] Can create a 16-entry tournament with random seeding and byes
- [ ] Byes auto-advance, do-nothing vs do-nothing resolves as coinflip
- [ ] Bracket viewer renders full tree with names, characters, and match status
- [ ] "Start Next Match" from admin panel queues both agents into arena
- [ ] Advancing a match updates the bracket and the viewer reflects it within 3 seconds
- [ ] Admin panel can force-advance, trigger rematch, and start next match
- [ ] Bracket state persists across arena restarts (SQLite)

## Verification

- `.venv/bin/python -m pytest tests/ -v -o "addopts="` passes
- `curl localhost:8000/tournaments/test/bracket` returns valid bracket JSON
- Bracket viewer renders correctly in Chrome at 1920x1080
