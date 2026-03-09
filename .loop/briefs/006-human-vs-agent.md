# Brief: Human vs Agent Station

**Branch:** scav/human-vs-agent
**Model:** sonnet

## Goal

Side station where humans walk up and fight Phillip. The "arcade cabinet" of Fight Night — something to do between bracket matches, and a hook even if nobody cares about watching agents fight each other. Running scoreboard of human wins vs agent wins.

## Setup

One machine, one Dolphin instance, no netplay:
- **Port 1:** `ControllerType.STANDARD` (named pipe) — Phillip, controlled by libmelee
- **Port 2:** `ControllerType.GCN_ADAPTER` (USB dongle) — human, real GameCube controller

libmelee supports `GCN_ADAPTER` as a controller type. Dolphin's `setup_dolphin_controller()` configures the port. The runner skips `act()` for the human port — the GCN adapter handles input directly.

**Hardware needed:** One Mac, one USB GCC adapter (Mattie has a dongle), one or more GCC controllers, one monitor.

## Tasks

1. Add a `challenge` mode to `MatchRunner` in `games/melee/runner.py`. Port 1 gets a Fighter (bot-controlled via STANDARD pipe), port 2 gets `ControllerType.GCN_ADAPTER` (human-controlled, no Fighter). The runner's game loop calls `act()` only for port 1 and skips port 2 entirely. Game-end detection works the same (stocks hit 0).

2. Add `nojohns challenge <fighter>` CLI command. Launches Dolphin with the challenge mode runner. Takes fighter name (default: phillip), character for the bot, and optionally the human's name for the scoreboard.

3. Report results to the arena with `match_type: "challenge"`. Arena needs a `match_type` field on matches (values: "tournament", "challenge", "exhibition") so the scoreboard can filter.

4. Build scoreboard — `tournaments/scoreboard.html`. Self-contained HTML, no build step. Shows:
   - Running tally: "Humans: 3 — Phillip: 12"
   - Recent matches with human name (entered before each match via CLI prompt or a simple form) and result
   - Win rate percentage
   Polls arena for challenge-type match results.

5. Test the GCN adapter setup. Verify: Dolphin detects the adapter, human inputs work on port 2, bot plays normally on port 1, game-end detection fires, result reports correctly.

## Risk

- **GCN adapter detection in Dolphin.** libmelee's `setup_dolphin_controller()` writes the adapter config, but Dolphin may need the adapter plugged in at launch. If it doesn't detect, fallback: human uses keyboard (Dolphin's built-in keyboard bindings for port 2).
- **Untested path.** We've never run STANDARD + GCN_ADAPTER mixed. Needs real hardware testing Tuesday.

## Completion Criteria

- [ ] Human can play against Phillip on one machine with a GCC controller
- [ ] Results reported to arena with match_type "challenge"
- [ ] Scoreboard page shows running human vs agent tally
- [ ] `nojohns challenge phillip` works as a one-command launcher

## Verification

- Manual test: plug in GCC, run `nojohns challenge phillip`, play one match, verify result on scoreboard
- Scoreboard page renders in Chrome at 1080p
