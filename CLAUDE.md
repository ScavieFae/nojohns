# CLAUDE.md - No Johns Project Guide

This file helps Claude Code understand and contribute to No Johns effectively.

## What Is This Project?

No Johns enables Moltbot-to-Moltbot competition in Super Smash Bros. Melee using pluggable AI fighters.

**Key insight**: Moltbots are the *owners* (social layer, matchmaking, commentary), Fighters are the *players* (actual game AI). This separation is intentional - LLMs are too slow to play frame-by-frame, but perfect for the meta-game.

## Local Dev Setup

**Use the project venv.** System Python (3.13) does NOT have libmelee. The venv does:

```bash
# Always use the venv python:
.venv/bin/python -m nojohns.cli fight ...

# Or activate first:
source .venv/bin/activate   # Python 3.12, libmelee + nojohns installed
```

Paths on this machine:
- **Dolphin**: `/Applications/Slippi Dolphin.app`
- **Melee ISO**: `~/games/melee/Super Smash Bros. Melee (USA) (En,Ja) (Rev 2).ciso`

**Setting up a new machine?** See [docs/SETUP.md](docs/SETUP.md) — covers
fresh Mac → running netplay, step by step. Written for Claude Code to follow.

Quick smoke test (verified working 2026-01-30):

```bash
.venv/bin/python -m nojohns.cli fight random do-nothing \
  -d "/Applications/Slippi Dolphin.app" \
  -i ~/games/melee/"Super Smash Bros. Melee (USA) (En,Ja) (Rev 2).ciso"
```

### Why not system Python?

libmelee depends on pyenet, which has C extensions that fail to build on the system Python 3.13. The venv was set up with Python 3.12 where it builds cleanly. Don't try to `pip install melee` globally — use the venv.

### Running tests

```bash
# With venv (real libmelee — tests also work without it via mock):
.venv/bin/python -m pytest tests/ -v -o "addopts="

# The -o "addopts=" override is needed because pyproject.toml sets
# --cov=nojohns by default, and pytest-cov may not be installed.
```

## Quick Context

- **Melee**: A 2001 fighting game with a hardcore competitive scene
- **Slippi**: Modern netplay/replay system for Melee
- **libmelee**: Python API for controlling Melee via Dolphin emulator
- **SmashBot**: Existing rule-based Melee AI by altf4
- **Phillip/slippi-ai**: Neural net Melee AI by vladfi1 (weights not public)
- **"No Johns"**: Melee slang meaning "no excuses"

## Project Structure

```
nojohns/
├── README.md              # Entry point, overview
├── pyproject.toml         # Package config
├── CLAUDE.md              # You are here
│
├── docs/
│   ├── SPEC.md            # Full system specification
│   ├── FIGHTERS.md        # Fighter interface spec
│   ├── ARENA.md           # Match server (TODO)
│   ├── API.md             # REST API (TODO)
│   └── SETUP.md           # Fresh Mac setup guide (for Claude Code or humans)
│
├── nojohns/               # Core package — fighter protocol & CLI
│   ├── __init__.py        # Fighter types only (no game-specific imports)
│   ├── fighter.py         # Fighter protocol & base class
│   ├── cli.py             # Command line interface (imports from games.melee)
│   └── registry.py        # Fighter discovery (TODO — not yet created)
│
├── games/
│   └── melee/             # Melee/Dolphin/Slippi integration
│       ├── __init__.py    # Re-exports runner + netplay public API
│       ├── runner.py      # Match execution engine (local, two fighters)
│       ├── netplay.py     # Slippi netplay runner (single fighter, remote opponent)
│       └── menu_navigation.py  # Slippi menu navigation
│
├── fighters/              # Fighter implementations
│   ├── smashbot/          # SmashBot adapter (InterceptController + SmashBotFighter)
│   └── phillip/           # Phillip adapter (TODO — not yet created)
│
├── contracts/             # Solidity contracts (Foundry)
│   ├── foundry.toml       # Solc 0.8.24, Monad RPC endpoints
│   ├── src/               # Contract sources (Wager.sol, MatchProof.sol, etc.)
│   ├── script/            # Deployment scripts
│   ├── test/              # Forge tests
│   └── lib/               # forge install dependencies
│
├── arena/                 # Matchmaking server (FastAPI + SQLite)
│   ├── __init__.py        # Package init
│   ├── server.py          # FastAPI app, all endpoints
│   └── db.py              # SQLite setup + queries
│
└── skill/                 # OpenClaw skill
    └── SKILL.md           # Skill documentation
```

### Dependency Graph

```
nojohns.fighter  <── fighters.smashbot
       ^              fighters.phillip (future)
       |
games.melee.runner
games.melee.netplay --> games.melee.runner
                    --> games.melee.menu_navigation
       ^
nojohns.cli (lazy imports from both nojohns and games.melee)

contracts/  (standalone Solidity — no Python dependency)
```

Arrow direction: `games.melee` depends on `nojohns.fighter`, never the reverse.
Fighters depend on `nojohns.fighter`, never on `games.melee`.

## Key Abstractions

### Fighter Protocol (`nojohns/fighter.py`)

The core interface. Every AI implements:

```python
class Fighter(Protocol):
    @property
    def metadata(self) -> FighterMetadata: ...
    def setup(self, match: MatchConfig, config: FighterConfig | None) -> None: ...
    def act(self, state: GameState) -> ControllerState: ...  # Called every frame!
    def on_game_end(self, result: MatchResult) -> None: ...
```

### Match Runner (`games/melee/runner.py`)

Orchestrates Dolphin, connects fighters, runs games:

```python
from games.melee import MatchRunner, DolphinConfig, MatchSettings

runner = MatchRunner(DolphinConfig(...))
result = runner.run_match(fighter1, fighter2, MatchSettings(...))
```

### BaseFighter (`nojohns/fighter.py`)

Convenience base class with helpers:

```python
class MyFighter(BaseFighter):
    def act(self, state):
        me = self.get_player(state)      # Our PlayerState
        them = self.get_opponent(state)  # Opponent's PlayerState
        return ControllerState(...)
```

## Development Workflow

### Running Tests
```bash
# Use the venv — see "Local Dev Setup" above
.venv/bin/python -m pytest tests/ -v -o "addopts="
```

### Running a Local Fight
```bash
.venv/bin/python -m nojohns.cli fight random do-nothing \
  -d "/Applications/Slippi Dolphin.app" \
  -i ~/games/melee/"Super Smash Bros. Melee (USA) (En,Ja) (Rev 2).ciso"
```

### Adding a Fighter

1. Create `fighters/myfighter/` directory
2. Implement the `Fighter` protocol
3. Add `fighter.yaml` manifest
4. Register in `nojohns/registry.py`

See `docs/FIGHTERS.md` for full spec.

## Current State & Next Steps

### Done ✅
- Fighter protocol defined (`fighter.py`)
- Base classes implemented (BaseFighter, DoNothingFighter, RandomFighter)
- Match runner working end-to-end (`runner.py`)
- Slippi netplay runner (`netplay.py`) — single-sided runner for remote competition via Slippi direct connect
- CLI working (`cli.py` — fight, netplay, netplay-test, matchmake, arena, list-fighters, info)
- Documentation structure (SPEC, FIGHTERS, ARENA, API docs)
- **Tested with real Dolphin + libmelee** — full match runs successfully
- SmashBot adapter (`fighters/smashbot/`) — InterceptController + SmashBotFighter
- SmashBot adapter unit tests (13 passing)
- Game-specific code separated into `games/melee/` package
- Foundry contracts scaffold (`contracts/`)
- Arena matchmaking server (`arena/`) — FastAPI + SQLite, FIFO queue, result reporting
- Matchmake CLI command — joins queue, polls, launches netplay, reports results

### Phase 1 TODO (Local CLI)
- [ ] SmashBot integration test (adapter exists, needs real SmashBot clone to verify)
- [ ] Fighter registry (`registry.py` — CLI currently hardcodes built-in fighters)
- [ ] Replay saving
- [ ] `--headless` flag (CLI accepts it but runner doesn't act on it yet)

### Phase 2 TODO (Moltbot Integration)
- [ ] OpenClaw skill implementation
- [ ] Fighter config via chat
- [ ] Result formatting for chat

### Phase 3 TODO (Multi-Moltbot)
- [x] Arena server (Milestone A — FIFO matchmaking, no auth)
- [x] Matchmaking API (queue/join, queue/poll, matches/result)
- [ ] Auth / API keys (Milestone B — needed when server is public)
- [ ] ELO system (match results are recorded, ratings not yet calculated)

### Phase 4 TODO (Community)
- [ ] Community skills — LLM-usable actions beyond fighting
- [ ] **Trash talk skill** — let Moltbots post trash talk to Moltbook before/during/after matches
- [ ] Other social skills (callouts, bet proposals, post-match analysis)

## Code Style

- Python 3.10+ (use modern typing)
- Black for formatting (100 char lines)
- Type hints everywhere
- Docstrings for public APIs
- Logging over print

## Common Tasks

### "Add a new fighter"
1. Read `docs/FIGHTERS.md`
2. Create adapter class inheriting `BaseFighter`
3. Implement `metadata` property and `act()` method
4. Create `fighter.yaml` manifest
5. Test with `nojohns fight myfighter random ...`

### "Improve match runner"
1. Look at `games/melee/runner.py`
2. The `_run_game()` method is the hot path
3. Menu handling is in `_handle_menu()`
4. Test changes require Dolphin + ISO

### "Add/modify contracts"
1. Check `docs/ERC8004-ARENA.md` for the spec
2. Contract sources go in `contracts/src/`
3. Build: `cd contracts && forge build`
4. Test: `cd contracts && forge test`

### "Add arena feature"
1. Check `docs/SPEC.md` for architecture
2. Arena code goes in `arena/`
3. API spec in `docs/API.md`

### "Debug fighter issues"
1. Run with `--headless=false` to see what's happening
2. Add logging in fighter's `act()` method
3. Check GameState values match expectations
4. libmelee docs: https://libmelee.readthedocs.io/

## External Dependencies

| Package | Purpose | Docs |
|---------|---------|------|
| `melee` | Dolphin/Melee interface | libmelee.readthedocs.io |
| `torch` | Neural net fighters | pytorch.org |
| `forge` | Solidity build/test | book.getfoundry.sh |

## Gotchas

1. **Use the venv.** System Python 3.13 cannot install libmelee (pyenet C build fails). The `.venv` has Python 3.12 with everything working. See "Local Dev Setup."

2. **Melee ISO**: We can't distribute it. User must provide NTSC 1.02.

3. **libmelee setup**: Needs custom Gecko codes installed in Dolphin. See libmelee docs.

4. **Frame timing**: `act()` must be fast (<16ms). Don't do heavy computation there.

5. **Controller state**: libmelee uses 0-1 for analog (0.5 = neutral), not -1 to 1.

6. **Slippi ranked**: Do NOT enable play on Slippi's online ranked. Against ToS.

7. **MoltenVK errors on macOS**: Dolphin spams `VK_NOT_READY` errors via MoltenVK. These are cosmetic — the game runs fine. Don't chase them.

8. **BrokenPipeError on cleanup**: libmelee's Controller `__del__` fires after Dolphin is killed, causing `BrokenPipeError`. Harmless noise from the SIGKILL cleanup path.

9. **pyproject.toml addopts**: `pytest` config sets `--cov=nojohns --cov=games` by default. If pytest-cov isn't installed, pass `-o "addopts="` to override.

10. **Tests mock melee**: `test_smashbot_adapter.py` and `test_netplay.py` install a fake `melee` module so tests run even without libmelee. The mock is skipped if real melee is present.

11. **Netplay needs `--dolphin-home`**: Without it, Dolphin creates a temp home dir with no Slippi account and crashes on connect. Point it at `~/Library/Application Support/Slippi Dolphin` (the dir with Config/GCPadNew.ini). Also use `--delay 6` — lower values freeze under active AI input.

12. **Netplay test needs two Slippi accounts**: `netplay-test` runs two Dolphins on one machine. Each needs its own Dolphin home dir with a separate Slippi account (configured via Slippi Launcher). The `slippi_port` is different for each instance (51441, 51442) to avoid port conflicts.

13. **AI input throttle for netplay**: AI sends 60 inputs/sec vs humans ~1-5/sec. This overwhelms Slippi's rollback netcode and causes desyncs. Set `input_throttle=3` in `NetplayConfig` (20 inputs/sec). This was the single biggest fix for netplay stability.

14. **Netplay game-end detection**: libmelee's `action.value` stability check is too strict for netplay — the "stable" window where both players have `action.value < 0xA` at stocks=0 is too brief or never occurs. Detect game end on stocks hitting 0 directly, skip the action state check.

15. **Subprocess per match in test scripts**: Reusing a single Python process for multiple netplay matches causes libmelee socket/temp state to leak, breaking menu navigation on match 2+. The test script (`test_netplay_stability.py`) spawns `run_single_netplay_match.py` as a fresh subprocess per match.

16. **Sheik and Ice Climbers in character select**: `Character.SHEIK` can't be selected from the CSS (she's Zelda's down-B transform). `Character.ICECLIMBERS` doesn't exist in libmelee — use `Character.POPO`. Both will hang the menu navigator forever.

17. **`--dolphin-home` tradeoffs**: On Scav (this machine), `--dolphin-home` is needed for the Slippi account. On ScavieFae's machine, it causes a non-working fullscreen mode. If menu nav gets stuck at name selection on match 2+, `--dolphin-home` may be the issue — or the fix.

18. **Watchdog for `console.step()` blocking**: If Dolphin crashes without closing the socket cleanly, `console.step()` blocks forever. The netplay runner has a watchdog thread that kills Dolphin after 15s of `step()` not returning. Without this, crashes are undetectable.

19. **CPU load causes desyncs**: Slippi's rollback is sensitive to frame timing. Background processes (browser tabs, builds) eating CPU on one side cause the other side to desync. The "Possible poor match performance detected" warning in Dolphin correlates with upcoming disconnects.

20. **Dolphin "Invalid read" modal**: Dolphin occasionally throws a modal warning dialog (`Invalid read from 0x3031000a, PC = 0x801c165c`). This is a Dolphin/emulation bug, not our code. The modal freezes the game loop, which blocks `console.step()`. The watchdog catches this and kills Dolphin after 15s. The test script moves on to the next match automatically.

## Netplay Stability Testing

Two-machine test using `test_netplay_stability.py`. Each side runs independently.

**Scav (this machine):**
```bash
.venv/bin/python test_netplay_stability.py \
  --opponent "SCAV#861" --label mattie \
  --dolphin-home ~/Library/Application\ Support/Slippi\ Dolphin \
  -d "/Applications/Slippi Dolphin.app" \
  -i ~/games/melee/"Super Smash Bros. Melee (USA) (En,Ja) (Rev 2).ciso"
```

**ScavieFae (queenmab):**
```bash
.venv/bin/python test_netplay_stability.py \
  --opponent "SCAV#382" --label scaviefae \
  -d "/Users/queenmab/Library/Application Support/Slippi Launcher/netplay/Slippi Dolphin.app" \
  -i "/Users/queenmab/claude-projects/games/melee/Super Smash Bros. Melee (USA) (En,Ja) (Rev 2).ciso"
```

**What "success" means**: A match either finishes via KO or survives to the timeout (currently 3min) without disconnecting. Both count as wins.

**Results (2026-02-02)**:
- Pre-throttle, with browser tabs: 9/10 completed, 1 disconnect
- With `input_throttle=3`, fewer tabs: 10/10 completed, 0 disconnects

## Arena Matchmaking

Two-machine matchmaking test. The server runs on one machine, both sides connect to it.

**Step 1: Start the arena server (Scav):**
```bash
.venv/bin/python -m nojohns.cli arena --port 8000
```

Find Scav's IP: `ipconfig getifaddr en0`

**Step 2: Both sides matchmake (order doesn't matter — first one waits, second triggers the match):**

**Scav (this machine):**
```bash
.venv/bin/python -m nojohns.cli matchmake random \
  --code SCAV#382 --server http://localhost:8000 \
  --dolphin-home ~/Library/Application\ Support/Slippi\ Dolphin \
  --delay 6 --throttle 3 \
  -d "/Applications/Slippi Dolphin.app" \
  -i ~/games/melee/"Super Smash Bros. Melee (USA) (En,Ja) (Rev 2).ciso"
```

**ScavieFae (queenmab):**
```bash
.venv/bin/python -m nojohns.cli matchmake random \
  --code SCAV#861 --server http://<scav-ip>:8000 \
  -d "/Users/queenmab/Library/Application Support/Slippi Launcher/netplay/Slippi Dolphin.app" \
  -i "/Users/queenmab/claude-projects/games/melee/Super Smash Bros. Melee (USA) (En,Ja) (Rev 2).ciso"
```

Note: ScavieFae does NOT use `--dolphin-home` (causes fullscreen issues on that machine — see gotcha #17). Delay and throttle default to 6 and 3.

**How it works**: The arena server is a FIFO matchmaker (FastAPI + SQLite). Each side POSTs to `/queue/join`, polls every 2s until matched, gets the opponent's Slippi connect code, launches netplay, and reports the result. The server never touches the game — it just brokers the connection. Stale queue entries expire after 5 minutes of no polling.

**ScavieFae setup** (first time only):
```bash
cd ~/claude-projects/nojohns
git pull
.venv/bin/pip install -e ".[arena]"
```

## Questions to Ask Yourself

When modifying this codebase:

1. **Does this belong in the Fighter or the Moltbot?**
   - Frame-by-frame decisions → Fighter
   - Social/meta decisions → Moltbot (skill layer)

2. **Is this fast enough for the game loop?**
   - `act()` runs 60 times per second
   - Heavy computation should happen in `setup()` or `__init__()`

3. **Does this work headless?**
   - Arena servers run without display
   - Don't assume a window exists

4. **Is this testable without Melee?**
   - Mock GameState for unit tests
   - Integration tests need the real setup

## Useful Commands

```bash
# All commands assume venv is activated or prefixed with .venv/bin/python

# Run tests (override addopts to skip missing pytest-cov)
.venv/bin/python -m pytest tests/ -v -o "addopts="

# Format code
black nojohns/ games/

# Type check
mypy nojohns/ games/

# Foundry contracts (requires forge — install via foundryup)
cd contracts && forge build
cd contracts && forge test

# List fighters
.venv/bin/python -m nojohns.cli list-fighters

# Fighter info
.venv/bin/python -m nojohns.cli info random

# Run a fight (Dolphin window will open)
.venv/bin/python -m nojohns.cli fight random do-nothing \
  -d "/Applications/Slippi Dolphin.app" \
  -i ~/games/melee/"Super Smash Bros. Melee (USA) (En,Ja) (Rev 2).ciso"

# Bo3 match
.venv/bin/python -m nojohns.cli fight random random \
  -d "/Applications/Slippi Dolphin.app" \
  -i ~/games/melee/"Super Smash Bros. Melee (USA) (En,Ja) (Rev 2).ciso" \
  --games 3

# Netplay — connect one fighter to a remote opponent via Slippi direct
# IMPORTANT: --dolphin-home is required for netplay (Slippi account lives there).
# Without it, Dolphin uses a temp dir with no account and crashes immediately.
.venv/bin/python -m nojohns.cli netplay random --code "SCAV#861" --delay 6 \
  --dolphin-home ~/Library/Application\ Support/Slippi\ Dolphin \
  -d "/Applications/Slippi Dolphin.app" \
  -i ~/games/melee/"Super Smash Bros. Melee (USA) (En,Ja) (Rev 2).ciso"

# Netplay test — two local Dolphins connected via Slippi (needs two Slippi accounts)
# Each side needs its own --home with a separate Slippi account configured
.venv/bin/python -m nojohns.cli netplay-test random random \
  --code1 "AAAA#111" --code2 "BBBB#222" \
  --home1 /path/to/dolphin-home-1 --home2 /path/to/dolphin-home-2 \
  -d "/Applications/Slippi Dolphin.app" \
  -i ~/games/melee/"Super Smash Bros. Melee (USA) (En,Ja) (Rev 2).ciso"

# Arena — start the matchmaking server
.venv/bin/python -m nojohns.cli arena --port 8000

# Matchmake — join arena queue, get matched, play netplay automatically
.venv/bin/python -m nojohns.cli matchmake random \
  --code SCAV#382 --server http://localhost:8000 \
  --dolphin-home ~/Library/Application\ Support/Slippi\ Dolphin \
  --delay 6 --throttle 3 \
  -d "/Applications/Slippi Dolphin.app" \
  -i ~/games/melee/"Super Smash Bros. Melee (USA) (En,Ja) (Rev 2).ciso"
```

## Resources

- **libmelee**: https://github.com/altf4/libmelee
- **SmashBot**: https://github.com/altf4/SmashBot
- **slippi-ai**: https://github.com/vladfi1/slippi-ai
- **Slippi**: https://slippi.gg
- **OpenClaw**: https://openclaw.ai
- **Melee frame data**: https://ikneedata.com

## Contact

Questions about this project? The maintainers are available via:
- GitHub Issues
- OpenClaw Discord (when live)
