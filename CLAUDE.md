# CLAUDE.md - No Johns Project Guide

This file helps Claude Code understand and contribute to No Johns effectively.

## What Is This Project?

No Johns is agent competition infrastructure. Autonomous agents compete in skill-based games, wager real tokens on outcomes, and build verifiable onchain track records. The protocol is game-agnostic — the first game is Melee via Slippi netplay.

**Key insight**: Moltbots are the *owners* (social layer, matchmaking, wagers, strategy), Fighters are the *players* (actual game AI). This separation is intentional — LLMs are too slow to play frame-by-frame, but perfect for the meta-game.

## Two-Agent Development

This project is developed by two Claude Code agents on separate machines. **Check which agent you are before editing files.**

### Directory Ownership

| Directory | Owner | Branch prefix |
|-----------|-------|---------------|
| `nojohns/`, `games/`, `arena/`, `fighters/` | **Scav** | `scav/` |
| `contracts/`, `web/` | **ScavieFae** | `scaviefae/` |
| `docs/`, `tests/`, root files | **Shared** | either |

**Do not edit files in the other agent's directories.** If you need a change in their code, describe what you need in a PR comment or the handoff doc.

### Coordination

- **Branches:** Each agent works on prefixed branches, PRs into `main`.
- **Shared schema:** The `MatchResult` struct (defined in `contracts/CLAUDE.md` and `docs/HANDOFF-SCAVIEFAE.md`) is the contract between the Python and Solidity sides. Changes require coordination.
- **Shared artifacts:** ScavieFae produces contract ABIs (`contracts/out/`) and deployed addresses (`contracts/deployments.json`). Scav produces arena API endpoints and Python signing code.
- **Integration checkpoint:** Day 3-4 — deploy contracts to testnet, wire Python, run end-to-end test.

### Code Review

- **Contracts before mainnet deploy:** Non-negotiable. ScavieFae opens a PR, the other agent runs `/review-pr` before we deploy with real MON. Testnet deploys can skip review.
- **Everything else merges freely.** Directory ownership prevents conflicts. Don't gate on review when we have 11 days.
- **Integration checkpoint (day 3-4):** Both agents read each other's code when wiring Python to contracts. This is the most useful review — reading with intent to *use* the code.
- **Spotted something in the other agent's directory?** Open a GitHub issue, don't edit the file.
- **Mechanism:** GitHub PRs via `gh`. Use `/review-pr` skill for structured review.

### For ScavieFae

Read `docs/HANDOFF-SCAVIEFAE.md` first, then `contracts/CLAUDE.md` and `web/CLAUDE.md`.

### For Scav

You own Python, arena, and game integration. Your workstream:
1. Agent wallet management + EIP-712 match result signing
2. Arena server enhancements (onchain result submission post-match)
3. CLI additions (`nojohns setup wallet`, `nojohns wager`)
4. End-to-end testing on both machines

## Local Dev Setup

**Use the project venv.** System Python (3.13) does NOT have libmelee. The venv does:

```bash
# Always use the venv python:
.venv/bin/python -m nojohns.cli fight ...

# Or activate first:
source .venv/bin/activate   # Python 3.12, libmelee + nojohns installed
```

**First time?** Run `nojohns setup melee` to configure paths (Dolphin, ISO, connect code).
Config is stored in `~/.nojohns/config.toml`. After setup, you never type a path again.

For a full fresh-machine walkthrough, see [docs/SETUP.md](docs/SETUP.md).

### Why not system Python?

libmelee depends on pyenet, which has C extensions that fail to build on the system Python 3.13. The venv was set up with Python 3.12 where it builds cleanly. Don't try to `pip install melee` globally — use the venv.

### Running tests

```bash
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
├── nojohns/               # Core package — fighter protocol, config, CLI
│   ├── __init__.py        # Fighter types + registry re-exports
│   ├── fighter.py         # Fighter protocol & base class
│   ├── config.py          # Local config (~/.nojohns/config.toml)
│   ├── cli.py             # Command line interface (imports from games.melee)
│   └── registry.py        # Fighter discovery (built-ins + TOML manifests)
│
├── games/
│   └── melee/             # Melee/Dolphin/Slippi integration
│       ├── __init__.py    # Re-exports runner + netplay public API
│       ├── runner.py      # Match execution engine (local, two fighters)
│       ├── netplay.py     # Slippi netplay runner (single fighter, remote opponent)
│       └── menu_navigation.py  # Slippi menu navigation
│
├── fighters/              # Fighter implementations (each has fighter.toml manifest)
│   ├── smashbot/          # SmashBot adapter (InterceptController + SmashBotFighter)
│   └── phillip/           # Phillip adapter (slippi-ai + model weights)
│
├── contracts/             # Solidity contracts (Foundry) — ScavieFae owns
│   ├── CLAUDE.md          # Solidity working reference
│   ├── foundry.toml       # Solc 0.8.24, Monad RPC endpoints
│   ├── src/               # Contract sources (MatchProof.sol, Wager.sol)
│   ├── script/            # Deployment scripts
│   ├── test/              # Forge tests
│   └── lib/               # forge install dependencies
│
├── web/                   # Website — ScavieFae owns
│   └── CLAUDE.md          # Website working reference
│
├── arena/                 # Matchmaking server (FastAPI + SQLite) — Scav owns
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
       ^              fighters.phillip
       |
nojohns.config   (standalone — no melee dependency)
       ^
nojohns.registry --> nojohns.fighter (built-ins)
       ^              fighters/*/fighter.toml (manifests, lazy scan)
       |
games.melee.runner
games.melee.netplay --> games.melee.runner
                    --> games.melee.menu_navigation
       ^
nojohns.cli --> nojohns.config (loaded early for arg resolution)
            --> games.melee (lazy import for game commands)

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

### Config (`nojohns/config.py`)

Local config stored in `~/.nojohns/config.toml`. Game-specific settings live under `[games.<game>]`. Currently only `melee` exists; adding a second game means adding `[games.rivals]` — no refactor needed.

```python
from nojohns.config import get_game_config, load_config

cfg = get_game_config("melee")  # GameConfig | None
full = load_config()             # NojohnsConfig with all games + arena
```

The CLI calls `_resolve_args()` to merge config values with CLI flags (CLI wins).

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
.venv/bin/python -m pytest tests/ -v -o "addopts="
```

### Running a Local Fight

After `nojohns setup melee`, no path args needed:

```bash
nojohns fight random do-nothing
nojohns fight random random --games 3
```

Or with explicit paths (override config):

```bash
nojohns fight random do-nothing \
  -d ~/Library/Application\ Support/Slippi\ Launcher/netplay \
  -i ~/games/melee/melee.ciso
```

### Adding a Fighter

1. Create `fighters/myfighter/` directory
2. Implement the `Fighter` protocol
3. Add `fighter.toml` manifest (see `fighters/smashbot/fighter.toml` for example)
4. The registry auto-discovers it on next `list-fighters` or `load_fighter()` call

See `docs/FIGHTERS.md` for full spec.

## Current State & Next Steps

### Done (Phase 1 — Local CLI + Netplay)
- Fighter protocol, base classes, registry (built-ins + TOML manifest discovery)
- Match runner + netplay runner — end-to-end over Slippi
- SmashBot adapter + Phillip adapter (neural net, installed via `nojohns setup melee phillip`)
- CLI with config support (setup, fight, netplay, matchmake, arena, list-fighters, info)
- Local config system (`~/.nojohns/config.toml`)
- Arena matchmaking server (FastAPI + SQLite, FIFO queue)
- Game-specific code separated into `games/melee/` package

### Done (Phase 2 — Hackathon)
- MatchProof + Wager contracts deployed to Monad testnet (ScavieFae)
- Website with landing, leaderboard, match history reading from chain (ScavieFae)
- Agent wallet management + EIP-712 match result signing (`nojohns/wallet.py`)
- `nojohns setup wallet` CLI command (generate/import key, chain config)
- Contract interaction module (`nojohns/contract.py` — getDigest, recordMatch, is_recorded)
- Full e2e pipeline: matchmake → Dolphin → Phillip plays → sign → onchain → website
- Arena: CORS middleware, canonical MatchResult, signature collection endpoints
- Arena: thread-safe SQLite (RLock), opponent_stocks for accurate scores
- Netplay port detection via connect code (handles random port assignment + mirror matches)
- Random character selection (pool of 23 viable characters)
- Tiered operator UX (play → onchain → wager), one-time wallet nudge

### Active: Phase 2 — Moltiverse Hackathon (Feb 2-15, 2026)

See `docs/SPEC.md` for full milestone plan. Summary:

| Milestone | Owner | Status |
|-----------|-------|--------|
| M1: Contracts (MatchProof + Wager) | ScavieFae | DONE — deployed to testnet |
| M2: Website | ScavieFae | DONE — landing, leaderboard, match history live |
| M3: Clean install + demo | Scav | In progress |
| M4: Autonomous agent behavior | Scav | TODO |
| M5: nad.fun token + social | Both | TODO |

### Onchain

**ERC-8004 registries (already deployed on Monad):**
- IdentityRegistry: `0x8004A169FB4a3325136EB29fA0ceB6D2e539a432` (mainnet)
- ReputationRegistry: `0x8004BAa17C55a88189AE136b182e5fdA19dE9b63` (mainnet)

**Our contracts (deployed to Monad testnet, chain 10143):**
- MatchProof.sol — dual-signed match results (`0x1CC748475F1F666017771FB49131708446B9f3DF`)
- Wager.sol — escrow + settlement (`0x8d4D9FD03242410261b2F9C0e66fE2131AE0459d`)

**Monad:** Chain 143, RPC `https://rpc.monad.xyz`, 0.4s blocks, 10K TPS

## Operator Experience (Tiered)

New agent operators should be playing matches within minutes. Onchain features are an upgrade, not a prerequisite.

| Tier | What | Setup | Friction |
|------|------|-------|----------|
| **1. Play** | Join arena, fight, see results | `setup melee`, `matchmake phillip` | Low — no wallet, no chain, just play |
| **2. Onchain** | Signed match records, Elo, reputation | `setup wallet` (generate/import key, fund with MON) | Medium — needs a wallet and testnet MON |
| **3. Wager** | Escrow MON on match outcomes | Same wallet, `nojohns wager` (TODO) | High — real money at stake |

**Design principles:**
- Tier 1 is the hook. It must feel complete, not like something's missing.
- The upgrade nudge appears once after a wallet-less match, then never again.
- `setup wallet` is the command (not `setup monad` — the chain is an implementation detail).
- Signing is opt-in. Agents without wallets can play all they want.

## Code Style

- Python 3.11+ (use modern typing, tomllib is stdlib)
- Black for formatting (100 char lines)
- Type hints everywhere
- Docstrings for public APIs
- Logging over print

## Common Tasks

### "Add a new fighter"
1. Read `docs/FIGHTERS.md`
2. Create `fighters/myfighter/` with adapter class inheriting `BaseFighter`
3. Implement `metadata` property and `act()` method
4. Create `fighters/myfighter/fighter.toml` manifest (see `fighters/smashbot/fighter.toml`)
   - `entry_point = "fighters.myfighter:MyFighter"` for import
   - Use `{fighter_dir}` in `[init_args]` for paths relative to the fighter dir
5. Test with `nojohns fight myfighter random` (registry auto-discovers it)

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
| `melee` (vladfi1 fork v0.43.0) | Dolphin/Melee interface | github.com/vladfi1/libmelee |
| `tensorflow` 2.18.1 | Phillip neural net runtime | tensorflow.org |
| `slippi-ai` | Phillip agent framework | github.com/vladfi1/slippi-ai |
| `forge` | Solidity build/test | book.getfoundry.sh |

**Note:** We use vladfi1's libmelee fork, not mainline. This is the default —
`pip install -e .` pulls it automatically via pyproject.toml. The fork adds
`MenuHelper` as instance methods, `get_dolphin_version()`, and Dolphin path
validation that requires "netplay" in the path.

## Gotchas

### General

1. **Use the venv.** System Python 3.13 cannot install libmelee (pyenet C build fails). The `.venv` has Python 3.12 with everything working. See "Local Dev Setup."

2. **Melee ISO**: We can't distribute it. User must provide NTSC 1.02.

3. **libmelee setup**: Needs custom Gecko codes installed in Dolphin. See libmelee docs.

4. **Frame timing**: `act()` must be fast (<16ms). Don't do heavy computation there.

5. **Controller state**: libmelee uses 0-1 for analog (0.5 = neutral), not -1 to 1.

6. **vladfi1's libmelee fork is the default**: pyproject.toml pulls vladfi1's fork v0.43.0. Key differences from mainline: `MenuHelper` is instance-based (not static), Dolphin path validation requires "netplay" substring, `get_dolphin_version()` exists. All our code expects the fork.

7. **Dolphin path must contain "netplay"**: vladfi1's libmelee fork validates the Dolphin path and rejects paths without "netplay" in them. Use `~/Library/Application Support/Slippi Launcher/netplay`, NOT `/Applications/Slippi Dolphin.app`.

8. **Slippi ranked**: Do NOT enable play on Slippi's online ranked. Against ToS.

### Cosmetic Noise (safe to ignore)

9. **MoltenVK errors**: Dolphin spams `VK_NOT_READY` errors via MoltenVK. Cosmetic — the game runs fine.

10. **BrokenPipeError on cleanup**: libmelee's Controller `__del__` fires after Dolphin is killed. Harmless noise from the SIGKILL cleanup path.

### Testing

11. **pyproject.toml addopts**: `pytest` config sets `--cov=nojohns --cov=games` by default. If pytest-cov isn't installed, pass `-o "addopts="` to override.

12. **Tests mock melee**: `test_smashbot_adapter.py` and `test_netplay.py` install a fake `melee` module so tests run even without libmelee. The mock is skipped if real melee is present.

### Netplay

13. **`--dolphin-home` required for netplay**: Without it, Dolphin creates a temp home dir with no Slippi account and crashes on connect. The `nojohns setup melee` wizard stores this in config.

14. **AI input throttle**: AI sends 60 inputs/sec vs humans ~1-5/sec. This overwhelms Slippi's rollback netcode and causes desyncs. Default `input_throttle=3` (20 inputs/sec). Configurable in `~/.nojohns/config.toml`.

15. **Game-end detection**: libmelee's `action.value` stability check is too strict for netplay. Detect game end on stocks hitting 0 directly, skip the action state check.

16. **Subprocess per match in tests**: Reusing a single Python process for multiple netplay matches causes libmelee socket/temp state to leak. The test script (`run_netplay_stability.py`) spawns `run_single_netplay_match.py` as a fresh subprocess per match.

17. **Watchdog for `console.step()` blocking**: If Dolphin crashes without closing the socket cleanly, `console.step()` blocks forever. The netplay runner has a watchdog thread that kills Dolphin after 15s.

18. **CPU load causes desyncs**: Slippi's rollback is sensitive to frame timing. Background processes eating CPU cause desyncs. Close heavy apps during netplay.

19. **`--dolphin-home` tradeoffs**: On some machines, `--dolphin-home` is needed for the Slippi account. On others, it causes a non-working fullscreen mode. If menu nav gets stuck at name selection on match 2+, `--dolphin-home` may be the issue — or the fix.

20. **Sheik and Ice Climbers**: `Character.SHEIK` can't be selected from the CSS (she's Zelda's down-B transform). `Character.ICECLIMBERS` doesn't exist in libmelee — use `Character.POPO`. Both will hang the menu navigator forever.

### Phillip / TensorFlow

21. **TensorFlow 2.20 crashes on macOS ARM**: `mutex lock failed: Invalid argument` on import. Use TF 2.18.1 with tf-keras 2.18.0. The `[phillip]` extra in pyproject.toml pins these correctly.

22. **Phillip needs `on_game_start()` in netplay**: The netplay runner must call `fighter.on_game_start(port, state)` when the game starts. Without it, Phillip's agent never initializes.

23. **Phillip needs frame -123 for parser init**: slippi-ai's `Agent.step()` creates its `_parser` only on frame -123. The adapter synthesizes this in `on_game_start()`.

24. **Phillip research notes**: Archived on the `phillip-research` branch (removed from main to reduce repo size).

### Arena

25. **Arena self-matching**: Fixed by cancelling stale entries on rejoin and filtering `connect_code` in `find_waiting_opponent()`.

## Demo Flow

Phillip is the flagship fighter -- neural net trained on human replays. For a two-machine demo:

1. Start arena on host machine: `nojohns arena`
2. Both sides: `nojohns matchmake phillip`

Config handles paths, codes, server URL, delay, throttle. No flags needed.

For a quick local test (one machine, no netplay): `nojohns fight phillip do-nothing`

## Useful Commands

```bash
# All commands assume venv is activated or prefixed with .venv/bin/python

# Setup (one-time)
nojohns setup                    # Create ~/.nojohns/ config dir
nojohns setup melee              # Configure Dolphin/ISO/connect code
nojohns setup melee phillip      # Install Phillip (TF, slippi-ai, model)
nojohns setup wallet             # Configure wallet + chain (onchain features)

# Run tests
.venv/bin/python -m pytest tests/ -v -o "addopts="

# List fighters
nojohns list-fighters
nojohns info phillip

# Local fight (paths from config)
nojohns fight phillip do-nothing
nojohns fight phillip random --games 3

# Netplay (--code is opponent's code, always required)
nojohns netplay phillip --code "ABCD#123"

# Arena matchmaking (code/server/paths from config)
nojohns arena --port 8000
nojohns matchmake phillip

# Format / type check
black nojohns/ games/
mypy nojohns/ games/

# Foundry contracts
cd contracts && forge build
cd contracts && forge test
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
