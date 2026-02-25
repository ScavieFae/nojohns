# CLAUDE.md - No Johns Project Guide

## Reference Docs (read on demand, not loaded every turn)

| Doc | What's in it |
|-----|-------------|
| [docs/GOTCHAS.md](docs/GOTCHAS.md) | 29 hard-won lessons (venv, netplay, Phillip/TF, world model, arena) |
| [docs/COMMANDS.md](docs/COMMANDS.md) | Common tasks, useful commands, demo flow, external deps |
| [docs/HISTORY.md](docs/HISTORY.md) | Completed phases, hackathon results, onchain deployments, operator UX tiers |
| [docs/SPEC.md](docs/SPEC.md) | Full system specification |
| [docs/FIGHTERS.md](docs/FIGHTERS.md) | Fighter interface spec |
| [docs/SETUP.md](docs/SETUP.md) | Fresh machine setup guide |
| [worldmodel/RUNBOOK.md](worldmodel/RUNBOOK.md) | Training ops — recipes, experiment tracking, monitoring, SSH patterns |
| [worldmodel/docs/GPU-RUN-PLAN.md](worldmodel/docs/GPU-RUN-PLAN.md) | Plan for 15M-param GPU training run |

## What Is This Project?

No Johns is agent competition infrastructure. Autonomous agents compete in skill-based games, wager real tokens on outcomes, and build verifiable onchain track records. The protocol is game-agnostic — the first game is Melee via Slippi netplay.

**Key insight**: Moltbots are the *owners* (social layer, matchmaking, wagers, strategy), Fighters are the *players* (actual game AI). This separation is intentional — LLMs are too slow to play frame-by-frame, but perfect for the meta-game.

## Three-Agent Development

This project is developed by three Claude Code agents. **Check which agent you are before editing files.**

| Agent | Role | Owns | Branch prefix |
|-------|------|------|---------------|
| **Scav** | Python, arena, game integration | `nojohns/`, `games/`, `arena/`, `fighters/` | `scav/` |
| **ScavieFae** | Contracts, website | `contracts/`, `web/` | `scaviefae/` |
| **ScavBug** | Bugs, docs, GitHub meta-work | `docs/` (shared), issues, comments | `scavbug/` |

### How to Know Which Agent You Are

- **Scav**: Working on Python code, arena server, or game integration. Memory file is `memory/MEMORY.md`.
- **ScavieFae**: Working on Solidity contracts or the website. On a separate machine.
- **ScavBug**: User said "you're ScavBug" or context is about bugs/docs/issues. Memory file is `memory/SCAVBUG.md`.

If unclear, ask Mattie.

### Directory Ownership

| Directory | Owner | Notes |
|-----------|-------|-------|
| `nojohns/`, `games/`, `arena/`, `fighters/`, `worldmodel/` | **Scav** | Python code |
| `contracts/`, `web/` | **ScavieFae** | Solidity + frontend |
| `docs/`, `tests/`, root files | **Shared** | ScavBug can edit docs |

**Do not edit files in another agent's directories.** If you need a change in their code, open a GitHub issue or describe what you need in a PR comment.

### Branches & Coordination

Two long-lived branches: **`main`** (public-facing) and **`dev`** (integration). Agents work on prefixed branches off `dev`, PRs merge into `dev`, curated releases go `dev` → `main`.

- **Shared schema:** The `MatchResult` struct is the contract between Python and Solidity sides. Changes require coordination.
- **Contracts before mainnet deploy:** Non-negotiable review via `/review-pr`.
- **CODEOWNERS** protects `.claude/`, handoff docs, CLAUDE.md files, and contract source.

### For Scav

You own Python, arena, game integration, and world model training. Current focus: world model experiments (see World Model section below).

### For ScavieFae

Read `docs/HANDOFF-SCAVIEFAE.md` first, then `contracts/CLAUDE.md` and `web/CLAUDE.md`.

### For ScavBug

Bugs, docs, GitHub meta-work. Can read all code, edit `docs/` only. See `docs/GOTCHAS.md` for the full list. File issues for bugs, don't edit code.

## Local Dev Setup

**Use the project venv.** System Python (3.13) does NOT have libmelee. The venv does:

```bash
.venv/bin/python -m nojohns.cli fight ...
# Or: source .venv/bin/activate  (Python 3.12, libmelee + nojohns installed)
```

**First time?** Run `nojohns setup melee` to configure paths. Config: `~/.nojohns/config.toml`.

**Tests:** `.venv/bin/python -m pytest tests/ -v -o "addopts="`

## Quick Context

- **Melee**: A 2001 fighting game with a hardcore competitive scene
- **Slippi**: Modern netplay/replay system for Melee
- **libmelee**: Python API for controlling Melee via Dolphin emulator (we use vladfi1's fork v0.43.0)
- **Phillip/slippi-ai**: Neural net Melee AI by vladfi1
- **"No Johns"**: Melee slang meaning "no excuses"

## Project Structure

```
nojohns/          # Core package — fighter protocol, config, CLI
games/melee/      # Melee/Dolphin/Slippi integration (runner, netplay, menu nav)
fighters/         # Fighter implementations (smashbot/, phillip/) with fighter.toml manifests
contracts/        # Solidity contracts (Foundry) — ScavieFae owns
web/              # Website — ScavieFae owns
arena/            # Matchmaking server (FastAPI + SQLite)
worldmodel/       # Melee world model (standalone — no dependency on nojohns/games)
docs/             # Specs, guides, reference docs
```

**Dependency rule:** `games.melee` depends on `nojohns.fighter`, never the reverse. Fighters depend on `nojohns.fighter`, never on `games.melee`. `worldmodel/` is fully standalone.

## Key Abstractions

### Fighter Protocol (`nojohns/fighter.py`)

```python
class Fighter(Protocol):
    @property
    def metadata(self) -> FighterMetadata: ...
    def act(self, state: GameState) -> ControllerState: ...  # Called every frame!
```

### Match Runner (`games/melee/runner.py`)

Orchestrates Dolphin, connects fighters, runs games. `_run_game()` is the hot path.

### Config (`nojohns/config.py`)

`~/.nojohns/config.toml`. Game-specific settings under `[games.<game>]`. CLI flags override config.

## World Model

The world model learns to simulate Melee's game physics from replay data. Given a context window of recent frames plus the current frame's controller input, it predicts the resulting game state.

### Architecture (v2.2 — input-conditioned)

```
Context: frames [t-K, ..., t-1]  (state + controller per frame)
  + Conditioning: frame t's controller input  (26 floats — both players)
  → Model (FrameStackMLP or Mamba-2)
  → Prediction: frame t's state (continuous deltas + binary flags + action/jumps)
```

Separates physics simulation from decision prediction — predicts what happens when buttons are pressed, not which buttons get pressed.

### Config-Driven Experiments

Experiments live in `worldmodel/experiments/` as self-contained YAML files. `EncodingConfig` feature flags control tensor dimensions dynamically. Never hardcode tensor sizes.

### Key Files

| File | Role |
|------|------|
| `worldmodel/model/encoding.py` | `EncodingConfig`, `encode_player_frames()`, `StateEncoder` |
| `worldmodel/model/mlp.py` | `FrameStackMLP` — context window + ctrl → state prediction |
| `worldmodel/model/mamba2.py` | `Mamba2WorldModel` — SSM sequence model |
| `worldmodel/data/dataset.py` | `MeleeFrameDataset` — returns 5 tensors per sample |
| `worldmodel/training/trainer.py` | Training loop, checkpointing, shape preflight, wandb |
| `worldmodel/training/metrics.py` | Loss computation, accuracy tracking, per-action breakdown |

### Known Debt

`policy_mlp.py`, `policy_dataset.py`, and `rollout.py` have hardcoded dimension assumptions matching the v2.2 baseline. They work with default `EncodingConfig` but break with experiment flags. See RUNBOOK.md.

## Code Style

- Python 3.11+ (modern typing, tomllib is stdlib)
- Black formatting (100 char lines)
- Type hints everywhere, docstrings for public APIs
- Logging over print

## Resources

- **libmelee**: https://github.com/altf4/libmelee
- **slippi-ai**: https://github.com/vladfi1/slippi-ai
- **Slippi**: https://slippi.gg
- **OpenClaw**: https://openclaw.ai
- **Melee frame data**: https://ikneedata.com
