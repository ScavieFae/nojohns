# CLAUDE.md - No Johns Project Guide

This file helps Claude Code understand and contribute to No Johns effectively.

## What Is This Project?

No Johns enables Moltbot-to-Moltbot competition in Super Smash Bros. Melee using pluggable AI fighters.

**Key insight**: Moltbots are the *owners* (social layer, matchmaking, commentary), Fighters are the *players* (actual game AI). This separation is intentional - LLMs are too slow to play frame-by-frame, but perfect for the meta-game.

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
│   └── API.md             # REST API (TODO)
│
├── nojohns/               # Core Python package
│   ├── __init__.py        # Public exports
│   ├── fighter.py         # Fighter protocol & base class
│   ├── runner.py          # Match execution engine
│   ├── cli.py             # Command line interface
│   └── registry.py        # Fighter discovery (TODO)
│
├── fighters/              # Fighter implementations
│   ├── smashbot/          # SmashBot adapter (TODO)
│   └── phillip/           # Phillip adapter (TODO)
│
├── arena/                 # Arena server (TODO)
│
└── skill/                 # OpenClaw skill
    └── SKILL.md           # Skill documentation
```

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

### Match Runner (`nojohns/runner.py`)

Orchestrates Dolphin, connects fighters, runs games:

```python
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
pip install -e ".[dev]"
pytest
```

### Running a Local Fight
```bash
# With test fighters
nojohns fight random random -d /path/to/dolphin -i /path/to/melee.iso

# Headless (faster)
nojohns fight random random -d /path/to/dolphin -i /path/to/melee.iso --headless
```

### Adding a Fighter

1. Create `fighters/myfighter/` directory
2. Implement the `Fighter` protocol
3. Add `fighter.yaml` manifest
4. Register in `nojohns/registry.py`

See `docs/FIGHTERS.md` for full spec.

## Current State & Next Steps

### Done ✅
- Fighter protocol defined
- Base classes implemented  
- Match runner skeleton
- CLI skeleton
- Documentation structure

### Phase 1 TODO (Local CLI)
- [ ] Test with actual libmelee (need Dolphin + ISO)
- [ ] Fix menu navigation in runner
- [ ] SmashBot adapter (wrap existing SmashBot)
- [ ] Replay saving

### Phase 2 TODO (Moltbot Integration)
- [ ] OpenClaw skill implementation
- [ ] Fighter config via chat
- [ ] Result formatting for chat

### Phase 3 TODO (Multi-Moltbot)
- [ ] Arena server
- [ ] Matchmaking API
- [ ] ELO system

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
1. Look at `nojohns/runner.py`
2. The `_run_game()` method is the hot path
3. Menu handling is in `_handle_menu()`
4. Test changes require Dolphin + ISO

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

## Gotchas

1. **Melee ISO**: We can't distribute it. User must provide NTSC 1.02.

2. **libmelee setup**: Needs custom Gecko codes installed in Dolphin. See libmelee docs.

3. **Frame timing**: `act()` must be fast (<16ms). Don't do heavy computation there.

4. **Controller state**: libmelee uses 0-1 for analog (0.5 = neutral), not -1 to 1.

5. **Slippi ranked**: Do NOT enable play on Slippi's online ranked. Against ToS.

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
# Install in dev mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black nojohns/

# Type check
mypy nojohns/

# List fighters
nojohns list-fighters

# Fighter info
nojohns info smashbot

# Run a fight
nojohns fight random random -d $DOLPHIN -i $ISO --games 3
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
