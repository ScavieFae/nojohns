# Commands & Common Tasks

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

### "Train the world model"
1. Parsed replay data lives in `~/claude-projects/nojohns-training/data/parsed-v2`
2. Config-driven: `--config worldmodel/experiments/<name>.yaml` sets all hyperparams
3. Checkpoints save to `~/claude-projects/nojohns-training/checkpoints/<experiment-name>/`
4. Use `--streaming --buffer-size 500` for large datasets (>1K games)
5. See `worldmodel/RUNBOOK.md` for full recipes, monitoring, and SSH launch patterns

### "Run a new experiment"
1. Copy an existing YAML from `worldmodel/experiments/`
2. Change the encoding flags and/or training hyperparams
3. Run with `--config worldmodel/experiments/your-experiment.yaml`
4. Experiment name (filename minus `.yaml`) auto-drives wandb run name + save dir
5. Shape preflight in trainer catches config/data mismatches before training starts

### "Debug fighter issues"
1. Run with `--headless=false` to see what's happening
2. Add logging in fighter's `act()` method
3. Check GameState values match expectations
4. libmelee docs: https://libmelee.readthedocs.io/

## Demo Flow

Phillip is the flagship fighter — neural net trained on human replays. For a two-machine demo:

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
nojohns matchmake phillip          # After match, prompts for optional wager

# Wager commands (testnet only for now)
nojohns wager propose 0.01         # Propose open wager (0.01 MON)
nojohns wager propose 0.01 -o 0x.. # Propose wager to specific opponent
nojohns wager accept 0             # Accept wager ID 0
nojohns wager settle 0 <match_id>  # Settle using MatchProof record
nojohns wager cancel 0             # Cancel and refund (before accept)
nojohns wager status 0             # Check wager details
nojohns wager list                 # List your wagers

# World model training
.venv/bin/python -m worldmodel.scripts.train \
  --config worldmodel/experiments/baseline-v22.yaml \
  --dataset ~/claude-projects/nojohns-training/data/parsed-v2 \
  --streaming --buffer-size 500 --device mps -v
# Experiment configs: worldmodel/experiments/*.yaml
# Checkpoints: ~/claude-projects/nojohns-training/checkpoints/<experiment>/

# Format / type check
black nojohns/ games/
mypy nojohns/ games/

# Foundry contracts
cd contracts && forge build
cd contracts && forge test
```

## External Dependencies

| Package | Purpose | Docs |
|---------|---------|------|
| `melee` (vladfi1 fork v0.43.0) | Dolphin/Melee interface | github.com/vladfi1/libmelee |
| `tensorflow` 2.18.1 | Phillip neural net runtime | tensorflow.org |
| `slippi-ai` | Phillip agent framework | github.com/vladfi1/slippi-ai |
| `torch` | World model training | pytorch.org |
| `wandb` | Experiment tracking | wandb.ai |
| `forge` | Solidity build/test | book.getfoundry.sh |

**Note:** We use vladfi1's libmelee fork, not mainline. This is the default —
`pip install -e .` pulls it automatically via pyproject.toml. The fork adds
`MenuHelper` as instance methods, `get_dolphin_version()`, and Dolphin path
validation that requires "netplay" in the path.
