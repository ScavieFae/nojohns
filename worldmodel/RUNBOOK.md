# Melee World Model — Runbook

## What Is This?

A structured-state world model that predicts next-frame Melee game state from current state + player inputs. Phase 1 is a frame-stacking MLP; Phase 2 will swap in Mamba-2.

Nobody has built this before — existing Melee AI (Phillip/slippi-ai) is model-free.

## Architecture

```
.slp files → peppi_py → numpy arrays → PyTorch tensors → MLP → next-frame predictions
```

### Per-player per-frame encoding (v2: 72 dims)

**Continuous (12 dims):**

| Field | Type | Encoding | Dims |
|-------|------|----------|------|
| percent | uint16 | float × 0.01 | 1 |
| x, y | float32 | float × 0.05 | 2 |
| shield_strength | float32 | float × 0.01 | 1 |
| speed_air_x, speed_y, speed_ground_x | float32 | float × 0.05 | 3 |
| speed_attack_x, speed_attack_y | float32 | float × 0.05 | 2 |
| state_age | float32 | float × 0.01 | 1 |
| hitlag | float32 | float × 0.1 | 1 |
| stocks | uint8 | float × 0.25 | 1 |

**Binary (3 dims):** facing, invulnerable, on_ground

**Controller (13 dims):** main_stick x/y, c_stick x/y, shoulder, 8 buttons

**Categorical embeddings (44 dims):**

| Field | Vocab | Embed dims |
|-------|-------|------------|
| action_state | 400 | 32 |
| jumps_left | 8 | 4 |
| character | 33 | 8 |

**Per-frame shared:** stage (33 → 4 dims)

**Total**: 72 per player × 2 + 4 stage = **148 dims/frame**. With K=10 context: **1,480 input dims**.

### Combat context fields (wired in v2.1)

| Field | Type | Encoding | Embed dims |
|-------|------|----------|------------|
| l_cancel | uint8 | embed (vocab=3) | 2 |
| hurtbox_state | uint8 | embed (vocab=3) | 2 |
| ground | uint16 | embed (vocab=32) | 4 |
| last_attack_landed | uint8 | embed (vocab=64) | 8 |
| combo_count | uint8 | continuous × 0.1 | 1 |

See `~/.agent/diagrams/worldmodel-encoding-v2.html` for the full reference page.

### Model (Phase 1: FrameStackMLP)

v2.2 (current — input-conditioned):
- **Trunk**: Linear(1846, 512) → ReLU → Dropout → Linear(512, 256) → ReLU → Dropout
- **Input**: K context frames (flattened) + next-frame controller input (26 floats)
- **Heads**: continuous_delta (8, MSE), velocity_delta (10, MSE), dynamics (6, MSE), binary (6, BCE), action (CE), jumps (CE)
- **Parameters**: ~1.3M
- **Loss weighting**: continuous=1.0, velocity=0.5, dynamics=0.5, binary=0.5, action=2.0, jumps=0.5

## Data Pipeline

### Data sources

All stored in `/Users/mattiefairchild/claude-projects/nojohns-training/data/`:

| Source | Files | Origin |
|--------|-------|--------|
| Phillip Matchup Compilation | 596 | Dropbox (see below) |
| Gang tournament | 3,561 | GCS: `storage.googleapis.com/slippi.appspot.com/dump/gang-replays.7z` |
| Full Bloom 5 | 4,502 | GCS: `storage.googleapis.com/slippi.appspot.com/dump/full-bloom-5-replays.7z` |
| Fight Pitt 9 | 1,150 | GCS: `storage.googleapis.com/slippi.appspot.com/dump/fight-pitt-9-replays.7z` |
| Pound 2019 | 11,019 | GCS: `storage.googleapis.com/slippi.appspot.com/dump/pound-2019-replays.7z` |
| NMA 2 | 2,817 | GCS: `storage.googleapis.com/slippi.appspot.com/dump/national-melee-arcadian-2.7z` |
| Summit 11 | 419 | GCS: `storage.googleapis.com/slippi.appspot.com/replays/bundles/Summit-11.zip` |
| Local (our matches) | 47 | ~/Slippi/ |
| **Total** | **~24K** | |

After parsing + dedup: **22,162 games** (v2.1 schema with velocity, dynamics, combat context).

### Additional sources (not yet downloaded)

- **Slippi Anonymized Ranked Dump (6 parts)**: Google Drive links, massive
  - `1pFjgh1dapX34s0T-Q1TC7JUO-qjYbQZf` through `1g8yZ-Q4ldyhDEmXLSPBoWxywJRMRVGc3`
- **Nikki's collection**: `https://1drv.ms/f/s!Ah8e99o9nnXOg_A7hLzRRKEkjifcxw` (OneDrive)
- **TestDataset-32**: `https://www.dropbox.com/scl/fi/xbja5vqqlg3m8jutyjcn7/TestDataset-32.zip?rlkey=nha6ycc6npr3wmxzickeyqpfh&st=i87xxfxk&dl=1`

### Directory layout

```
nojohns-training/           # OUTSIDE the repo — never committed
├── data/
│   ├── raw/
│   │   ├── phillip-matchup-compilation/    # Already extracted by matchup
│   │   └── tournaments/
│   │       ├── gang/
│   │       ├── fight-pitt-9/
│   │       ├── full-bloom-5/
│   │       ├── pound-2019/
│   │       ├── nma-2/
│   │       ├── summit-11/
│   │       └── *.7z, *.zip             # Original archives
│   └── parsed/
│       ├── meta.json                   # Array of game metadata dicts
│       └── games/                      # MD5-named zlib-compressed parquet
│           ├── 6827921534249607dafa57e9a67f8df4
│           └── ...
└── checkpoints/
    └── run1/
        ├── best.pt
        └── final.pt
```

### Building the parsed dataset

```bash
# Step 1: Download tournament archives
# GCS links are direct — just curl them:
curl -L -o data/raw/tournaments/gang-replays.7z \
    "https://storage.googleapis.com/slippi.appspot.com/dump/gang-replays.7z"

# Step 2: Extract archives
# Need p7zip: brew install p7zip
7z x -o"gang" gang-replays.7z
unzip -d summit-11 Summit-11.zip  # For .zip files

# Step 3: Parse .slp → compressed parquet
.venv/bin/python -m worldmodel.scripts.build_dataset \
    --input data/raw/phillip-matchup-compilation data/raw/tournaments/* ~/Slippi \
    --output data/parsed \
    --workers 6

# Incremental: re-run same command to add new replays (skips existing by MD5)
```

### Key gotchas

1. **peppi_py 0.8.6 doesn't have `RollbackMode`**. The `build_dataset.py` script calls `peppi_py.read_slippi()` directly, NOT through `slippi_db.parse_peppi.read_slippi()` which expects a newer peppi_py.

2. **peppi_py 0.8.6 doesn't have `FodPlatform`**. We can't use `slippi_db.parse_peppi.from_peppi()` either. The `build_dataset.py` script does its own conversion from peppi_py frames → PyArrow StructArrays.

3. **peppi_py returns PyArrow scalars**, not Python types. Must use `.to_pylist()` to get numpy-compatible arrays.

4. **`direction` is float (1.0/-1.0)**, not bool. **`airborne` is uint8**, not bool.

5. **`triggers` in peppi_py**: Use `pre.triggers_physical.l` and `.r` (the physical triggers), not `pre.triggers` (which is a plain FloatArray).

6. **Button bitmask**: `pre.buttons_physical` is uint16 — A=bit8, B=bit9, X=bit10, Y=bit11, Z=bit4, L=bit6, R=bit5, D_UP=bit3.

7. **Parsed files are zlib-compressed parquet** (not plain parquet). Our `load_game()` handles decompression.

8. **Stage comes from `game.start.stage`**, not from frame data. Frame `post.ground` is the ground surface ID.

## Training

### Quick start

```bash
# Toy dataset (1 game — pipeline validation):
.venv/bin/python -m worldmodel.scripts.train \
    --dataset docs/phillip-research/slippi-ai/slippi_ai/data/toy_dataset \
    --epochs 20 --device mps

# Real training (2000 games):
.venv/bin/python -m worldmodel.scripts.train \
    --dataset ~/claude-projects/nojohns-training/data/parsed \
    --max-games 2000 --epochs 10 --batch-size 512 --device mps \
    --save-dir ~/claude-projects/nojohns-training/checkpoints/run1 \
    --run-name "baseline-2k" -v

# Resume from checkpoint:
.venv/bin/python -m worldmodel.scripts.train \
    --dataset ~/claude-projects/nojohns-training/data/parsed \
    --max-games 2000 --epochs 20 --batch-size 512 --device mps \
    --save-dir ~/claude-projects/nojohns-training/checkpoints/run1 \
    --resume ~/claude-projects/nojohns-training/checkpoints/run1/best.pt \
    --run-name "baseline-2k-continued" -v

# Full dataset (all 22K games, ~6GB RAM):
.venv/bin/python -m worldmodel.scripts.train \
    --dataset ~/claude-projects/nojohns-training/data/parsed \
    --epochs 10 --batch-size 512 --device mps \
    --run-name "full-22k" -v

# Disable wandb (offline/testing):
.venv/bin/python -m worldmodel.scripts.train \
    --dataset ... --no-wandb
```

### Remote training (ScavieFae)

ScavieFae's machine: `queenmab@100.93.8.111` (Tailscale), 18GB RAM, M3 Pro.

```bash
# Start training (survives SSH disconnect):
ssh queenmab@100.93.8.111 "cd ~/claude-projects/nojohns && nohup .venv/bin/python -m worldmodel.scripts.train \
    --dataset ~/claude-projects/nojohns-training/data/parsed \
    --max-games 2000 --epochs 10 --batch-size 512 --device mps \
    --save-dir ~/claude-projects/nojohns-training/checkpoints/run1 \
    --run-name 'scaviefae-baseline' -v \
    > ~/claude-projects/nojohns-training/train.log 2>&1 &"

# Check progress (from anywhere):
ssh queenmab@100.93.8.111 "tail -20 ~/claude-projects/nojohns-training/train.log"

# Stop training:
ssh queenmab@100.93.8.111 "pkill -f worldmodel.scripts.train"
```

Key points:
- `nohup` keeps training alive after SSH disconnect
- Log file at `~/claude-projects/nojohns-training/train.log`
- Tailscale reconnects automatically when you change networks
- ScavieFae is ~20% faster per epoch than Scav (552s vs 692s)

### Training parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| lr | 1e-3 | AdamW with cosine annealing |
| weight_decay | 1e-5 | |
| batch_size | 256 | 512 works well for 2K+ games |
| context_len | 10 | K frames of history |
| dropout | 0.1 | |
| train_split | 0.9 | By game (not by frame) |
| device | auto | MPS on Mac, CUDA on Linux |

### Hyperparameter knobs

These are the things worth varying between runs:

| Knob | Current | What it controls | Try next |
|------|---------|-----------------|----------|
| batch_size | 512 | Gradient noise vs stability. Smaller = noisier but better generalization | 256 |
| context_len | 10 | How many frames of history the model sees | 5, 20 |
| hidden_dim | 512 | Trunk capacity. Wider = more expressive, more overfitting risk | 256, 1024 |
| lr | 1e-3 | Optimizer step size | 3e-4 |
| dropout | 0.1 | Regularization strength | 0.2 |
| loss weights | 1/0.5/2/0.5 | Relative importance of each head | action=5.0 |
| max_games | 2000 | Dataset size. More = better coverage of rare situations | 4000, all |

### Experiment tracking

**Wandb**: Every run logs to [wandb.ai](https://wandb.ai) project `melee-worldmodel`. Both machines are authenticated. Tracks hyperparams, loss curves, system metrics, git commit. Disable with `--no-wandb`.

**Manifest**: Every run saves `manifest.json` alongside checkpoints:
```
checkpoints/run1/
├── best.pt          # Best model (lowest val loss)
├── final.pt         # Model after last epoch
└── manifest.json    # Full record of this run
```

The manifest contains:
- `data_fingerprint` — short hash of which games were used (quick comparison)
- `game_md5s` — full list of every game hash (verify no overlap between runs)
- `config` — all hyperparams, machine name, git commit
- `all_epochs` — complete metrics history

Two runs with the same `data_fingerprint` trained on the same data. Different fingerprint = different data — check `game_md5s` to see what changed.

### Memory budget

The dataset loads all frames into contiguous tensors in RAM:

| Games | Float tensor | Int tensor | Total ~RAM |
|-------|-------------|-----------|------------|
| 500 | ~800 MB | ~160 MB | ~1.5 GB |
| 2000 | ~3.1 GB | ~625 MB | ~4.1 GB |
| 4000 | ~6.2 GB | ~1.2 GB | ~8 GB |
| 22000 | ~34 GB | ~7 GB | won't fit |

Use `--max-games` to stay within your machine's RAM. Scav (36GB) can handle ~8K games. ScavieFae (18GB) can handle ~4K.

### Performance

| Dataset size | Data load | Encoding | Per epoch (MPS) |
|-------------|-----------|----------|-----------------|
| 1 game | <1s | <1s | 0.9s |
| 50 games | 4s | <1s | ~8s |
| 500 games | 35s | 2s | ~80s |
| 2000 games | 140s | 8s | ~9min |

### Metrics to watch

- **loss/total**: Should decrease monotonically. If train drops but val rises → overfitting → stop and add more data.
- **metric/p0_action_acc**: Overall action prediction. >90% is mostly holds — easy frames inflate this.
- **metric/action_change_acc**: The hard metric — accuracy on frames where action *changes*. This is where the model actually has to predict something non-trivial.
- **metric/position_mae**: Game units, un-normalized. <1.0 is solid.
- **metric/damage_mae**: Percent, un-normalized.
- **val_loss/total**: The real measure of generalization. If this flattens while train_loss still drops, more epochs won't help — try more data.

### Results

**Toy dataset (1 game, 20 epochs, CPU):**

| Metric | Epoch 1 | Epoch 20 |
|--------|---------|----------|
| Loss | 7.50 | 0.25 |
| Action acc | 39.8% | 97.0% |
| Action-change acc | 5.9% | 67.2% |
| Position MAE | 6.57 | 1.10 |

**Real dataset v1 (2000 games, 10 epochs, ScavieFae M3 Pro):**

| Metric | Epoch 1 | Epoch 7 | Trend |
|--------|---------|---------|-------|
| Loss | 1.236 | 0.951 | ↓ still dropping |
| Action acc | 86.2% | 87.9% | ↑ slow gains |
| Action-change acc | 22.2% | 29.8% | ↑ steady improvement |
| Position MAE | 0.91 | 0.72 | ↓ getting tighter |
| Val loss | 1.068 | 0.930 | ↓ no overfitting yet |

**Real dataset v2 (2000 games, 10 epochs, Scav M3 Max):**

| Metric | Epoch 1 | Epoch 10 | v1 best |
|--------|---------|----------|---------|
| Loss | 1.169 | **0.782** | 0.951 |
| Action acc | 86.7% | **89.7%** | 87.9% |
| Action-change acc | 24.9% | **40.2%** | 29.8% |
| Position MAE | 0.92 | **0.68** | 0.72 |
| Val loss | 0.985 | **0.779** | 0.930 |

v2 adds velocity, state_age, hitlag, stocks, character/stage embeddings. The 10-point jump in action-change accuracy is primarily from velocity — knowing direction of movement makes transitions predictable.

**v2.1 validation (4000 games, 2 epochs, Scav M3 Max):**

| Metric | Epoch 1 | Epoch 2 |
|--------|---------|---------|
| Loss | 1.086 | **0.894** |
| Action acc | 87.0% | **88.4%** |
| Action-change acc | 29.2% | **36.2%** |
| Position MAE | 0.89 | **0.78** |
| Val loss | 0.886 | **0.807** |

**v2.2 input-conditioned (2000 games, 2 epochs, ScavieFae M3 Pro):**

| Metric | Epoch 1 | Epoch 2 | vs v2.1 (2K) |
|--------|---------|---------|--------------|
| Loss | 0.569 | **0.349** | -66% |
| Action acc | 94.3% | **96.1%** | +3.1pp |
| Action-change acc | 45.8% | **62.4%** | **+91% relative** |
| Position MAE | 0.91 | **0.79** | same |
| Val loss | 0.378 | **0.299** | -67% |

The v2.2 jump is entirely from input conditioning — same data, same machine, same hyperparams. The model no longer guesses player decisions; it just simulates physics. See `worldmodel/research/architecture/INPUT-CONDITIONING.md` for the full writeup.

**Phase 1 encoding experiments (2K games, 2 epochs each):**

| Experiment | action_acc | change_acc | Δ change | pos_mae | Notes |
|-----------|-----------|-----------|----------|---------|-------|
| baseline-v22 | 96.4% | 64.5% | — | 0.79 | Control |
| **1a (state_age embed)** | **97.0%** | **71.7%** | **+7.2pp** | **0.75** | **Winner — no regression** |
| 2a (press events) | 96.3% | 65.3% | +0.8pp | 0.78 | Marginal |
| 1a+2a combined | 96.7% | 70.1% | +5.6pp | 0.75 | 1a does the lifting |
| 3a (lookahead) | 94.3% | 70.8% | +6.3pp | 0.74 | Trades action_acc for change_acc |
| 3a+1a | 95.1% | 73.7% | +9.2pp | 0.73 | Best change_acc but -1.3pp action |
| 3a+1a+2a | 93.8% | 67.0%* | +2.5pp | 0.92 | *Only 1 epoch completed |

1a is the clear winner: only experiment that improves change_acc without regressing action_acc. Full analysis: `~/.agent/diagrams/worldmodel-experiment-results.html`

**MLP ceiling (22K games, 4 epochs, stopped — flattening):**

| Metric | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 |
|--------|---------|---------|---------|---------|
| change_acc | 74.8% | 76.3% | 77.2% | 77.5% |
| action_acc | 97.3% | 97.4% | 97.4% | 97.4% |
| pos_mae | 0.80 | 0.79 | 0.79 | 0.79 |

Diminishing returns: +1.5pp, +0.3pp, +0.1pp per epoch. Architecture is the bottleneck.

**Overnight runs (launched Feb 23, 2026):**
- Scav: v2.2 world model, 22K games, 10 epochs (stopped at epoch 4 — ceiling reached)
- ScavieFae: imitation policy, 4K games, 50 epochs (converged at epoch 26 — btn_acc=0.992, val plateau)

**First GPU runs — Mamba-2 on A100 (Feb 25, 2026):**

Run `mamba2-first-complete`: 2K games, Mamba-2 4.3M params, A100 40GB SXM4, batch_size=1024, `num_workers=0`.
Run `smoke-nw4-v2`: same config, `num_workers=4`. Both epochs completed.

| Metric | Epoch 1 (no workers) | Epoch 1 (nw=4) | Epoch 2 (nw=4) |
|--------|---------------------|----------------|----------------|
| loss | 0.4698 | 0.4672 | **0.2873** |
| action_acc | 95.2% | 95.2% | **96.7%** |
| change_acc | 51.9% | 52.1% | **67.2%** |
| pos_mae | 0.78 | 0.77 | **0.64** |
| val_loss | 0.3405 | 0.3420 | **0.2897** |
| val_acc | 96.0% | 95.9% | **96.5%** |
| wall time | 3679.6s (61 min) | 2738.8s (46 min) | 2737.3s (46 min) |

Key findings:
- **num_workers=4 gives 25% speedup** (61min → 46min per epoch)
- **Epoch 2 flips the story**: Mamba-2 beats MLP baseline on change_acc (67.2% vs 64.5%, same data+epochs)
- **pos_mae improves significantly** (0.77 → 0.64) — SSM state accumulates useful physics context
- val_loss < train_loss throughout → model is underfitting, more data/epochs will help

## Experiment Framework

Config-driven experiments. Each experiment is a self-contained YAML file in `worldmodel/experiments/`. No code changes to switch — just `--config`.

### How it works

```bash
# Run an experiment:
.venv/bin/python -m worldmodel.scripts.train \
    --config worldmodel/experiments/exp-2a-press-events.yaml \
    --dataset ~/claude-projects/nojohns-training/data/parsed-v2 \
    --streaming --buffer-size 500 --device mps -v
```

The experiment name (filename minus `.yaml`) auto-drives:
- **wandb run name** (unless `--run-name` overrides)
- **checkpoint directory**: `{save_dir}/{experiment_name}/`

### Feature flags (EncodingConfig)

Two flags control tensor dimensions at config time:

| Flag | Effect | Dims changed |
|------|--------|-------------|
| `state_age_as_embed: true` | Learned embedding (vocab=150, dim=8) replaces scaled float | float_per_player: 29→28, int_per_frame: 15→17, embed_dim: 60→68 |
| `press_events: true` | 16 binary rising-edge features appended to next_ctrl | ctrl_conditioning_dim: 26→42 |

When both are false (default), behavior is identical to v2.2. No existing runs break.

### Current experiments

```
worldmodel/experiments/
├── baseline-v22.yaml              # v2.2 control (all flags off)
├── exp-1a-state-age-embed.yaml    # state_age as integer embedding ← Phase 1 WINNER
├── exp-2a-press-events.yaml       # button press events in next_ctrl
├── exp-1a2a-combined.yaml         # 1a + 2a stacked
├── exp-3a-lookahead.yaml          # 1-frame lookahead (ctrl(t)+ctrl(t+1))
├── exp-3a-1a-lookahead.yaml       # 3a + 1a stacked
├── exp-3a-1a2a-lookahead.yaml     # 3a + 1a + 2a (full kitchen sink)
├── mamba2-1a.yaml                 # Mamba-2 with 1a encoding (2 layers, SSD chunk_size=10)
├── mamba2-1a-3layer.yaml          # Mamba-2 with 1a encoding (3 layers)
└── mamba2-1a-k60.yaml             # Mamba-2 K=60 (1 second context, SSD chunk_size=30)
```

Encoding experiments are 2K games, 2 epochs, batch_size 512 — quick directional runs.
Mamba-2 experiments use the same data/epoch settings for direct comparison with MLP baselines.

### Running experiments on ScavieFae

MPS doesn't share between processes. Only one training run per machine.

**DO NOT** launch via inline `nohup ... &` over SSH — if the command fails partway through, the backgrounded process still spawns. Three botched PID-capture attempts = three zombie training runs fighting over MPS. Use a script instead:

```bash
# Write a launcher script, scp it over, then run it
scp launch-exp.sh queenmab@100.93.8.111:~/launch-exp.sh
ssh queenmab@100.93.8.111 "nohup ~/launch-exp.sh > ~/exp.log 2>&1 & echo PID=\$!"
```

**Chaining experiments** (wait for one to finish, then start the next):

```bash
#!/bin/bash
# run-after-prev.sh — wait for PID, then launch
echo "[$(date)] Waiting for PID $1 to finish..."
while kill -0 $1 2>/dev/null; do sleep 60; done
echo "[$(date)] Done. Launching next experiment..."
cd ~/claude-projects/nojohns
.venv/bin/python -m worldmodel.scripts.train --config $2 \
    --dataset ~/claude-projects/nojohns-training/data/parsed-v2 \
    --streaming --buffer-size 500 --device mps -v
```

### Monitoring

**Wandb API access** — use for pulling metrics programmatically:

```python
import wandb
api = wandb.Api()
# Entity is "shinewave", project is "melee-worldmodel"
run = api.run("shinewave/melee-worldmodel/<run_id>")
history = list(run.scan_history(keys=["epoch", "loss/total", "metric/p0_action_acc", "metric/action_change_acc"]))
```

Key identifiers (don't guess these — they're not obvious):
- **Entity:** `shinewave` (check with `api.default_entity`)
- **Project:** `melee-worldmodel` (check with `api.projects("shinewave")`)
- **Run IDs:** visible in wandb URL or via `api.runs("shinewave/melee-worldmodel")`

**SSH monitoring** (ScavieFae at `queenmab@100.93.8.111`):

```bash
# Check if training is running:
ssh queenmab@100.93.8.111 "ps aux | grep worldmodel | grep -v grep"

# Check checkpoint manifests for finished runs:
ssh queenmab@100.93.8.111 'python3 << "PYEOF"
import json
name = "exp-2a-press-events"  # or exp-1a-state-age-embed
with open(f"/Users/queenmab/claude-projects/nojohns-training/checkpoints/{name}/manifest.json") as f:
    d = json.load(f)
for i, ep in enumerate(d["all_epochs"]):
    print(f"Epoch {i+1}: loss={ep[\"loss/total\"]:.4f} change_acc={ep.get(\"metric/action_change_acc\",0):.3f}")
PYEOF'

# Or just check wandb — both machines log there automatically
```

### Shape preflight check

The trainer runs a shape verification on the first sample before training starts. If the dataset's tensor shapes don't match what the EncodingConfig expects, it fails immediately with a clear error:

```
ValueError: Shape preflight FAILED — data/config mismatch:
  float_ctx: got (10, 58), expected (10, 56)
  Config: state_age_as_embed=True, press_events=False
```

This catches the most dangerous failure mode: loading a checkpoint or dataset built with one config and training with another.

### Known hardcoded-dimension debt

These files assume v2.2 baseline dimensions (continuous_dim=13, int_per_player=7, ctrl=26). They work fine with default `EncodingConfig()` but **will silently produce wrong results** if used with experiment configs:

| File | What's hardcoded | Breaks when |
|------|-----------------|-------------|
| `model/policy_mlp.py` | int_ctx column indices (0-14) | `state_age_as_embed=True` (p1 columns shift) |
| `data/policy_dataset.py` | `CTRL_OFFSET=16`, `FLOAT_PER_PLAYER=29` import | `state_age_as_embed=True` (controller at 15, not 16) |
| `scripts/rollout.py` | `float_data[t, 16:29]`, `decode_continuous(float_frame[0:13])` | `state_age_as_embed=True` (all offsets shift) |

**Why it's safe for now:** These files only run through `train_policy.py` and `rollout.py`, which construct default `EncodingConfig()` (no experiment flags). The experiment training path (`train.py` → `dataset.py` → `mlp.py`) never imports them.

**When it becomes unsafe:** If someone runs rollout on an exp-1a checkpoint, or trains a policy on data encoded with experiment flags. The shape preflight won't catch this because these files have their own data loading paths.

**Fix:** Same pattern as `mlp.py` — compute column indices from `cfg.int_per_player` and slice offsets from `cfg.continuous_dim`. Do this before running rollout or policy training on experiment checkpoints.

## Dependencies

Added `[worldmodel]` extra to `pyproject.toml`:
```
torch>=2.1, pyyaml>=6.0, tqdm>=4.60, matplotlib>=3.7
```

Already in venv: numpy, pyarrow, peppi_py, slippi_ai/slippi_db.

Install: `.venv/bin/pip install nojohns[worldmodel]` or `.venv/bin/pip install torch pyyaml tqdm matplotlib`

System: `brew install p7zip` (for extracting .7z archives)

## Encoding versions

| Version | Per-player dims | Frame dim | Input dim (K=10) | Params | Dataset |
|---------|----------------|-----------|-----------------|--------|---------|
| v1 | 56 | 112 | 1,120 | 931K | `data/parsed/` (22,049 games) |
| v2 | 72 | 148 | 1,480 | 1.1M | `data/parsed-v2/` (22,162 games) |
| v2.1 | 89 | 182 | 1,820 | 1.3M | `data/parsed-v2/` (same parquet) |
| v2.2 (MLP) | 89 | 182 | 1,846 | 1.3M | `data/parsed-v2/` (same parquet) |
| v2.2+1a (MLP) | 96 | 196 | 1,986 | 1.3M | `data/parsed-v2/` (same parquet) |
| **v3 (Mamba-2+1a, K=10)** | 96 | 196 | **seq (10,196)** | **1.15M** | `data/parsed-v2/` (same parquet) |
| **v3 (Mamba-2+1a, K=60)** | 96 | 196 | **seq (60,196)** | **1.15M** | `data/parsed-v2/` (same parquet) |

v2 adds: velocity (5), state_age, hitlag, stocks, character embed, stage embed.
v2.1 adds: l_cancel embed (2d), hurtbox_state embed (2d), ground embed (4d), last_attack_landed embed (8d), combo_count continuous. Same parquet, no re-parsing needed.
**v2.2: input-conditioned prediction.** Frame t's controller input (26 floats) is fed to the model alongside the context window. Target shifts from frame t+1 to frame t. Same parquet, same encoding — only the model architecture and data loading change. See `worldmodel/research/architecture/INPUT-CONDITIONING.md` for the full writeup.

Each version requires fresh training (input layer shape change).

### Streaming dataloader

For datasets too large for RAM, use `--streaming`:

```bash
.venv/bin/python -m worldmodel.scripts.train \
    --dataset ~/claude-projects/nojohns-training/data/parsed-v2 \
    --streaming --buffer-size 1000 \
    --epochs 10 --batch-size 512 --device mps \
    --save-dir ~/claude-projects/nojohns-training/checkpoints/v21-full \
    --run-name "v21-full-22k" -v
```

Loads games in chunks of `buffer_size`, shuffles within each chunk, frees memory between chunks. Val set stays in-memory (only 10% of games). ~30% overhead from disk I/O vs in-memory, but handles any dataset size.

## Phase 2: Mamba-2 Architecture

Implemented in `worldmodel/model/mamba2.py`. Config-driven: set `model.arch: mamba2` in experiment YAML. The data pipeline, encoding, loss functions, and training loop are identical — same forward signature as FrameStackMLP.

### Architecture (FrameStackMamba2)

```
frame_enc (B,K,frame_dim) → Linear(frame_dim, d_model) → Dropout
→ N× [RMSNorm → Mamba2Block → residual add]
→ last timestep → RMSNorm → add ctrl_proj(next_ctrl)
→ prediction heads
```

Key differences from MLP:
- **Temporal structure preserved** — frames processed as a sequence, not flattened
- **Sequential SSM scan** — hidden state carries context forward through K frames
- **Additive controller conditioning** — ctrl projected to d_model and added, not concatenated
- **Weight sharing across timesteps** — same Mamba2Block processes frame 1 and frame 10

### Mamba-2 hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| d_model | 256 | Model dimension (replaces hidden_dim/trunk_dim) |
| d_state | 64 | SSM state dimension (Mamba-2 sweet spot) |
| n_layers | 2 | Number of Mamba2Block layers |
| headdim | 64 | Dimension per SSM head |
| expand | 2 | Inner dimension = expand × d_model |
| d_conv | 4 | Depthwise conv kernel (~67ms at 60fps) |

### Parameter count comparison

| Architecture | Trunk params | Total params |
|-------------|-------------|-------------|
| MLP (hidden=512, trunk=256) | ~1.08M | ~1.31M |
| Mamba-2 (d_model=256, 2 layers) | ~850K | ~1.15M |
| Mamba-2 (d_model=256, 3 layers) | ~1.28M | ~1.58M |

### Scan modes: SSD vs Sequential

**Always use SSD for training.** Set `model.chunk_size` in the experiment YAML. Sequential scan is a for-loop — MPS launches one kernel per timestep per layer per batch, and the serial overhead compounds badly at scale. SSD uses chunked matmuls that MPS parallelizes.

| Config | ms/batch (B=512) | Relative | Use case |
|--------|-----------------|----------|----------|
| K=10 sequential | 87ms | 1.0x | Never — use SSD instead |
| **K=10 SSD/10** | **32ms** | **0.37x** | Directional runs (2K/2ep) |
| K=60 sequential | 493ms | 5.7x | Never — unusable |
| K=60 SSD/30 | 89ms | 1.03x | Overnight runs (longer context) |

**chunk_size rules:**
- Must evenly divide context_len (K=10 → chunk_size=10 or 5; K=60 → 30, 20, 15)
- Larger chunks = fewer inter-chunk passes = faster, until the quadratic attention within each chunk blows up memory
- K=10: use chunk_size=10 (single chunk, fastest)
- K=60: use chunk_size=30 (2 chunks, fastest)

**When sequential scan is still useful:**
- Correctness reference for debugging SSD changes
- Context lengths that don't evenly divide any chunk_size (SSD falls back automatically)
- The `benchmark_ssd.py` script validates SSD against sequential — run it after any changes to `mamba2.py`

### SSD troubleshooting

**Symptom: training loss is NaN or explodes after a few batches.**
- Check chunk_size divides context_len evenly. If not, SSD silently falls back to sequential (correct but slow) — but a partial mismatch could indicate config issues.
- Run `benchmark_ssd.py` to verify correctness hasn't regressed.

**Symptom: training is much slower than expected (>100ms/batch at K=10 B=512).**
- Verify `chunk_size` is set in the YAML. If missing, defaults to None → sequential scan.
- Check `ps aux | grep train` to confirm only one training process per machine (MPS can't share).
- On ScavieFae: verify the venv python is being used, not system python (system python3 is 3.9, missing torch).

**Symptom: SSD output doesn't match sequential on `benchmark_ssd.py`.**
- Two bugs we've hit before: (1) x must be pre-scaled by dt before entering SSD — the sequential scan multiplies dt inside the loop, SSD needs it upfront. (2) Inter-chunk state slice must be `[:, :-1]` (state *entering* each chunk), not `[:, 1:]` (off-by-one gives wrong initial states to each chunk).

**Symptom: MPS out of memory.**
- Reduce batch_size first (256 or 128). K=60 SSD uses more memory per batch than K=10 due to the (chunk_size × chunk_size) attention matrices.
- If that doesn't help, reduce chunk_size (more chunks = smaller attention matrices, but slower).

See `worldmodel/research/architecture/MAMBA2-EXPLAINER.md` for the full architecture explanation.

## Prediction Heads — Full Layout (as of Feb 24, 2026)

The model predicts 34 values per frame (30 float + 4 int). The rollout loop (`scripts/rollout.py`) feeds all of these back into the context window for autoregressive generation. State_age is rules-based (increment if same action, reset if changed).

**Note:** Autoregressive rollouts still drift significantly (characters float off stage, percent climbs to 500%+). This is compounding prediction error from a small model (1.15M params), not missing plumbing. The fix is scale: bigger model, more data, Mamba-2.

### Prediction head layout

| Category | Values | Head |
|---|---|---|
| Continuous deltas | percent, x, y, shield ×2 (8) | `continuous_head` |
| Velocity deltas | 5 velocities ×2 (10) | `velocity_head` |
| Binary logits | facing, invuln, on_ground ×2 (6) | `binary_head` |
| Dynamics (absolute) | hitlag, stocks, combo ×2 (6) | `dynamics_head` |
| Action | 400-class ×2 | `p0_action_head`, `p1_action_head` |
| Jumps | 7-class ×2 | `p0_jumps_head`, `p1_jumps_head` |

**Design choices:**
- Velocities as **deltas** (like position) — they change smoothly frame-to-frame
- Dynamics as **absolute values** with MSE — hitlag/stocks/combo are discrete-ish
- State_age stays **rules-based** in rollout (increment if same action, reset if changed)
- `float_tgt` expands from `(14,)` to `(30,)`: `[cont_delta(8), vel_delta(10), binary(6), dynamics(6)]`

### Loss weights

```yaml
loss_weights:
  continuous: 1.0     # position/shield deltas
  velocity: 0.5       # velocity deltas (NEW)
  dynamics: 0.5       # hitlag/stocks/combo absolute (NEW)
  binary: 0.5         # facing/invuln/on_ground
  action: 2.0         # action state CE
  jumps: 0.5          # jumps_left CE
```

### Backward compatibility

- **Old checkpoints load fine.** `strict=False` in `load_state_dict` — missing `velocity_head`/`dynamics_head` weights init randomly, existing heads load from checkpoint.
- **Old training data works.** New targets extracted from existing float columns — no re-parsing needed.
- **Rollout guarded.** `if "velocity_delta" in preds` checks protect against old models that don't output the new keys.

### Files modified

| File | Change |
|------|--------|
| `model/encoding.py` | `predicted_velocity_dim` (10), `predicted_dynamics_dim` (6) properties |
| `data/dataset.py` | `float_tgt` from (14,) to (30,) — velocity deltas + dynamics absolute |
| `model/mlp.py` | `velocity_head`, `dynamics_head` linear layers + forward outputs |
| `model/mamba2.py` | Same two heads |
| `training/metrics.py` | `LossWeights` gains velocity/dynamics; new loss terms and MAE metrics |
| `training/trainer.py` | Shape preflight updated (14→30) |
| `scripts/rollout.py` | Applies velocity deltas, dynamics, rules-based state_age in autoregressive loop |
| `scripts/generate_demo.py` | `_build_predicted_player` outputs velocity/hitlag/combo; `strict=False` for old checkpoints |

### Validation (Feb 24)

1. **Quick train** (10 games, 1 epoch): `loss/velocity=0.132`, `loss/dynamics=0.162` — computes and backprops ✓
2. **Teacher-forced demo** (old checkpoint): JSON includes velocity/hitlag/combo fields, values are random (untrained heads) ✓
3. **Autoregressive demo** (old checkpoint): 20/20 unique velocity values in first 20 frames — mechanism works, values update frame-to-frame ✓

## Future Direction: Paired (State, Video) Dataset

A potential research path: create paired datasets mapping game state tensors to rendered pixel data. Melee's rendering pipeline works like this:

```
Game state (~280 bytes) → Dolphin rendering engine + game ROM assets → Raw pixels (~900 KB/frame)
                                                                       → H.264 compressed (~3 KB/frame)
```

The game state is the "sheet music" — positions, velocities, action states, controller inputs. The renderer is the "performer" — it reads those ~280 bytes and produces a 720p frame using the game's 3D models, animations, textures, and camera logic. The mapping is deterministic: same state → same pixels, always.

### Why this matters

- **Visual world model**: Train a model that predicts future *pixels* from current state, not just future state. This is the "Sora for Melee" direction.
- **State → video codec**: If you can predict the next frame's pixels from state, you don't need to store/stream video. Just send the state tensor (~280 bytes vs ~3 KB compressed video per frame).
- **Inverse rendering**: Learn what visual features correspond to which state variables. Could enable pixel-input agents (no libmelee needed).

### How to build it

1. **Headless Dolphin replay** with frame dumping enabled (`--dump-frames` flag)
2. For each .slp replay, Dolphin replays the match and dumps raw BMP frames at 60fps
3. **ffmpeg** encodes raw frames → H.264 (or per-frame PNG for lossless)
4. **Pair** frame N's state tensor with frame N's pixel data

### Scale estimate (22K games)

| Item | Per game (avg 4500 frames) | 22K games |
|------|---------------------------|-----------|
| Raw BMP frames | ~4 GB | ~88 TB (don't store) |
| H.264 video | ~14 MB | ~300 GB |
| PNG per-frame | ~50 MB | ~1.1 TB |
| State tensors | ~500 KB | ~11 GB (already have) |
| Wall time (1 Dolphin) | ~75s (real-time playback) | ~19 days |
| Wall time (6 parallel) | ~75s | ~3.2 days |

### Open questions

- Dolphin headless frame dump quality (resolution, color space)
- Camera angle consistency across replays
- Whether H.264 compression artifacts matter for training
- Storage: even H.264 is 300 GB for the full set

This is a Phase 3+ direction. Not blocking current work.

## GPU Training via Modal

**Use Modal for all GPU training.** It spins up an A100 on demand, runs your code, streams logs to your terminal, and shuts down automatically. No SSH, no pods, no environment management. Pay-per-second.

### Prerequisites (one-time)

1. **Install Modal**: `.venv/bin/pip install modal`
2. **Auth**: `.venv/bin/modal setup` (opens browser, creates token)
3. **Volume**: `.venv/bin/modal volume create melee-training-data`
4. **wandb secret**: `.venv/bin/modal secret create wandb-key WANDB_API_KEY=<key>`

### Upload training data (one-time)

Individual files are slow — tar first, upload the single blob. The `pre_encode` function auto-extracts.

```bash
# Tar locally (fast)
cd ~/claude-projects/nojohns-training/data && tar cf /tmp/parsed-v2.tar parsed-v2/

# Upload to Modal volume (3.4GB, ~5 min)
.venv/bin/modal volume put melee-training-data /tmp/parsed-v2.tar /parsed-v2.tar

# Verify
.venv/bin/modal run worldmodel/scripts/modal_train.py::check_volume
```

### Pre-encode on Modal

Encode raw parquet → `.pt` directly on Modal's CPUs. No local encoding or 7GB upload needed.

```bash
# Encode all games (reads tar, encodes, writes .pt to volume)
.venv/bin/modal run worldmodel/scripts/modal_train.py::pre_encode

# Subset for testing
.venv/bin/modal run worldmodel/scripts/modal_train.py::pre_encode \
    --max-games 2000 --output /encoded-2k.pt

# Verify the .pt landed
.venv/bin/modal run worldmodel/scripts/modal_train.py::check_volume
```

The tar is auto-extracted on first `pre_encode` run and committed to the volume. Subsequent runs skip extraction.

### Launch training

```bash
# Default config (Mamba-2 medium, A100, 10 epochs)
.venv/bin/modal run worldmodel/scripts/modal_train.py::train

# Custom config
.venv/bin/modal run worldmodel/scripts/modal_train.py::train \
    --config worldmodel/experiments/mamba2-medium-gpu.yaml

# Override epochs
.venv/bin/modal run worldmodel/scripts/modal_train.py::train --epochs 5

# Custom run name for wandb
.venv/bin/modal run worldmodel/scripts/modal_train.py::train \
    --run-name "my-experiment"

# Detached mode — returns immediately, check wandb for progress
.venv/bin/modal run --detach worldmodel/scripts/modal_train.py::train \
    --run-name "overnight-run" --epochs 20
```

Logs stream live to your terminal (unless `--detach`). wandb tracking at https://wandb.ai/shinewave/melee-worldmodel.

### Sweep (parallel runs)

Launch multiple A100 runs simultaneously for hyperparameter sweeps:

```bash
.venv/bin/modal run worldmodel/scripts/modal_train.py::sweep \
    --names "sweep-lr3e4,sweep-lr1e3,sweep-lr3e3" --epochs 10
```

Each run name gets its own A100 and wandb run. The sweep entrypoint waits for all runs to complete.

### Check training status

If you launched with `--detach` or lost the terminal:
- **wandb**: Check the project dashboard for live loss curves
- **Modal dashboard**: https://modal.com/apps/scaviefae/main — shows running functions, logs, GPU usage

### Download checkpoints

```bash
# List what's on the volume
.venv/bin/modal run worldmodel/scripts/modal_train.py::check_volume

# Pull files from volume to local
.venv/bin/modal volume get melee-training-data /checkpoints ./downloaded-checkpoints
```

### Cost

- A100 40GB: ~$2.78/hr on Modal
- Observed: 1 epoch on 2K games = 46 min (num_workers=4) or 61 min (no workers). 2 epochs = 91 min → ~$4.21.
- You only pay while the function is running — no idle costs
- Check spend at https://modal.com/settings/billing

### How it works (for future Claudes)

`worldmodel/scripts/modal_train.py` is the launcher. It defines:
- **Image**: debian_slim + PyTorch CUDA + pyarrow/pyyaml/wandb + local `worldmodel/` code baked in
- **Volume**: `melee-training-data` persists training data + checkpoints across runs
- **Secret**: `wandb-key` provides WANDB_API_KEY to the container
- **`pre_encode()`**: CPU-only function — reads raw parquet tar from volume, encodes to `.pt`, writes back to volume
- **`train()`**: GPU function — loads pre-encoded `.pt`, builds model, trains with `num_workers=4` for DataLoader parallelism
- **`sweep()`**: Local entrypoint — calls `train.spawn()` to launch parallel A100 runs
- **`check_volume()`**: Lists volume contents (replaces old `upload_data`/`download_checkpoints`)

The local code is baked into the container image via `add_local_dir()`. This means code changes are automatically picked up on every `modal run` — no manual sync needed. Data lives on the persistent volume.

### Troubleshooting

**"Volume not found"**
```bash
.venv/bin/modal volume create melee-training-data
```

**"Token missing" / auth errors**
```bash
.venv/bin/modal setup   # Re-authenticate in browser
```

**"Secret not found"**
```bash
.venv/bin/modal secret create wandb-key WANDB_API_KEY=<your-key>
```
To train without wandb, edit `modal_train.py` and remove the `secrets=` line, or add `--no-wandb` to the subprocess command.

**Training crashes with import errors**
The code is baked from your local `worldmodel/` directory. Check that the code works locally first:
```bash
.venv/bin/python -c "from worldmodel.model.mamba2 import FrameStackMamba2; print('OK')"
```

**"No data" errors**
Check the volume has data:
```bash
.venv/bin/modal run worldmodel/scripts/modal_train.py::check_volume
```
If empty, re-upload the tar (see "Upload training data" above). Then run `pre_encode` to generate the `.pt` file.

**CUDA out of memory**
Reduce batch_size in the experiment YAML. A100 40GB should handle batch_size=1024 for the 4.3M param model easily. If you're running a larger model, try 512 or 256.

**"module 'modal' has no attribute X"**
Modal's API changes between versions. Check `.venv/bin/pip show modal` for version. Current working version: 1.3.4. Key changes from older versions:
- `modal.Mount` → `Image.add_local_dir()` (mounts removed)
- `Secret.from_name(required=False)` → just `Secret.from_name()` (no required param)
- `torch.cuda.get_device_properties(0).total_mem` → attribute name varies by PyTorch version; use `getattr()` fallback

### Why Modal, not RunPod

We burned 8+ hours on RunPod (Feb 25, 2026). Core problems:
1. **SSH proxy forces PTY** — `ssh.runpod.io` breaks rsync/scp/piped commands. Non-interactive SSH is impossible through their proxy.
2. **Direct TCP ports unreliable** — connection-refused on public IP:port despite pod showing "running."
3. **Pip packages ephemeral** — installed to container disk, lost on pod restart. Only `/workspace` volume persists.
4. **runpodctl** uses croc P2P transfer with interactive code phrases — impossible for an agent to automate.
5. **SSH key auth flaky** — both ed25519 and RSA keys registered correctly, both rejected by proxy. No clear error.

RunPod is built for humans in Jupyter notebooks, not agents driving SSH. Modal eliminates the entire category of remote-machine problems.

## Run Cards

Every significant training run gets a **Run Card** — a pre-flight document reviewed by Scav, Mattie, and ScavieFae before launch. Never start a large run without one.

Use `/run-card` to generate one. The skill reads the config, calculates batch counts and timing, and produces the card for review.

### Template

```markdown
# Run Card: {run_name}

## Goal
What are we trying to learn? One sentence.

## Target Metrics
| Metric | Baseline (best prior) | Target | "Something is wrong" |
|--------|----------------------|--------|---------------------|
| val_change_acc | 68.3% (K=60 2K) | >72% | <60% after epoch 1 |
| val_pos_mae | 0.55 (K=60 2K) | <0.50 | >1.0 |
| val_loss/total | 0.289 (K=10 2K) | <0.25 | not decreasing after 5% |

## Data
- **Encoded file**: /encoded-22k.pt
- **Games**: 22,000
- **Total frames**: 206,754,081
- **Train examples**: 185,648,963 (90%)
- **Val examples**: 20,885,118 (10%)
- **Data fingerprint**: {hash}

## Model
- **Architecture**: Mamba-2 (FrameStackMamba2)
- **Config**: worldmodel/experiments/mamba2-medium-gpu.yaml
- **Parameters**: 4,282,386
- **Key hyperparams**: d_model=384, n_layers=4, d_state=64, K=10, chunk_size=10

## Training
- **Epochs**: 3
- **Batch size**: 1024
- **Learning rate**: 0.0005
- **Scheduled sampling**: rate=0.30, noise=0.10, anneal=3ep, corrupt=3 frames
- **Optimizer**: AdamW (weight_decay=1e-5), cosine LR schedule

## Infrastructure
- **GPU**: A100-SXM4-40GB (Modal)
- **System RAM**: ~128GB
- **num_workers**: 4 (persistent, prefetch=4)
- **wandb**: shinewave/melee-worldmodel, run name: {run_name}

## Logging
- **Batches per epoch**: {train_examples // batch_size}
- **log_interval**: {value} (every {human_time})
- **wandb**: logs every batch (loss scalars)
- **Stdout**: loss + pct every log_interval batches

## Timing
- **Estimated batch speed**: ~{ms}ms/batch (based on {prior_run})
- **Estimated epoch time**: ~{hours}h
- **Estimated total time**: ~{hours}h
- **Timeout**: {seconds}s ({human_readable})
- **Estimated cost**: ~${cost} (at $2.78/hr)

## Escape Hatches
- **Kill if**: loss not decreasing after 10% of epoch 1, or loss explodes (>10.0)
- **Resume with**: `--resume {checkpoint_path}`
- **Fallback**: {what to do if this run fails}

## Prior Runs
| Run | Data | Epochs | change_acc | pos_mae | Notes |
|-----|------|--------|------------|---------|-------|
| ... | ... | ... | ... | ... | ... |

## Diff from Last Run
What's different about THIS run vs the most comparable prior run?

## Launch Command
\`\`\`bash
.venv/bin/modal run --detach worldmodel/scripts/modal_train.py::train \
  --encoded-file /encoded-22k.pt --epochs 3 --run-name {run_name}
\`\`\`

## Sign-off
- [ ] Scav reviewed
- [ ] Mattie reviewed
- [ ] ScavieFae reviewed
```

### What Counts as "Significant"

Use your judgment, but generally:
- Any run expected to cost >$5 or take >2 hours
- First run with a new model architecture or config
- Runs on new/larger datasets
- Runs with new training features (SS, new encoding, etc.)

Quick smoke tests (50 games, 1 epoch, <$1) don't need a card.

### Log Interval Guidelines

The default `log_interval` (10x per epoch) is fine for small runs. For large datasets:

| Train examples | Batches/epoch (bs=1024) | Default interval | Time between logs* | Recommendation |
|---------------|------------------------|-----------------|-------------------|----------------|
| 500K | 488 | 48 | ~2 min | Default is fine |
| 5M | 4,882 | 488 | ~3 min | Default is fine |
| 50M | 48,828 | 4,882 | ~30 min | Set log_interval: 1000 |
| 185M | 181,298 | 18,129 | ~90 min | **Set log_interval: 1000** |

*Assuming ~0.06s/batch on A100 with num_workers=4.

Set `log_interval` in the experiment YAML under `training:`:
```yaml
training:
  log_interval: 1000  # Log every 1000 batches (~60s on A100)
```

## Autonomous World Model (Solana hackathon — deadline Feb 27)

Research direction: decompose the monolithic model into subsystem models (movement, hit detection, damage/knockback, action transitions, shield/grab) for onchain deployment via MagicBlock ephemeral rollups + BOLT ECS.

Details: `~/claude-projects/rnd-2026/projects/autonomous-world-model/README.md`

Key insight: the monolithic model directly answers the decomposition question. Train monolithic v2, measure per-subsystem accuracy, then train standalone subsystem models and compare. Subsystems that match the monolithic ceiling decompose well → candidates for onchain execution.
