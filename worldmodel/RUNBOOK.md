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

### Fields in parquet but not yet in model

| Field | Type | Values | Why |
|-------|------|--------|-----|
| l_cancel | uint8 | 0=N/A, 1=success, 2=miss | L-cancelled aerials have half landing lag |
| hurtbox_state | uint8 | 0=vulnerable, 1=invuln, 2=intangible | ~10% of frames have invulnerability |
| ground | uint16 | surface IDs, 65535=airborne | Which platform — needed for platform drops |
| last_attack_landed | uint8 | ~60 attack IDs | What move connected — helps predict DI |
| combo_count | uint8 | 0-10+ | Combo context |

See `~/.agent/diagrams/worldmodel-encoding-v2.html` for the full reference page.

### Model (Phase 1: FrameStackMLP)

- **Trunk**: Linear(1480, 512) → ReLU → Dropout → Linear(512, 256) → ReLU → Dropout
- **Heads**: continuous_delta (MSE), binary (BCE), action (CE), jumps (CE)
- **Parameters**: 1,116,138 (~1.1M)
- **Loss weighting**: continuous=1.0, binary=0.5, action=2.0, jumps=0.5

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

**Real dataset (2000 games, 10 epochs, ScavieFae M3 Pro):**

| Metric | Epoch 1 | Epoch 7 | Trend |
|--------|---------|---------|-------|
| Loss | 1.236 | 0.951 | ↓ still dropping |
| Action acc | 86.2% | 87.9% | ↑ slow gains |
| Action-change acc | 22.2% | 29.8% | ↑ steady improvement |
| Position MAE | 0.91 | 0.72 | ↓ getting tighter |
| Val loss | 1.068 | 0.930 | ↓ no overfitting yet |

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

v2 adds: velocity (5), state_age, hitlag, stocks, character embed, stage embed. Parquet also stores l_cancel, hurtbox_state, ground, last_attack_landed, combo_count for future model use.

v1 → v2 requires fresh training (input layer shape change). v1 results serve as baseline.

## Phase 2 Prep

Swapping MLP → Mamba-2 means changing only `worldmodel/model/mlp.py`. The data pipeline, encoding, loss functions, and training loop stay identical. The key change: Mamba takes `(B, T, frame_dim)` sequences instead of `(B, K*frame_dim)` flat vectors.

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

## Autonomous World Model (Solana hackathon — deadline Feb 27)

Research direction: decompose the monolithic model into subsystem models (movement, hit detection, damage/knockback, action transitions, shield/grab) for onchain deployment via MagicBlock ephemeral rollups + BOLT ECS.

Details: `~/claude-projects/rnd-2026/projects/autonomous-world-model/README.md`

Key insight: the monolithic model directly answers the decomposition question. Train monolithic v2, measure per-subsystem accuracy, then train standalone subsystem models and compare. Subsystems that match the monolithic ceiling decompose well → candidates for onchain execution.
