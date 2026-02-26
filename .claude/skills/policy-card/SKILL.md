---
name: policy-card
description: Generate a Run Card for fighter policy (imitation learning) training. Use before training a new Phillip on significant data (>1K games). The card captures data scale, model config, timing estimates, and baselines. Use when user says "policy card", "train a new phillip", "fighter training card", or similar.
---

# Policy Training Run Card

Generate a pre-flight card for imitation learning policy training. Policy models learn to predict controller inputs from game state — they're the "brains" that drive fighters inside the world model.

**Key difference from world model cards:** Policy training runs locally (MPS/CPU), not on Modal. No cost concerns, but data loading time is significant for large datasets (22K games = ~15 min to parse and encode).

**Rule: Always pair policy scale with world model scale.** A policy trained on 50 games produces near-random inputs relative to a world model trained on 22K games. The policy's training data should match or exceed the world model's.

## What the User Provides

- Number of games (or "match the world model")
- Player to train (0, 1, or both)
- Context: "for the demo", "for the hackathon", etc.
- Optional: config YAML, epoch override, device preference

## What You Do

### 1. Gather Facts

Read/check:
- **World model checkpoint** being used — what data scale was it trained on?
- **Prior policy runs** on wandb (tag: "policy"):
```bash
.venv/bin/python -c "
import wandb; api = wandb.Api()
for run in list(api.runs('shinewave/melee-worldmodel', filters={'tags': 'policy'}, order='-created_at'))[:5]:
    s = run.summary
    games = run.config.get('data', {}).get('num_games_loaded', '?')
    player = run.config.get('data', {}).get('predict_player', '?')
    stick = s.get('val_metric/stick_mae', s.get('metric/stick_mae', '?'))
    btn_p = s.get('val_metric/button_pressed_acc', s.get('metric/button_pressed_acc', '?'))
    print(f'{run.name}: {games}g p{player} | stick_mae={stick} btn_pressed_acc={btn_p}')
"
```
- **Available data**: Check `~/claude-projects/nojohns-training/data/parsed-v2/games/` game count
- **Existing policy checkpoints**: `worldmodel/checkpoints/policy-*/manifest.json`

### 2. Calculate Derived Values

Use these observed constants:

| Hardware | Parse+encode speed | Batch speed (bs=512) | Batch speed (bs=1024) |
|----------|-------------------|---------------------|----------------------|
| MPS M3 Pro | ~22K games in ~15 min | ~0.02s/batch | ~0.04s/batch |
| CPU | ~22K games in ~25 min | ~0.08s/batch | ~0.15s/batch |

Data estimates:
- **~9,400 frames/game** average (from 22K dataset: 206M frames / 22K games)
- **Per-player examples ≈ total_frames × 0.9** (90% train split, minus K context per game)

Calculate:
- `total_frames = num_games * 9400`
- `train_examples = total_frames * 0.9`
- `batches_per_epoch = train_examples // batch_size`
- `epoch_time = batches_per_epoch * batch_speed`
- `data_load_time = num_games * 0.04s` (parse) + `num_games * 0.002s` (encode)
- `recommended_epochs`: 5-8 for >10K games, 10-15 for 1-5K, 25-50 for <1K

**Flag if:**
- Policy data < world model data (the whole point is distribution match)
- >50 epochs requested on >5K games (will overfit — convergence happens by epoch 8-10)
- Using CPU for >5K games (suggest MPS if available)
- No wandb (training is fast but tracking matters for comparison)

### 3. Write the Card

Save to `worldmodel/docs/run-cards/{run_name}.md`:

```markdown
# Policy Run Card: {run_name}

**Created**: {date}
**Status**: PENDING REVIEW

## Goal

{One sentence: why are we training this policy? What world model will it pair with?}

## Distribution Match

| Component | Data scale | Notes |
|-----------|-----------|-------|
| World model | {games} games | checkpoint: {name} |
| This policy | {games} games | {same / different / subset} |

{If mismatch, explain why and flag the risk.}

## Data

| Field | Value |
|-------|-------|
| Dataset path | `{path}` |
| Games | {n:,} |
| Est. total frames | ~{n:,} |
| Train examples | ~{n:,} (90%) |
| Val examples | ~{n:,} (10%) |
| Predict player | {0 / 1 / both} |

## Model

| Field | Value |
|-------|-------|
| Architecture | PolicyMLP |
| Parameters | ~{n:,} |
| Context len (K) | {n} |
| Hidden dim | {n} |
| Trunk dim | {n} |
| Dropout | {n} |

## Training

| Field | Value |
|-------|-------|
| Epochs | {n} |
| Batch size | {n} |
| Learning rate | {lr} |
| Weight decay | {wd} |
| Optimizer | AdamW + cosine LR |
| Device | {mps / cpu / cuda} |

## Timing

| Field | Value |
|-------|-------|
| Est. data load | ~{min} min (parse + encode) |
| Est. batches/epoch | ~{n:,} |
| Est. epoch time | ~{min} min |
| Est. total training | ~{h}h {m}m |
| Est. total wall time | ~{h}h {m}m (incl. data load) |

## Prior Baselines

| Run | Games | Player | stick_mae | btn_pressed_acc | val_loss |
|-----|-------|--------|-----------|-----------------|----------|
| {name} | {n} | p{n} | {val} | {val} | {val} |

## Target Metrics

| Metric | Baseline | Target | Notes |
|--------|----------|--------|-------|
| val_stick_mae | {best} | <{target} | Lower = more precise stick control |
| val_button_pressed_acc | {best}% | >{target}% | Accuracy when buttons are active (the hard signal) |
| val_loss/total | {best} | <{target} | Should improve with data scale |

## Launch Commands

```bash
# Player 0
.venv/bin/python -m worldmodel.scripts.train_policy \
    --dataset ~/claude-projects/nojohns-training/data/parsed-v2 \
    --max-games {n} --predict-player 0 \
    --epochs {n} --batch-size {bs} --device {dev} \
    --save-dir worldmodel/checkpoints/{save_dir_p0} \
    --run-name "{run_name}-p0" -v

# Player 1
.venv/bin/python -m worldmodel.scripts.train_policy \
    --dataset ~/claude-projects/nojohns-training/data/parsed-v2 \
    --max-games {n} --predict-player 1 \
    --epochs {n} --batch-size {bs} --device {dev} \
    --save-dir worldmodel/checkpoints/{save_dir_p1} \
    --run-name "{run_name}-p1" -v
```

{If training both players, note they can run in parallel (separate processes).}

## After Training

1. Verify both checkpoints exist and manifests have results
2. Wire into play_match.py demo:
```bash
.venv/bin/python -m worldmodel.scripts.play_match \
    --world-model {world_model_checkpoint} \
    --seed-game {seed_game_path} \
    --p0 policy:{p0_checkpoint} --p1 policy:{p1_checkpoint} \
    --max-frames 20000 --output demo-agents.json -v
```
3. Compare agent behavior in viewer against autoregressive (replay inputs) baseline
4. If agents look random/degenerate, check distribution match and convergence curves in wandb

## Sign-off

- [ ] Scav reviewed
- [ ] Mattie reviewed
```

### 4. Present to User

After writing the card, print a compact summary:

```
Policy Card: {name}
  Goal: {one line}
  Data: {games} games (~{frames} frames), player {n}
  Model: PolicyMLP {params} params, K={context_len}
  Training: {epochs} epochs, bs={batch_size}, {device}
  Timing: ~{time} (incl. ~{load_time} data load)
  Baseline: stick_mae={best}, btn_pressed={best}%

  Concerns:
    - {any flags}

Card written to: worldmodel/docs/run-cards/{name}.md
```

Then ask: "Ready to review, or want to adjust anything?"

**Do NOT launch the training.** The card must be reviewed first.
