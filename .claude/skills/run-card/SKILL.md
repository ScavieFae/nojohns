---
name: run-card
description: Generate a Run Card for a training run. Use before any significant GPU training run (>$5 or >2 hours). The card is reviewed by Mattie, Scav, and ScavieFae before launch. Use when the user says "run card", "create a card", "prepare a run", or when you're about to launch a training run that qualifies.
---

# Run Card Generator

Generate a pre-flight Run Card before any significant training run. The card is written to `worldmodel/docs/run-cards/{run_name}.md`, presented to the user for review, and must be approved before launch.

**Rule: Never launch a significant run without a reviewed card.**

## What the User Provides

- Config path (default: `worldmodel/experiments/mamba2-medium-gpu.yaml`)
- Run name
- Epoch count, encoded file, or other overrides
- Context: "for the overnight", "quick test of SS", etc.

## What You Do

### 1. Gather Facts

Read these files to populate the card:

- **Config YAML**: `worldmodel/experiments/{name}.yaml` — all model/training/encoding params
- **Runbook results**: `worldmodel/RUNBOOK.md` — prior run results for baseline comparison
- **Handoff doc**: `worldmodel/docs/HANDOFF-MODAL.md` — recent changes, review status

If an encoded file is specified, use these known stats:
- `/encoded-22k.pt`: 22,000 games, 206,754,081 frames, 74.4 GB, train=185,648,963, val=20,885,118
- `/encoded-2k.pt`: ~2,000 games, ~18M frames, ~6.6 GB

For unknown encoded files, estimate: ~9,400 frames/game average.

Check wandb for the most recent comparable runs:
```bash
.venv/bin/python -c "
import wandb; api = wandb.Api()
for run in list(api.runs('shinewave/melee-worldmodel', order='-created_at'))[:5]:
    s = run.summary
    ca = s.get('metric/action_change_acc', s.get('val_metric/action_change_acc', '?'))
    pm = s.get('metric/position_mae', s.get('val_metric/position_mae', '?'))
    vl = s.get('val_loss/total', '?')
    print(f'{run.name}: {run.state} | change_acc={ca} pos_mae={pm} val_loss={vl}')
"
```

### 2. Calculate Derived Values

Use these constants (observed Feb 25-26, 2026):

| Hardware | Batch speed | Cost/hr |
|----------|------------|---------|
| A100-SXM4-40GB (Modal) | ~0.06s/batch (Mamba-2 4.3M, K=10, bs=1024) | $2.78 |
| MPS M3 Max | ~0.4s/batch | free |
| CPU | ~7s/batch | free |

Calculate:
- `batches_per_epoch = train_examples // batch_size`
- `epoch_time_h = batches_per_epoch * batch_speed / 3600`
- `total_time_h = epoch_time_h * epochs`
- `cost = total_time_h * cost_per_hr`
- `log_interval` from config (or recommend one if missing)
- `time_between_logs = log_interval * batch_speed`
- `timeout`: estimated total + data load time (~3 min per GB), then **3x safety margin**, minimum 4 hours

**Flag if:**
- `time_between_logs > 300s` (5 min) — recommend `log_interval: 1000` in YAML
- `timeout < 2 * estimated_total` — too tight
- `cost > $50` — confirm intentional
- `batches_per_epoch > 100K` — large epoch, verify logging
- Any new/untested feature enabled (check SS, projectiles, etc.)

### 3. Write the Card

Save to `worldmodel/docs/run-cards/{run_name}.md`:

```markdown
# Run Card: {run_name}

**Created**: {date}
**Config**: `{config_path}`
**Status**: PENDING REVIEW

## Goal

{One sentence: what are we trying to learn from this run?}

## Target Metrics

| Metric | Baseline (source) | Target | Kill threshold |
|--------|-------------------|--------|---------------|
| val_change_acc | {best}% ({run}) | >{target}% | <{kill}% after epoch 1 |
| val_pos_mae | {best} ({run}) | <{target} | >{kill} |
| val_loss/total | {best} ({run}) | <{target} | not decreasing after 5% |

## Data

| Field | Value |
|-------|-------|
| Encoded file | {path on volume} |
| File size | {GB} |
| Games | {n} |
| Total frames | {n:,} |
| Train examples | {n:,} ({split}%) |
| Val examples | {n:,} ({split}%) |

## Model

| Field | Value |
|-------|-------|
| Architecture | {Mamba-2 / MLP} |
| Config | `{path}` |
| Parameters | {n:,} |
| d_model | {n} |
| n_layers | {n} |
| d_state | {n} |
| context_len (K) | {n} |
| chunk_size (SSD) | {n} |

## Training

| Field | Value |
|-------|-------|
| Epochs | {n} |
| Batch size | {n} |
| Learning rate | {lr} |
| Weight decay | {wd} |
| Optimizer | AdamW + cosine LR |
| Scheduled sampling | {rate} (noise={scale}, anneal={n}ep, corrupt={n} frames) |
| Loss weights | continuous={}, velocity={}, ... |

## Infrastructure

| Field | Value |
|-------|-------|
| GPU | {type} |
| System RAM | {GB} |
| num_workers | {n} |
| Modal timeout | **{seconds}s ({human_readable})** |
| wandb | `shinewave/melee-worldmodel` / `{run_name}` |

## Logging & Monitoring

| Field | Value |
|-------|-------|
| Batches per epoch | {n:,} |
| log_interval | {n} batches |
| Time between logs | ~{n}s ({human}) |
| Logs per epoch | ~{n} |
| wandb batch logging | Yes (loss at every log_interval) |
| wandb URL | https://wandb.ai/shinewave/melee-worldmodel |
| Modal dashboard | https://modal.com/apps/scaviefae/main |

## Timing & Cost

| Field | Value |
|-------|-------|
| Est. batch speed | ~{ms}ms |
| Est. epoch time | ~{h}h ({m} min) |
| Est. total training | ~{h}h |
| Est. data load | ~{m} min |
| Est. total wall time | ~{h}h |
| **Est. cost** | **~${cost}** |
| Timeout | {seconds}s = {hours}h (safety margin: {x}x) |

## Escape Hatches

- **Kill if**: {conditions — loss not decreasing, explosion, OOM}
- **Resume command**: `.venv/bin/modal run worldmodel/scripts/modal_train.py::train --resume {path} ...`
- **Fallback plan**: {what to do if this fails}

## Prior Comparable Runs

| Run | Data | Epochs | change_acc | pos_mae | val_loss | Notes |
|-----|------|--------|------------|---------|----------|-------|
| {name} | {games} | {ep} | {%} | {mae} | {loss} | {what's different} |

## What's New in This Run

{Bullet list of what's different from the most comparable prior run. New features, different data, config changes, etc.}

## Launch Command

```bash
.venv/bin/modal run --detach worldmodel/scripts/modal_train.py::train \
  --encoded-file {file} --epochs {n} --run-name {name}
```

## Sign-off

- [ ] Scav reviewed
- [ ] Mattie reviewed
- [ ] ScavieFae reviewed (via HANDOFF-MODAL.md)
```

### 4. Present to User

After writing the card, print a compact summary:

```
Run Card: {name}
  Goal: {one line}
  Data: {games} games, {frames} frames
  Model: {arch} {params} params
  Training: {epochs} epochs, bs={batch_size}, lr={lr}
  Timing: ~{hours}h, ~${cost}
  Logging: every {interval} batches (~{seconds}s)
  Timeout: {hours}h ({margin}x safety)

Concerns:
  - {any flags from step 2}

Card written to: worldmodel/docs/run-cards/{name}.md
```

Then ask: "Ready to review, or want to adjust anything?"

**Do NOT launch the run.** The card must be reviewed first.
