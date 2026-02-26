---
name: check-training
description: Check on a running or recent Modal training run. Use when user says "check training", "how's the run", "training status", or similar. Safe read-only — will never disrupt a running job.
---

# Check Training Status

Safe, read-only check on Modal GPU training runs. Pulls metrics from wandb and status from Modal without connecting to or disrupting the running container.

**CRITICAL**: This skill is READ-ONLY. Never run `modal app logs` in streaming mode or `modal run` (non-detached) against a running app — that risks killing it on disconnect. Everything here uses the wandb API or `modal app list`, which are safe.

## Arguments

The user may provide:
- A wandb run name or ID (e.g., `mamba2-22k-ss-resumed`, `8za01pdm`)
- `latest` or nothing — checks the most recent run
- `all` — summary of last 5 runs
- `pull checkpoint` — download latest checkpoint from volume

If no arguments, check the most recent run.

## Steps

### 1. Modal App Status

```bash
.venv/bin/modal app list 2>&1 | head -15
```

Report: running/stopped, when it started, how long it's been up.

### 2. wandb Metrics

```python
import wandb, time
api = wandb.Api()

# Find the target run (by name, ID, or most recent)
runs = api.runs('shinewave/melee-worldmodel', order='-created_at')
run = runs[0]  # or filter by name/ID from user args

# Run state
print(f"Run: {run.name} ({run.state})")
print(f"URL: {run.url}")
runtime_h = run.summary.get("_runtime", 0) / 3600
print(f"Runtime: {runtime_h:.1f}h")

# Config summary
cfg = run.config
games = cfg.get("games", "?")
epochs = cfg.get("epochs", "?")
batch_size = cfg.get("training", {}).get("batch_size", cfg.get("batch_size", "?"))
arch = cfg.get("model", {}).get("arch", "?")
print(f"Config: {arch}, {games} games, {epochs} epochs, batch_size={batch_size}")

# Latest summary metrics (these are the most recent logged values)
s = run.summary
print(f"\n--- Current State ---")
print(f"Epoch: {s.get('epoch', '?')}")
print(f"Batch: {s.get('batch/step', '?')} ({s.get('batch/pct', 0):.1f}%)")
print(f"Batch loss: {s.get('batch/loss', '?')}")

# Validation metrics (from epoch boundaries)
history = list(run.scan_history(page_size=1000))
val_rows = [h for h in history if any(k.startswith('val_') for k in h.keys())]
if val_rows:
    print(f"\n--- Validation Results ({len(val_rows)} epoch(s) complete) ---")
    for row in val_rows:
        ep = row.get("epoch", "?")
        rt = row.get("_runtime", 0) / 3600
        print(f"  Epoch {ep} ({rt:.1f}h):")
        print(f"    val_change_acc: {row.get('val_metric/action_change_acc', '?'):.4f}")
        print(f"    val_pos_mae:    {row.get('val_metric/position_mae', '?'):.4f}")
        print(f"    val_loss:       {row.get('val_loss/total', '?'):.4f}")

# Per-action category accuracy (from summary, latest epoch)
categories = [(k, v) for k, v in sorted(s.items()) if k.startswith("val_action/cat_")]
if categories:
    print(f"\n--- Action Category Accuracy ---")
    for k, v in categories:
        cat = k.split("cat_")[1]
        flag = " <<<" if v < 0.93 else ""
        print(f"    {cat:20s} {v:.3f}{flag}")

# Staleness check
last_ts = history[-1].get("_timestamp", 0) if history else 0
mins_ago = (time.time() - last_ts) / 60 if last_ts else 0
if run.state == "running" and mins_ago > 10:
    print(f"\n⚠️  STALE: Last log was {mins_ago:.0f} minutes ago — run may have died")
elif run.state == "finished":
    print(f"\n✓ Run finished {mins_ago:.0f} minutes ago")
```

### 3. Wall Time Estimate

Calculate and report:
- Time per epoch so far (from validation row timestamps)
- Estimated completion time for remaining epochs
- Cost estimate: runtime_h × $2.78/hr for A100

### 4. Checkpoint Pull (if requested)

Only if user asks for "pull checkpoint" or "download checkpoint":

```python
import modal
vol = modal.Volume.from_name('melee-training-data')

# Find the run's checkpoint directory
run_name = "..."  # from the target run
entries = list(vol.listdir(f'/checkpoints/{run_name}'))
for e in entries:
    print(f"  {e.path}")

# Download best.pt or latest.pt
local_path = f'worldmodel/checkpoints/{run_name}-latest.pt'
with open(local_path, 'wb') as f:
    for chunk in vol.read_file(f'checkpoints/{run_name}/latest.pt'):
        f.write(chunk)
```

This is safe — Modal volumes support concurrent reads. Downloading does not affect the running training container.

## Report Format

Present results as a clean table:

```
## Training Status: {run_name}

**State**: running | **Runtime**: 4.2h | **Cost**: ~$11.68
**Config**: mamba2, 22K games, 3 epochs, batch_size=4096

### Progress
Epoch 2/3, batch 12000/45312 (26.5%)

### Validation (completed epochs)
| Metric          | Epoch 1 | Epoch 2 | Target  |
|-----------------|---------|---------|---------|
| val_change_acc  | 75.9%   | —       | >75%    |
| val_pos_mae     | 0.549   | —       | <0.55   |
| val_loss        | 0.231   | —       | <0.20   |

### Action Categories (latest)
| Category      | Accuracy | Flag |
| movement      | 91.0%    | <<<  |
| (others)      | 97%+     |      |

### Links
- wandb: {url}
- Modal: https://modal.com/apps/scaviefae/main
```

## What NOT To Do

- **NEVER** run `modal run` without `--detach` to check on something — that creates a new client connection
- **NEVER** stream logs via `modal app logs <id>` in a blocking way from a Claude session — if the session drops, it could signal the app to stop (only a risk for non-detached runs, but avoid the habit)
- **NEVER** `modal app stop` unless the user explicitly asks to kill a run
- The wandb API and `modal app list` are always safe — they don't connect to the running container

## Key Files

- **Launcher**: `worldmodel/scripts/modal_train.py`
- **Configs**: `worldmodel/experiments/*.yaml`
- **Run cards**: `worldmodel/docs/run-cards/` (targets and review history)
- **Runbook**: `worldmodel/RUNBOOK.md`
