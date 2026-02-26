---
name: gpu-train
description: Launch GPU training on Modal. Handles setup verification, data upload, training launch, and status checks. Use when user says "train on GPU", "launch training", "run on Modal", or similar.
---

# GPU Training via Modal

Launch world model training on a cloud A100 GPU. Handles the full pipeline from verification to launch.

## Arguments

The user may provide:
- A config path (e.g., `worldmodel/experiments/mamba2-medium-gpu.yaml`)
- Overrides: `--epochs N`, `--max-games N`, `--run-name "name"`
- `status` — just check on a running/recent run
- `setup` — first-time setup only

If no arguments, use the default config (`mamba2-medium-gpu.yaml`).

## Steps

### 1. Verify Modal Setup

```bash
# Check Modal is installed and authenticated
.venv/bin/modal volume list 2>&1
```

If this fails with "Token missing":
- Tell the user to run `.venv/bin/modal setup` (needs browser)
- Then come back and re-run this skill

If `melee-training-data` volume is not in the list:
```bash
.venv/bin/modal volume create melee-training-data
```

### 2. Verify Data on Volume

```bash
.venv/bin/modal run worldmodel/scripts/modal_train.py::upload_data
```

If it says "No data found", upload is needed:

```bash
# Check if tar exists locally
ls -lh /tmp/parsed-v2.tar 2>/dev/null

# If not, create it
cd ~/claude-projects/nojohns-training/data && tar cf /tmp/parsed-v2.tar parsed-v2/

# Upload (3.4GB, ~5 min)
.venv/bin/modal volume put melee-training-data /tmp/parsed-v2.tar /parsed-v2.tar
```

The train script auto-extracts the tar on first run.

### 3. Verify wandb Secret

```bash
.venv/bin/modal secret list 2>&1 | grep wandb-key
```

If not found, ask the user for their wandb API key and create it:
```bash
.venv/bin/modal secret create wandb-key WANDB_API_KEY=<key>
```

### 4. Launch Training

**CRITICAL: Always use `--detach` for any run longer than a smoke test.** Without it, if the local terminal disconnects (session timeout, laptop sleep, Claude session ends), Modal kills the container. We lost an 8-hour run to this exact failure. The only exception is a <5 minute smoke test where you want to watch stdout.

```bash
# ALWAYS use --detach for real training runs
.venv/bin/modal run --detach worldmodel/scripts/modal_train.py::train \
    --config <config-path> \
    --encoded-file /encoded-22k.pt \
    --epochs <N> \
    --run-name "<name>"
```

For resuming from a checkpoint:
```bash
.venv/bin/modal run --detach worldmodel/scripts/modal_train.py::train \
    --config <config-path> \
    --encoded-file /encoded-22k.pt \
    --epochs <N> \
    --run-name "<name>-resumed" \
    --resume <run-name>/latest.pt
```

**NEVER** use `modal run` without `--detach` and then walk away. **NEVER** background with `&` as a substitute for `--detach` — if the shell dies, so does the run.

### 5. Report

After launching, tell the user:
- **wandb URL**: https://wandb.ai/shinewave/melee-worldmodel (look for latest run)
- **Modal dashboard**: https://modal.com/apps/scaviefae/main
- **Cost estimate**: A100 at ~$2.78/hr, typical 10-epoch run = ~$3-6
- To monitor: use `/check-training` (safe, read-only, won't disrupt the run)

### Status Check (if user asked for status)

```bash
# Check if modal function is running
.venv/bin/modal app list 2>&1 | head -10

# Check wandb for latest run
.venv/bin/python -c "
import wandb
api = wandb.Api()
runs = api.runs('shinewave/melee-worldmodel', order='-created_at')
for run in list(runs)[:3]:
    print(f'{run.name}: {run.state} — {run.url}')
    if run.state == 'running':
        h = list(run.scan_history(keys=['epoch', 'loss/total', 'metric/action_change_acc']))
        if h:
            last = h[-1]
            print(f'  Epoch {last.get(\"epoch\",\"?\")}: loss={last.get(\"loss/total\",\"?\"):.4f} change_acc={last.get(\"metric/action_change_acc\",\"?\"):.3f}')
"
```

## Troubleshooting Quick Reference

| Error | Fix |
|-------|-----|
| Token missing | `.venv/bin/modal setup` |
| Volume not found | `.venv/bin/modal volume create melee-training-data` |
| Secret not found | `.venv/bin/modal secret create wandb-key WANDB_API_KEY=<key>` |
| No data on volume | Tar + upload (see step 2) |
| Import errors in container | Code broken locally — fix first, Modal bakes latest code on each run |
| CUDA OOM | Reduce `batch_size` in experiment YAML |
| `modal.Mount` not found | Modal API changed — use `Image.add_local_dir()` instead |
| `Secret.from_name(required=)` | Remove `required` kwarg — not supported in modal 1.3+ |

## Key Files

- **Launcher**: `worldmodel/scripts/modal_train.py`
- **Configs**: `worldmodel/experiments/*.yaml`
- **Full runbook**: `worldmodel/RUNBOOK.md` (section: "GPU Training via Modal")
