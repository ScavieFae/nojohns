# Run Card: mamba2-22k-ss

**Created**: Feb 26, 2026
**Config**: `worldmodel/experiments/mamba2-medium-gpu.yaml`
**Status**: PENDING REVIEW

## Goal

Validate Mamba-2 scaling on 11x more data (22K vs 2K games) with scheduled sampling enabled — first real overnight training bet on the SSM architecture.

## Target Metrics

| Metric | Baseline (source) | Target | Kill threshold |
|--------|-------------------|--------|---------------|
| val_change_acc | 68.4% (k60-compare, 2K/2ep) | >75% | <60% after epoch 1 |
| val_pos_mae | 0.632 (k60-compare) | <0.55 | >0.80 |
| val_loss/total | 0.271 (k60-compare) | <0.20 | not decreasing after 5% of epoch 1 |

Note: MLP on 22K/4ep hit 77.5% change_acc. Mamba-2 should match or beat that with fewer epochs given its sequence modeling advantage.

## Data

| Field | Value |
|-------|-------|
| Encoded file | `/encoded-22k.pt` |
| File size | 74.4 GB |
| Games | 22,000 |
| Total frames | 206,754,081 |
| Train examples | 185,648,963 (90%) |
| Val examples | 20,885,118 (10%) |

## Model

| Field | Value |
|-------|-------|
| Architecture | Mamba-2 (SSD scan) |
| Config | `worldmodel/experiments/mamba2-medium-gpu.yaml` |
| Parameters | ~4,300,000 |
| d_model | 384 |
| n_layers | 4 |
| d_state | 64 |
| context_len (K) | 10 |
| chunk_size (SSD) | 10 |

## Training

| Field | Value |
|-------|-------|
| Epochs | **3** (override — config default is 10) |
| Batch size | 1024 |
| Learning rate | 0.0005 |
| Weight decay | 0.00001 |
| Optimizer | AdamW + cosine LR |
| Scheduled sampling | 0.3 (noise=0.1, anneal=3ep, corrupt=3 frames) |
| SS dynamics mask | Yes — only corrupts core_continuous + velocity, NOT stocks/hitlag/combo |
| Loss weights | continuous=1.0, velocity=0.5, dynamics=0.5, binary=0.5, action=2.0, jumps=0.5, l_cancel=0.3, hurtbox=0.3, ground=0.3, last_attack=0.3 |

## Infrastructure

| Field | Value |
|-------|-------|
| GPU | A100-SXM4-40GB (Modal) |
| System RAM | 128 GB (Modal container) |
| num_workers | 4 |
| Modal timeout | **86400s (24 hours)** |
| wandb | `shinewave/melee-worldmodel` / `mamba2-22k-ss` |

## Logging & Monitoring

| Field | Value |
|-------|-------|
| Batches per epoch | 181,298 |
| log_interval | 1000 batches |
| Time between logs | ~60s (~1 min) |
| Logs per epoch | ~181 |
| wandb batch logging | Yes (loss at every log_interval) |
| wandb URL | https://wandb.ai/shinewave/melee-worldmodel |
| Modal dashboard | https://modal.com/apps/scaviefae/main |

## Timing & Cost

| Field | Value |
|-------|-------|
| Est. batch speed | ~60ms (0.06s) |
| Est. epoch time | ~3.0h (181 min) |
| Est. total training | ~9.1h |
| Est. data load | ~3 min (observed: 164s for 74.4 GB .pt) |
| Est. total wall time | ~9.1h |
| **Est. cost** | **~$25** |
| Timeout | 86400s = 24h (safety margin: 2.6x) |

**Calibration warning**: Mattie has noted our time estimates are consistently off by 3-5x. Pessimistic estimate: ~27-45h, ~$75-125. The 24h timeout covers a 2.6x overrun but NOT a 5x overrun. If epoch 1 takes >8h, consider killing and re-evaluating.

## Escape Hatches

- **Kill if**: val_loss not decreasing after ~9K batches (5% of epoch 1, ~9 min). Change_acc below 60% after full epoch 1. OOM. Loss explosion (NaN or >10).
- **Resume command**: `.venv/bin/modal run worldmodel/scripts/modal_train.py::train --resume mamba2-22k-ss --encoded-file /encoded-22k.pt --epochs 3 --run-name mamba2-22k-ss-resumed`
- **Fallback plan**: If SS is causing issues (loss divergence vs comparable non-SS runs), disable SS and relaunch as `mamba2-22k-noss`. If 22K data load fails, fall back to `encoded-2k.pt` with 10 epochs.

## Prior Comparable Runs

| Run | Data | Epochs | change_acc | pos_mae | val_loss | Notes |
|-----|------|--------|------------|---------|----------|-------|
| k60-compare | 2K | 2 | 68.4% | 0.632 | 0.271 | Best Mamba-2 result so far (K=60) |
| k10-compare | 2K | 2 | 67.3% | 0.650 | 0.291 | Same arch as this run (K=10) |
| smoke-nw4-v2 | 2K | 2 | 67.2% | 0.639 | 0.290 | First completed Mamba-2 run |
| MLP 22K (local) | 22K | 4 | 77.5% | 0.79 | — | MLP ceiling on same data |

## What's New in This Run

- **Scheduled sampling (SS)**: First-ever run with SS enabled. Rate=0.3, noise=0.1, annealing over 3 epochs. Dynamics mask applied (stocks/hitlag/combo excluded from corruption per ScavieFae review).
- **11x data scale**: 22K games vs 2K in all prior Mamba-2 runs. First Mamba-2 run on the full encoded dataset.
- **Combat context heads**: l_cancel, hurtbox_state, ground, last_attack prediction heads active (from branch `scav/combat-context-heads`).
- **Configurable log_interval**: 1000 batches (~60s between logs) instead of the old hardcoded `num_batches // 10` which gave 30+ min gaps.
- **Batch-level wandb logging**: Per-batch loss visible in wandb dashboard, not just per-epoch.
- **3 epochs (not 10)**: Conservative overnight bet — validate scaling before committing more compute.

## Launch Command

```bash
.venv/bin/modal run --detach worldmodel/scripts/modal_train.py::train \
  --encoded-file /encoded-22k.pt --epochs 3 --run-name mamba2-22k-ss
```

## Sign-off

- [ ] Scav reviewed
- [ ] Mattie reviewed
- [x] ScavieFae reviewed

---

## ScavieFae Review — Feb 25, 2026

**Verdict: BLOCKED — one fix required before launch**

### BLOCKER: Checkpoints not committed to Modal volume until training ends

`modal_train.py:211-214` — `volume.commit()` only runs after all 3 epochs complete. The trainer saves `best.pt` to the container's local filesystem, but if Modal kills the container for any reason (timeout, OOM, spot preemption), **every checkpoint from every completed epoch is lost**. At ~3h per epoch, this means waking up to zero artifacts after 6+ hours of A100 time.

**Fix:** Add `volume.commit()` after each epoch's checkpoint save — either in the trainer as a callback or in the Modal launcher wrapping `trainer.train()`. This is the difference between "run died at hour 8, resume from epoch 2" and "run died at hour 8, start over."

### Also flagging: two silent-failure risks for overnight runs

**wandb failure is swallowed.** `modal_train.py:160-170` catches wandb init exceptions, prints a warning, and continues with no monitoring. With `--detach`, you won't see the warning. Training runs but is invisible — no dashboard, no way to check progress.

**`--detach` hides startup crashes.** If the function dies on startup (import error, missing encoded file, config mismatch), the error goes to Modal logs, not your terminal. You could launch, go to sleep, and it died 30 seconds later. Consider: launch attached first, confirm wandb URL + first few batches in the log, then detach or just let it run.

### Other Concerns (non-blocking)

**1. Three new variables at once.** Data scale (2K→22K), scheduled sampling, and combat context heads all change simultaneously. If metrics move, attribution is ambiguous. Acceptable for an overnight bet — but if results are mixed, the next run should isolate SS vs no-SS on the same data/heads.

**2. Timing calibration.** At the pessimistic end (45h), the 24h timeout won't cover it. The "kill if epoch 1 >8h" heuristic is good — but who is actually monitoring overnight? Specify whether Mattie will have wandb/Modal open, or if the timeout is the only circuit breaker.

**3. SS dynamics mask — confirm in code.** Card says SS corrupts core_continuous + velocity but NOT stocks/hitlag/combo. This was my review feedback and it's the right design. But has the mask actually been wired up in `trainer.py`? Scav should confirm before launch.

**4. Resume `--epochs` semantics.** The resume command passes `--epochs 3` — does that mean 3 more or 3 total? If total, resuming from epoch 2 would only train 1 more. Clarify.

**5. Cost range.** $25–$125 is a 5x spread. Fine for a one-off, just noting we're accepting the upper end.

### Looks Good

- Kill thresholds are concrete and actionable
- log_interval at 1000 batches (~60s) — massive improvement over 30-min gaps
- Batch-level wandb logging — essential for overnight monitoring
- 3 epochs instead of 10 — right call for a first scaling bet
- SS params (rate=0.3, noise=0.1, anneal=3ep) are reasonable
- Dynamics mask excluding stocks/hitlag/combo — correct design

— ScavieFae
