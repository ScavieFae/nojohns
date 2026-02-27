# Run Card: mamba2-v3-2k-test

**Created**: 2026-02-26
**Config**: `worldmodel/experiments/mamba2-v3-2k-test.yaml`
**Status**: PENDING REVIEW

## Goal

Test whether parser v3 features (state_flags, hitstun, projectiles, real stage IDs) improve world model predictions vs the old encoding baseline.

## Target Metrics

| Metric | Baseline (source) | Target | Kill threshold |
|--------|-------------------|--------|---------------|
| val_change_acc | 67.3% (k10-compare, 2ep/2K) | >68% | <50% after epoch 1 |
| val_pos_mae | 0.65 (k10-compare) | <0.64 | >0.80 |
| val_loss/total | 0.291 (k10-compare) | <0.285 | not decreasing after 5% |

Note: Baseline is the closest apples-to-apples comparison (same architecture, same game count, 2 epochs). The data itself changed (GAME-only filter + v3 encoding), so this is not a pure A/B — it's "does v3 data + features work at least as well."

## Data

| Field | Value |
|-------|-------|
| Encoded file | `/encoded-game-v3-2k.pt` |
| File size | 13.8 GB |
| Games | 2,000 (GAME-only filtered from 24K local) |
| Total frames | 19,442,187 |
| Train examples | ~17,518,726 (1,800 games, 90%) |
| Val examples | ~1,923,461 (200 games, 10%) |

## Model

| Field | Value |
|-------|-------|
| Architecture | Mamba-2 (SSD) |
| Config | `worldmodel/experiments/mamba2-v3-2k-test.yaml` |
| Parameters | 4,347,748 |
| d_model | 384 |
| n_layers | 4 |
| d_state | 64 |
| context_len (K) | 10 |
| chunk_size (SSD) | 10 |

## Encoding (v3 — what's new)

| Feature | Old | New | Impact |
|---------|-----|-----|--------|
| state_flags | off | **on** (40 binary/player) | binary_dim 3→43, predicted_binary 6→86 |
| hitstun | off | **on** (1 continuous/player) | dynamics_dim +1 |
| projectiles | off | **on** (nearest item/player) | adds projectile floats |
| state_age_as_embed | off | **on** (learned embed) | moves state_age from float→int |
| stage | always 0 (bug) | **real IDs** (32=FD, etc.) | stage embed now meaningful |
| float_per_player | 29 | **72** | 2.5x wider input |
| int_per_player | 7 | **8** | +state_age embed |

## Training

| Field | Value |
|-------|-------|
| Epochs | 2 |
| Batch size | 4,096 |
| Learning rate | 0.0005 |
| Weight decay | 0.00001 |
| Optimizer | AdamW + cosine LR |
| Scheduled sampling | 0.3 (noise=0.1, anneal=3ep, corrupt=3 frames) |
| Loss weights | continuous=1.0, velocity=0.5, dynamics=0.5, binary=0.5, action=2.0, jumps=0.5, l_cancel=0.3, hurtbox=0.3, ground=0.3, last_attack=0.3 |

## Infrastructure

| Field | Value |
|-------|-------|
| GPU | A100-SXM4-40GB (Modal) |
| num_workers | 4 |
| Modal timeout | 14400s (4h) |
| wandb | `shinewave/melee-worldmodel` / `mamba2-v3-2k-test` |

## Logging & Monitoring

| Field | Value |
|-------|-------|
| Batches per epoch | ~4,277 (17.5M / 4096) |
| log_interval | 1,000 batches |
| Time between logs | ~60s |
| Logs per epoch | ~4 |
| wandb URL | https://wandb.ai/shinewave/melee-worldmodel |
| Modal dashboard | https://modal.com/apps/scaviefae/main |

## Timing & Cost

| Field | Value |
|-------|-------|
| Est. batch speed | ~60ms (A100, Mamba-2 4.3M, K=10) |
| Est. epoch time | ~4.3 min (4,277 batches × 60ms) |
| Est. total training | ~8.5 min |
| Est. data load | ~4 min (13.8 GB) |
| Est. total wall time | ~13 min |
| **Est. cost** | **~$0.60** |
| Timeout | 14400s = 4h (safety margin: ~18x) |

## Escape Hatches

- **Kill if**: loss explodes, OOM, or change_acc <50% after epoch 1
- **Resume command**: `.venv/bin/modal run worldmodel/scripts/modal_train.py::train --resume mamba2-v3-2k-test/best.pt --run-name mamba2-v3-2k-test-resumed --encoded-file /encoded-game-v3-2k.pt`
- **Fallback plan**: If v3 features hurt performance, re-encode with individual flags toggled to isolate which feature is the problem

## Prior Comparable Runs

| Run | Data | Epochs | change_acc | pos_mae | val_loss | Notes |
|-----|------|--------|------------|---------|----------|-------|
| k10-compare | 2K (old enc) | 2 | 67.3% | 0.65 | 0.291 | Closest baseline — same arch, same game count |
| projectile-2k-test | 2K (old + proj) | 2 | 63.6% | 0.65 | 0.309 | Projectiles only — no state_flags/hitstun |
| mamba2-10m-2k-smoke | 2K (old enc) | 2 | 68.9% | 0.61 | 0.280 | 10M param model, same data |
| mamba2-22k-ss | 22K (old enc) | 2 | 70.8% | 0.64 | 0.231 | 22K data, scheduled sampling |
| mamba2-22k-ss-resumed | 22K (old enc) | 4 | 76.7% | 0.51 | 0.204 | Best result to date (more data + epochs) |

## What's New in This Run

- **Parser v3 data**: state_flags (40 binary bits from 5 state_flags bytes), hitstun_remaining, real stage IDs, fixed invulnerability derivation
- **All feature flags enabled**: state_age_as_embed, projectiles, state_flags, hitstun
- **GAME-only filtered data**: Only games that ended with a proper GAME completion (no disconnects/NO_CONTEST). Higher quality training signal.
- **Binary head 14x wider**: 86 predictions vs 6. Loss weight kept at 0.5 — watching if state_flags dominate.
- **Input 2.5x wider**: 144 floats/frame vs 58. Model capacity unchanged (4.35M) — testing whether richer features help without scaling model.

## Launch Command

```bash
# Upload encoded data to Modal volume first:
.venv/bin/modal volume put melee-training-data \
  /Users/mattiefairchild/claude-projects/nojohns-training/data/encoded-game-v3-2k.pt \
  /encoded-game-v3-2k.pt

# Then launch:
.venv/bin/modal run --detach worldmodel/scripts/modal_train.py::train \
  --config worldmodel/experiments/mamba2-v3-2k-test.yaml \
  --encoded-file /encoded-game-v3-2k.pt \
  --epochs 2 \
  --run-name mamba2-v3-2k-test
```

## ScavieFae Review (Feb 26)

**APPROVE** — cheap test ($0.60, ~13 min), low risk, answers the right question.

### Notes

1. **Binary loss dominance — watch closely.** `predicted_binary_dim` goes 6→86 at weight 0.5. Binary loss contribution up ~14x. If the model over-invests in predicting state_flags bits at the expense of position/velocity, you'll see `pos_mae` regress while `change_acc` looks fine (state_flags are easy — most don't change frame-to-frame). **Watch in wandb:** `loss/binary` vs `loss/continuous`. If binary dominates total loss by >3x, drop binary weight to 0.1 for a follow-up.

2. **Projectiles included despite previous regression.** `projectile-2k-test` showed 63.6% change_acc (baseline 67.3%). That was on old data with empty items — this run has real items, so it's a different test. But if this run regresses, projectiles are the first flag to turn off to isolate.

3. **Config validation fix is solid.** Resolving both sides to full `EncodingConfig` dataclass before comparing eliminates false mismatch errors from YAML subsets.

### Verified

- `press_events: false` — correct (SSM learns temporal button transitions natively)
- SS dynamics mask confirmed (`trainer.py:224`) — only corrupts core_continuous + velocity, not stocks
- `pre_encode.py` saves full resolved config for reproducibility
- Timing/cost estimates reasonable

## Sign-off

- [ ] Scav reviewed
- [ ] Mattie reviewed
- [x] ScavieFae reviewed
