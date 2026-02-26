# Run Card: projectile-2k-test

**Created**: Feb 26, 2026
**Status**: PENDING REVIEW

## Goal

Test whether projectile encoding improves world model accuracy. Projectiles (Fox lasers, Samus charge shots, etc.) are currently invisible to the model — damage appears from nowhere. This is the #1 known encoding gap.

If projectiles help, we include them when re-encoding for the big run. If not, we skip the re-encode cost.

## What Changes

| Feature | Baseline (no projectiles) | This run |
|---------|--------------------------|----------|
| `EncodingConfig.projectiles` | `False` | **`True`** |
| `float_per_player` | 28 | **31** (+3: nearest_dx, nearest_dy, n_active) |
| Total float dims/frame | 56 | **62** (+6) |

Everything else identical to the k10-compare baseline.

## Data

| Field | Value |
|-------|-------|
| Source | Re-encode 2K games from `parsed-v2.tar` on Modal volume |
| Output | `/encoded-2k-proj.pt` on volume |
| Games | 2,000 |
| Encoding | `EncodingConfig(state_age_as_embed=True, projectiles=True)` |

### Re-encode required

The existing `encoded-2k.pt` was encoded without projectiles. Need a fresh encode with `projectiles=True`. The projectile data is in the parsed parquet files (`items` field) — no re-parsing needed.

Re-encode cost: ~$0.50 CPU, ~5 min.

## Model

| Field | Value |
|-------|-------|
| Architecture | Mamba-2 (SSD scan) |
| Config | `mamba2-medium-gpu.yaml` with encoding override |
| Parameters | ~4.3M (slightly more due to +6 input dims) |
| d_model | 384 |
| n_layers | 4 |
| context_len (K) | 10 |

## Training

| Field | Value |
|-------|-------|
| Epochs | 2 |
| Batch size | 4096 |
| Learning rate | 0.0005 |
| Optimizer | AdamW + cosine LR |
| Scheduled sampling | 0 (isolate the projectile variable) |
| Device | A100 (Modal) |

**No scheduled sampling.** The baseline (k10-compare) didn't use SS. Isolate the projectile variable.

## Timing & Cost

| Field | Value |
|-------|-------|
| Re-encode | ~5 min, ~$0.50 |
| Training (2 ep) | ~1.5h (46 min/ep observed on 2K) |
| **Total cost** | **~$5** |

## Baseline (k10-compare, no projectiles)

| Metric | Value |
|--------|-------|
| val_change_acc | 67.70% |
| val_pos_mae | 0.694 |
| val_loss | 0.291 |
| val_movement | 90.2% |

## Target Metrics

| Metric | Baseline | Hope | Signal |
|--------|----------|------|--------|
| val_pos_mae | 0.694 | <0.65 | Projectile dodge distance should improve position prediction |
| val_change_acc | 67.70% | >68% | Modest — projectiles affect a subset of frames |
| val_movement | 90.2% | >92% | Long shot: projectile avoidance might unstick movement ceiling |

**The key metric is pos_mae.** Projectiles cause position changes (dodging, knockback from unseen hits) that the model currently can't explain. If pos_mae doesn't improve, projectiles aren't worth the encoding cost.

## Kill Thresholds

| Condition | Action |
|-----------|--------|
| val_loss > 0.35 after epoch 1 | Encoding bug — investigate |
| val_change_acc < 60% after epoch 1 | Encoding bug — investigate |
| Metrics identical to baseline (within noise) | Projectiles don't help at 2K scale — may still help at 200K |

## Implementation

### 1. Re-encode with projectiles

Option A — modify `pre_encode` to accept encoding overrides:
```bash
.venv/bin/modal run worldmodel/scripts/modal_train.py::pre_encode \
    --max-games 2000 --output /encoded-2k-proj.pt \
    --encoding-override '{"projectiles": true, "state_age_as_embed": true}'
```

Option B — local encode (slower but simpler):
```bash
.venv/bin/python -m worldmodel.scripts.pre_encode \
    --dataset ~/claude-projects/nojohns-training/data/parsed-v2 \
    --max-games 2000 --output /tmp/encoded-2k-proj.pt \
    --projectiles
```

### 2. Train

```bash
.venv/bin/modal run --detach worldmodel/scripts/modal_train.py::train \
    --encoded-file /encoded-2k-proj.pt --epochs 2 \
    --run-name projectile-2k-test
```

### 3. Compare

Pull val metrics from wandb, compare to k10-compare baseline. Focus on pos_mae and movement category.

## Launch Command

```bash
# Step 1: Re-encode
.venv/bin/modal run worldmodel/scripts/modal_train.py::pre_encode \
    --max-games 2000 --output /encoded-2k-proj.pt

# Step 2: Train (detached)
.venv/bin/modal run --detach worldmodel/scripts/modal_train.py::train \
    --encoded-file /encoded-2k-proj.pt --epochs 2 \
    --run-name projectile-2k-test
```

## After Training

1. Compare pos_mae to baseline — this is the decision metric
2. If pos_mae improves >5%: include projectiles in all future encodes
3. If no improvement: skip projectiles, save encoding complexity
4. Either way, update HOME-RUN-TRAINING.md with results

## Sign-off

- [ ] Scav reviewed
- [ ] Mattie reviewed
