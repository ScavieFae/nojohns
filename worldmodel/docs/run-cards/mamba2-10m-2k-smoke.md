# Run Card: mamba2-10m-2k-smoke

**Created**: Feb 26, 2026
**Status**: PENDING REVIEW

## Goal

Determine whether 10.6M params improve accuracy over 4.3M at equal data. This is the second point on the scaling curve — combined with the 4.3M results at 2K and 22K, it tells us whether the home run should scale params, data, or both.

## Scaling Context

We have one model size (4.3M) at two data scales:

| Model | Data | change_acc | Notes |
|-------|------|-----------|-------|
| Mamba-2 4.3M | 2K / 2ep | 67.7% | k10-compare baseline |
| Mamba-2 4.3M | 22K / 1ep | 75.9% | mamba2-22k-ss, epochs 2-3 running |
| **Mamba-2 10.6M** | **2K / 2ep** | **?** | **This run** |

This gives us the triangle {4.3M×2K, 4.3M×22K, 10.6M×2K}. From these three points we can estimate 10.6M×200K without spending $500 to find out.

**If 10.6M >> 4.3M at 2K:** Model capacity matters. Scale both params and data for the home run.
**If 10.6M ≈ 4.3M at 2K:** Params aren't the bottleneck at this data scale. Scale data first (cheaper).
**If 10.6M < 4.3M at 2K:** Overfitting on 2K — need more data before more params. Definitely scale data first.

## Model

| Field | Value | vs 4.3M baseline |
|-------|-------|-------------------|
| Architecture | Mamba-2 (SSD scan) | Same |
| Config | `mamba2-large-gpu.yaml` | New config |
| Parameters | **10,592,546** | **2.5×** |
| d_model | **512** | 384 → 512 |
| n_layers | **6** | 4 → 6 |
| d_state | 64 | Same |
| headdim | 64 | Same |
| chunk_size | 10 | Same |
| context_len (K) | 10 | Same |

Note: the config says "15M" but actual param count is 10.6M. Using the real number here.

## Data

| Field | Value |
|-------|-------|
| Source | `/encoded-2k.pt` on Modal volume (already exists) |
| Games | 2,000 |
| Encoding | `state_age_as_embed=True` (same as all recent runs) |

No re-encoding needed — uses the existing 2K pre-encoded file.

## Training

| Field | Value |
|-------|-------|
| Epochs | 2 |
| Batch size | 512 (from mamba2-large-gpu.yaml — larger model, smaller batch) |
| Learning rate | 0.0003 (lower than 4.3M's 0.0005 — standard for larger models) |
| Weight decay | 0.00001 |
| Optimizer | AdamW + cosine LR |
| Scheduled sampling | 0 (match baseline — no SS on k10-compare) |
| Device | A100 (Modal) |

### LR and batch size rationale

The large config uses lr=0.0003 (vs 0.0005 for medium). This is conservative — larger models are more sensitive to LR. Batch size 512 (vs 4096 for medium) keeps memory safe on a single A100. With 2K data this means more batches per epoch (~3.6K vs ~450 at bs=4096), so each epoch takes longer per frame but the data is tiny regardless.

## Timing & Cost

| Field | Value |
|-------|-------|
| Est. batches/epoch | ~3,600 (at bs=512) |
| Est. epoch time | ~15 min (2K data is tiny even at 10.6M) |
| Est. total training | ~30 min |
| Est. data load | ~5s (2K encoded is 7.4GB) |
| **Total wall time** | **~35 min** |
| **Cost** | **~$2** |

3x calibrated: ~$6. 5x calibrated: ~$10.

This is cheap. The risk is wasted A100 time if the run crashes on a config issue.

## Kill Thresholds

| Condition | Action |
|-----------|--------|
| OOM on A100 | Reduce batch_size to 256 in config |
| val_loss > 0.40 after epoch 1 | Possible LR issue — try 0.0001 |
| NaN loss | chunk_size issue with SSD — verify chunk_size=10 divides K=10 |
| change_acc < 55% after epoch 1 | Model not learning — check encoding config match |

## Baseline (k10-compare, 4.3M Mamba-2, 2K/2ep)

| Metric | Value |
|--------|-------|
| val_change_acc | 67.70% |
| val_pos_mae | 0.694 |
| val_loss | 0.291 |
| val_movement | 90.2% |

## Target Metrics

| Metric | Baseline (4.3M) | Target | Interpretation |
|--------|-----------------|--------|----------------|
| val_change_acc | 67.70% | >70% | Capacity helps — scale params for home run |
| val_pos_mae | 0.694 | <0.65 | Larger model captures finer physics |
| val_loss | 0.291 | <0.27 | General improvement |

**If change_acc ≈ 67-68%:** 2.5× params didn't help at 2K. Data is the bottleneck. Home run = 4.3M × 200K.
**If change_acc > 70%:** Capacity matters. Home run = 10.6M × 200K.

## Escape Hatches

- **Resume**: Not needed — run is <1h, cheaper to restart
- **Fallback**: If OOM, try batch_size=256. If config issues, verify `mamba2-large-gpu.yaml` encoding section matches `encoded-2k.pt`

## Launch Command

```bash
.venv/bin/modal run --detach worldmodel/scripts/modal_train.py::train \
    --config worldmodel/experiments/mamba2-large-gpu.yaml \
    --encoded-file /encoded-2k.pt \
    --epochs 2 \
    --run-name mamba2-10m-2k-smoke
```

## After Training

1. Pull val metrics from wandb
2. Compare change_acc to 67.70% baseline (the decision metric)
3. Update HOME-RUN-TRAINING.md Decision 1 with results
4. If 10.6M wins: plan 10.6M × 22K next (before committing to 200K)

## Sign-off

- [ ] Scav reviewed
- [ ] Mattie reviewed
- [x] ScavieFae reviewed

---

## ScavieFae Review — Feb 26, 2026

**Verdict: Approve.**

Clean experimental design. The scaling triangle {4.3M×2K, 4.3M×22K, 10.6M×2K} is the right way to decide before spending $500. Variable isolation correct (SS disabled to match baseline). $2-$10 cost, existing data, no prep needed.

Minor note: batch_size=512 vs baseline's 1024. LR/batch ratios are close (~5.9e-7 vs ~4.9e-7), shouldn't matter at this scale. No volume.commit() needed — "cheaper to restart" is honest for a 35-min run.

— ScavieFae
