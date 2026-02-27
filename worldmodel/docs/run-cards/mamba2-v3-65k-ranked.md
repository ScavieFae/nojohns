# Run Card: mamba2-v3-65k-ranked

**Created**: 2026-02-27
**Config**: `worldmodel/experiments/mamba2-v3-100k-ranked.yaml`
**Status**: PENDING REVIEW

## Goal

First large-scale Mamba-2 run with v3 encoding on ranked data. Test whether 30x more data (65K vs 2K) pushes past the MLP ceiling (77.5%) toward 80%+ change_acc.

## Target Metrics

| Metric | Baseline (source) | Target | Kill threshold |
|--------|-------------------|--------|---------------|
| val_change_acc | 65.6% (v3-2k-test-v2, 2K/2ep) | >75% | <60% after epoch 1 |
| val_pos_mae | 0.667 (v3-2k-test-v2) | <0.50 | >0.80 |
| val_loss/total | 0.311 (v3-2k-test-v2) | <0.25 | not decreasing after 5% of epoch 1 |

MLP ceiling for reference: 77.5% change_acc at 22K/4ep (v2 encoding).

## Data

| Field | Value |
|-------|-------|
| Source | 116K ranked Slippi replays (anonymized) |
| Filtered | GAME-only (game_end_method=2): 114,784 games (98.9%) |
| Used | First 65,000 games |
| Encoded files | 6 super-chunks: `/encoded-v3-ranked-65k-part-{0..5}.pt` |
| Est. total frames | 755,885,000 |
| Train examples | ~680,296,500 (90%) |
| Val examples | ~75,588,500 (10%) |
| Float dims/frame | 138 (69/player) |
| Int dims/frame | 17 (8/player + 1 stage) |
| Est. tensor size | ~520 GB total across 6 parts |

## Model

| Field | Value |
|-------|-------|
| Architecture | Mamba-2 (SSD) |
| Config | `mamba2-v3-100k-ranked.yaml` |
| Parameters | 4,347,748 |
| d_model | 384 |
| n_layers | 4 |
| d_state | 64 |
| context_len (K) | 10 |
| chunk_size (SSD) | 10 |
| headdim | 64 |

## Training

| Field | Value |
|-------|-------|
| Epochs | 2 |
| Batch size | 4096 |
| Learning rate | 0.0005 |
| Weight decay | 0.00001 |
| Optimizer | AdamW + cosine LR |
| Scheduled sampling | 0.30 (noise=0.100, anneal=3 epochs, corrupt=3 frames) |
| Loss weights | continuous=1.0, velocity=0.5, dynamics=0.5, binary=0.5, action=2.0, jumps=0.5 |
| DDP | 2x GPU, DistributedDataParallel with NCCL |

## Encoding (v3, projectiles OFF)

| Field | Value |
|-------|-------|
| state_flags | ON (40 binary dims/player) |
| hitstun | ON (1 float dim/player) |
| projectiles | **OFF** (regressed 7% in v2 experiment) |
| state_age_as_embed | ON (8-dim embedding) |
| press_events | OFF |

## Infrastructure

| Field | Value |
|-------|-------|
| GPU | **2x NVIDIA H100** (DDP) |
| Encode | 6 super-chunks × ~10 CPU workers each, 128 GB concat RAM |
| num_workers | 4 |
| Modal timeout | **114600s (31.8h)** |
| wandb | `shinewave/melee-worldmodel` / `mamba2-v3-65k-ranked` |

## Logging & Monitoring

| Field | Value |
|-------|-------|
| Batches per epoch (total) | 166,088 |
| Batches per GPU per epoch | 83,044 |
| log_interval | 1,000 batches |
| Time between logs | ~230s (3.8 min) |
| Logs per epoch | ~83 |
| wandb URL | https://wandb.ai/shinewave/melee-worldmodel |
| Modal dashboard | https://modal.com/apps/scaviefae/main |

## Timing & Cost

| Field | Value |
|-------|-------|
| Est. batch speed (H100) | ~230ms/batch per GPU |
| Est. epoch time | ~5.3h (318 min) |
| Est. total training | ~10.6h |
| Est. encode time | ~3h (6 super-chunks) |
| Est. data load | ~15 min |
| Est. total wall time | **~14h** |
| **Est. cost** | **~$94** (training $84 + encode ~$10) |
| Timeout | 114,600s = 31.8h (3x safety margin) |

## Escape Hatches

- **Kill if**: loss not decreasing after 10K batches, NaN loss, change_acc < 60% after epoch 1
- **Resume command**: `.venv/bin/modal run worldmodel/scripts/modal_train.py::train --resume mamba2-v3-65k-ranked/latest.pt --epochs 3 --encoded-file "..." --run-name mamba2-v3-65k-ranked-resume`
- **Fallback plan**: If DDP fails, fall back to single H100 (change `gpu="H100:2"` → `gpu="H100"` in modal_train.py). 2x wall time but no code risk.

## Prior Comparable Runs

| Run | Data | Epochs | change_acc | pos_mae | val_loss | Notes |
|-----|------|--------|------------|---------|----------|-------|
| mamba2-v3-2k-test-v2 | 2K v3 | 2 | 65.6% | 0.667 | 0.311 | Same encoding, 32x less data |
| k10-compare (v2) | 2K v2 | 2 | 67.3% | 0.65 | 0.291 | v2 encoding, Mamba-2 baseline |
| MLP 22K (v2) | 22K v2 | 4 | 77.5% | - | - | MLP ceiling, flattening |

## What's New in This Run

- **First DDP run** — 2x H100, new DDP codepath in trainer.py + modal_train_worker.py
- **30x data scale** — 65K ranked games vs 2K unranked in prior v3 test
- **Ranked data** — higher skill level, more complete games (98.9% GAME-only)
- **Projectiles OFF** — removed after 7% regression in v2 experiment
- **Super-chunked encoding** — 6 parts × ~12K games, first use of the super-chunk pipeline
- **H100 GPUs** — first run on H100 (prior runs all A100)

## Risk Register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| DDP bugs (first use) | Medium | Fallback to single GPU, zero code changes |
| H100 availability on Modal | Low | Fallback to A100:2 or single H100 |
| Super-chunk encode failure | Low | 12K tar already on volume as backup |
| NaN from hitstun (seen before) | Low | nan_to_num in _encode_chunk |
| Data load OOM (6 × ~96 GB parts) | Medium | Sequential load + gc.collect() between parts |

## Launch Commands

```bash
# Encode (already launched):
.venv/bin/modal run worldmodel/scripts/modal_train.py::pre_encode_parallel \
  --config worldmodel/experiments/mamba2-v3-100k-ranked.yaml \
  --dataset ~/claude-projects/nojohns-training/data/ranked-v3-gameonly \
  --max-games 65000 \
  --tar-name ranked-v3-gameonly.tar \
  --super-chunk-size 12000 \
  --output /encoded-v3-ranked-65k \
  --chunk-size 2000

# Train (launch after encode completes):
.venv/bin/modal run --detach worldmodel/scripts/modal_train.py::train \
  --config worldmodel/experiments/mamba2-v3-100k-ranked.yaml \
  --encoded-file "/encoded-v3-ranked-65k-part-0.pt,/encoded-v3-ranked-65k-part-1.pt,/encoded-v3-ranked-65k-part-2.pt,/encoded-v3-ranked-65k-part-3.pt,/encoded-v3-ranked-65k-part-4.pt,/encoded-v3-ranked-65k-part-5.pt" \
  --epochs 2 \
  --run-name mamba2-v3-65k-ranked
```

## Sign-off

- [ ] Scav reviewed
- [ ] Mattie reviewed
- [ ] ScavieFae reviewed
