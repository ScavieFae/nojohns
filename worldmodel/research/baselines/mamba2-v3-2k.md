# Baseline: mamba2-v3-2k

**Checkpoint:** `worldmodel/checkpoints/mamba2-v3-2k-test-v2/best.pt`
**Architecture:** Mamba-2, 4.3M params (d_model=384, n_layers=4, K=10)
**Data:** 2K games, 2 epochs, A100
**Encoding:** v3 (state_age_as_embed, state_flags, hitstun, projectiles)

## Batch Eval (5 games, teacher-forced)

| Metric | Value |
|--------|-------|
| Action accuracy | 96.7% |
| Change accuracy | 68.3% |
| on_ground accuracy | 77.8% |
| on_ground precision | 99.1% |
| on_ground recall | 52.3% |
| facing accuracy | 76.1% |
| facing precision | 99.5% |
| facing recall | 50.9% |
| Avg position error | 1.02 |

## Category Breakdown

| Category | Accuracy | Change Acc |
|----------|----------|------------|
| idle | 97.2% | 81.9% |
| movement | 89.8% | 55.6% |
| aerial | 97.5% | 81.7% |
| aerial_attack | 96.6% | 53.7% |
| ground_attack | 98.7% | 73.5% |
| damage | 97.3% | 24.1% |
| shield_dodge | 96.9% | 78.3% |
| grab | 97.5% | 78.9% |
| edge | 96.2% | 56.9% |
| special | 98.3% | 82.1% |

## Binary Flags

| Flag | Precision | Recall | F1 |
|------|-----------|--------|----|
| on_ground | 99.1% | 52.3% | 68.5% |
| facing | 99.5% | 50.9% | 67.3% |
| invulnerable | 0.0% | 0.0% | 0.0% |

## Full Analysis

See [BATCH-EVAL-ANALYSIS.md](../BATCH-EVAL-ANALYSIS.md)
