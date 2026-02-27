# E008: Focal Context (Predict Inside the Window)

**Category:** training
**Status:** implemented (dataset), model variants below
**Target metric:** change_acc, on_ground recall

## Motivation

The model currently predicts frame t given context [t-K, ..., t-1] and ctrl(t). The context window ends right before the prediction target — the model never sees what happens next.

In a real replay, the future exists. If the model sees D frames of future state while learning to predict frame t, it learns temporal patterns invisible from one-sided context: how transitions resolve, what trajectories look like through a state change, which action sequences are natural continuations. This is how sequence models naturally learn — from full sequences, predicting points within them.

### Context window vs commitment windows

Empirical analysis of 50 games (963 hit interactions) shows the full "commitment window" — from when an attacker commits to a move through the victim exiting hitstun:

| Threshold | Frames | Time | % of interactions covered |
|-----------|--------|------|--------------------------|
| Current K=10 | 10 | 167ms | 14.6% |
| K=20 | 20 | 333ms | 28.5% |
| K=30 | 30 | 500ms | 39.5% |
| K=45 | 45 | 750ms | 58.9% |
| **K=60** | **60** | **1000ms** | **71.7%** |
| K=90 | 90 | 1500ms | 87.6% |
| K=120 | 120 | 2000ms | 93.0% |

Median commitment: 38 frames (633ms). P95 attack duration: 43 frames. P95 hitstun: 107 frames.

The model sees ~15% of commitment windows. Context length is a fundamental constraint.

## The Core Problem

With focal_offset=D, the context window extends D frames past the prediction target — all with full state+ctrl (no masking). But the Mamba SSM reads `x[:, -1, :]` (last frame's hidden state) for prediction. With future frames in the window, "last frame" is now a future frame, not the frame before the target.

**The model has no signal telling it which position to predict.**

Four variants address this differently. Each is a separate sub-experiment.

## Dataset Implementation (shared by all variants)

`focal_offset=D` in EncodingConfig. The dataset shifts the context window forward by D frames:

```
Baseline (D=0): context [t-K, ..., t-1], predict t
Focal (D=3):    context [t-K+D, ..., t-1, t, t+1, t+2], predict t
```

All K frames have full state+ctrl. The target and conditioning ctrl are unchanged.

**Files:** `encoding.py` (parameter), `dataset.py` (window shift, valid index bounds).

## Variants

### E008a: Tap hidden state at focal position
### E008b: Positional conditioning
### E008c: Multi-position prediction
### E008d: Bidirectional Mamba

See individual hypothesis files in this directory.
