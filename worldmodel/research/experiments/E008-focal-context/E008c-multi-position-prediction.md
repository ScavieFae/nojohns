# E008c: Multi-Position Prediction (Predict Every Frame)

**Variant of:** E008 (focal context)
**Category:** architecture + loss
**Complexity:** significant refactor (~100+ lines)
**Status:** planned

## Idea

Predict state at **every position** in the context window, not just one. At each timestep i, the model predicts frame i+1 from the hidden state at position i. Loss is computed across all positions.

This is how language models train — predict the next token at every position in the sequence. The model learns from K prediction signals per example instead of 1.

```
Context: [t-7, t-6, t-5, t-4, t-3, t-2, t-1, t, t+1, t+2]
Predict:        t-5   t-4   t-3   t-2   t-1    t   t+1  t+2  t+3
         (from hidden state at each position)
```

No focal_offset needed — the model naturally learns to predict at every position. The "future context" benefit comes from the SSM seeing later frames and learning representations that are consistent with the future, which helps earlier predictions through gradient flow.

## Why it might work

- **K× more training signal per example.** Each example produces K loss terms instead of 1. Faster learning per training step.
- **The model learns state transition dynamics across the full window**, not just at the boundary. It must learn to predict idle frames, attack onsets, hitstun, and recovery all within the same sequence.
- **No inference mismatch.** At rollout time, we still predict the next frame from the last position — same as baseline. No missing future context to worry about.
- **This is proven to work.** GPT-style next-token prediction at every position is the standard for sequence models.

## Why it might not work

- **Prediction heads currently read a single vector.** Refactoring them to produce (B, K, ...) outputs and computing loss across all positions is a significant change.
- **Loss weighting.** Not all positions are equally informative. Early frames in the context are always holds/idles. Should we weight later positions higher?
- **Memory.** K× more loss computation per batch. May need to reduce batch_size.

## Implementation

Major changes needed:
1. **Prediction heads**: change from `(B, d_model) → (B, pred_dim)` to `(B, K, d_model) → (B, K, pred_dim)`. Process full sequence, not just last hidden state.
2. **Targets**: `__getitem__` returns targets for all K+1 frames (state at positions 1 through K, predicted from hidden states 0 through K-1).
3. **Loss computation**: MetricsTracker computes loss across all positions. May want per-position loss weighting.
4. **Ctrl conditioning**: remove or restructure — currently injected after taking last hidden state, but now every position needs its own conditioning.

This is the cleanest conceptual approach but the biggest implementation lift.

## Test plan

1. Implement the refactor
2. Run baseline equivalent (K=10, predict at all positions, no focal offset) on 2K/2ep
3. Compare per-position accuracy: are early positions easier? Does accuracy degrade at later positions?
4. Compare headline metrics vs single-position baseline

## Relationship to focal_offset

Multi-position prediction makes focal_offset unnecessary. The model predicts everywhere. But we could still combine them: longer context (K=60) with multi-position prediction would give the model full commitment windows to learn from.
