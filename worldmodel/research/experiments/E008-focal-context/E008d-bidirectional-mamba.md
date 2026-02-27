# E008d: Bidirectional Mamba (See Past and Future at Every Position)

**Variant of:** E008 (focal context)
**Category:** architecture
**Complexity:** moderate (~50-80 lines, new model component)
**Status:** planned

## Idea

Add a **reverse Mamba pass** that processes the context window right-to-left. At each position, the hidden state combines information from both directions:
- Forward pass: everything that happened before this frame
- Backward pass: everything that happens after this frame

The combined hidden state at position K-1 (last frame before target) has seen the full trajectory — past and future — and can use both to predict state(t).

```
Forward:   [t-10] → [t-9] → ... → [t-1]  →  h_fwd
Backward:  [t+2]  → [t+1] → [t]  → [t-1]  →  h_bwd
Combined:  h = h_fwd + h_bwd (or concat, or gated)
```

## Why it might work

This is the most principled version of "model sees the full trajectory." Bidirectional models are standard in NLP (BERT, bidirectional LSTMs) for tasks where the full sequence is available at training time. Our training setup has the same property — we have the full replay.

The forward pass captures causal physics (momentum, position trajectory, action sequences). The backward pass captures consequences (where is this going? does the character land? does the hit connect?). Together they give the richest possible representation of "what's happening at this moment."

For on_ground specifically: the forward pass knows the character jumped; the backward pass knows they landed 5 frames later. The combined representation should nail the landing prediction.

## Why it might not work

- **Inference mismatch.** At rollout time, there is no future. The backward pass would need to process predictions, not real data. This is the same problem as training with teacher forcing — the model sees real data during training but its own outputs during inference. However, the forward-only pass still works exactly like baseline, so we could use only the forward pass at inference time (dropping the backward context).
- **Doubled compute.** Two Mamba passes instead of one. ~2× wall time per step.
- **The backward pass breaks causality.** The Mamba-2 architecture is designed for causal (left-to-right) processing. Running it backwards is conceptually clean but means the model could "cheat" by reading the target frame's state directly from the backward pass. We'd need to offset the backward pass to exclude the target frame.

## Implementation

1. Add `backward_layers` to the model (same architecture as forward, separate weights)
2. In forward(): reverse the sequence, run through backward layers, reverse output
3. Combine forward and backward hidden states: `h = h_fwd + self.combine_proj(h_bwd)` or `h = self.combine(torch.cat([h_fwd, h_bwd], dim=-1))`
4. **Critical**: backward pass must start from frame t+D (or t+1 at minimum), NOT including frame t. Otherwise the model can read the answer directly.
5. Prediction heads read the combined hidden state

Could also be implemented as a second set of Mamba layers that share the same frame embeddings but process in reverse.

## Test plan

1. Implement bidirectional model as a flag in mamba2.py
2. First test: D=3 on 2K/2ep, forward+backward combined
3. Ablation: compare forward-only inference vs forward+backward inference
4. If forward-only inference works well, the backward pass is a pure training-time improvement (no cost at inference)

## Relationship to other variants

This subsumes E008a (focal hidden state tap) — with bidirectional processing, every position already has full context. It partially overlaps with E008c (multi-position prediction) in that both could be combined: predict at every position with bidirectional context.

E008d is the most expressive but most complex variant. If E008a or E008b show promise with simpler changes, this may not be needed.
