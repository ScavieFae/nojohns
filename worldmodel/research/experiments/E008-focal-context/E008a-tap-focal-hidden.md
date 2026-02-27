# E008a: Tap Hidden State at Focal Position

**Variant of:** E008 (focal context)
**Category:** architecture
**Complexity:** ~5 lines changed in mamba2.py
**Status:** planned

## Idea

Instead of reading the last frame's hidden state (`x[:, -1, :]`), read the hidden state at the focal position (`x[:, K-D-1, :]`). The SSM still processes all K frames including the D future frames, but the prediction heads read from the temporal position just before the target.

```python
# Current:
h = self.final_norm(x[:, -1, :])

# E008a:
focal_idx = K - D - 1  # last frame before the prediction target
h = self.final_norm(x[:, focal_idx, :])
```

## Why it might work

The hidden state at position K-D-1 has seen all past frames but not the future frames. However, during backpropagation, gradients flow through the future frames too — the model learns representations that are *informed by* the future context even though the prediction reads from the focal position.

This is similar to how encoder-decoder models work: the full sequence is processed, but decoding reads from a specific position.

## Why it might not work

The future frames only help through gradient flow, not through the hidden state the heads see. The model might not get much benefit since the forward pass at the focal position is identical to baseline — the hidden state hasn't seen the future yet. The benefit would be purely from the training signal (better gradients because the SSM has to learn representations that are consistent with the future).

## Implementation

- Pass `focal_offset` from EncodingConfig into the model constructor
- Change one line in `forward()`: `x[:, -1, :]` → `x[:, -(D+1), :]`
- Everything else unchanged

## Test plan

Run with D=3 on 2K/2ep. Compare change_acc vs baseline. If no improvement, this variant is a null — future context needs to be in the hidden state, not just the gradients.
