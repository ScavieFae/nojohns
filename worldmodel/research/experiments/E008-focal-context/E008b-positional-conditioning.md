# E008b: Positional Conditioning (Tell the Model Where to Predict)

**Variant of:** E008 (focal context)
**Category:** architecture
**Complexity:** ~15-20 lines
**Status:** planned

## Idea

Add a signal to the ctrl conditioning that tells the model which position in the sequence is the prediction target. The model still reads `x[:, -1, :]` (the last hidden state, which has seen past + future), but the conditioning says "predict the state at position 7 of 10."

Options for the signal:
1. **Scalar**: `focal_position / K` — normalized position (0.7 for K=10, D=3)
2. **One-hot**: K-dim vector with a 1 at the focal position
3. **Learned embedding**: like positional encoding in transformers

## Why it might work

The last hidden state has seen the entire sequence. It knows what happened before and after the target frame. The positional signal tells it which point in the trajectory to decode. This is lightweight — one extra input feature — and the model can learn to use it or ignore it.

The `ctrl_proj` already adds conditioning to the hidden state: `h = h + self.ctrl_proj(next_ctrl)`. Adding a positional signal to `next_ctrl` is minimally invasive.

## Why it might not work

The Mamba SSM is causal — the hidden state at position -1 has processed frames left-to-right. It has a compressed summary of the whole sequence, but the compression is biased toward recent frames (typical of recurrent/SSM models). Asking it to "go back" to an earlier position via a conditioning signal might be asking too much of the representation.

## Implementation

- Append focal position scalar to `next_ctrl` tensor in `__getitem__` (1 extra dim)
- Update `ctrl_conditioning_dim` property to account for +1
- Or: separate `focal_embed` layer in the model, added to `h` alongside `ctrl_proj`
- Cleanest: add `focal_position` as a separate scalar in the forward signature

## Test plan

Start with the scalar approach (simplest). D=3 on 2K/2ep. If the model learns to use the signal (check by comparing D=3 performance with vs without the signal), try the learned embedding variant.
