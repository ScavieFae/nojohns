# Mamba-2 Architecture Swap — How It Works

Date: Feb 24, 2026
Context: Replacing the MLP trunk with Mamba-2 for temporal sequence modeling.

## Why K=10?

K=10 is our context window — the number of previous frames we show the model. It's a hyperparameter we chose, not something inherent to Mamba.

Melee runs at 60fps. 10 frames ≈ 167ms ≈ 1/6 of a second. That's enough to see the current animation state, recent movement trajectory, and recent inputs. Most Melee action transitions resolve within that window.

The Mamba point is: Mamba-2's fancy SSD algorithm is designed for language models processing 2048+ tokens in parallel. It splits sequences into chunks and does matrix multiplies within each chunk. With only 10 elements, chunking is overhead — you'd have chunks of size 2 or 5, and the bookkeeping costs more than just looping. A simple for-loop over 10 steps is faster.

K=10 isn't sacred. If we later discover that longer context helps (K=30 for half a second, K=60 for a full second), Mamba scales naturally — and at K=60+ the SSD algorithm starts paying off. The interface doesn't change, just the internals.

## What are the pieces?

### Current MLP — what each piece does:

```
frame_enc (B, K, ~182)
```
Each of the K=10 frames becomes a ~182-dimensional vector. That's: 4 position/damage floats + 3 binary flags + 13 controller values per player (×2 players), plus learned embeddings for categorical things (action state → 32-dim, character → 8-dim, jumps remaining → 4-dim, etc.). All concatenated.

```
→ flatten
```
Smash all 10 frames into one long vector: (B, 10×182) = (B, 1820). **This is where temporal structure dies.** The model now sees 1820 numbers with no indication that numbers 0-182 are "oldest frame" and 1639-1820 are "newest frame." It has to figure that out from scratch.

```
→ concat ctrl
```
Append the 26 controller floats for the frame we're predicting: (B, 1820+26) = (B, 1846).

```
→ Linear(1846, 512) → ReLU → Dropout → Linear(512, 256) → ReLU → Dropout
```
The MLP trunk. Two layers. First layer takes the giant flat vector and compresses it. Second layer compresses further.

```
→ heads
```
Prediction heads: Linear(256, 400) for action classification, Linear(256, 8) for position/damage regression, etc.

### New Mamba-2 — what changes:

```
frame_enc (B, K, ~182)  →  project(182, 256)
```
Same per-frame encoding. But instead of flattening, we project each frame independently to 256 dimensions. The sequence shape is preserved: (B, 10, 256).

```
→ 2× Mamba2Block
```
This is the core replacement. Two layers of Mamba-2 process the 10-frame sequence **in order**. Each frame updates a hidden state that carries forward. See "What happens inside a Mamba2Block" below.

```
→ last timestep
```
Take the output at position 10 (the final frame). This vector has "seen" all 10 frames through the recurrence — old frames are summarized in the hidden state, recent frames are freshest. Shape: (B, 256).

```
→ add ctrl_proj
```
Project the 26-dim controller input to 256-dim and **add** it to the Mamba output. (The MLP concatenated; Mamba adds. Addition works because they're already in the same 256-dim space.)

```
→ heads
```
Same prediction heads. Identical. The trainer, the loss function, the dataset — none of them know the trunk changed.

## What happens inside a Mamba2Block

The core is a recurrence — a state that evolves as it reads each frame:

```
For each frame t = 1, 2, ..., 10:
    h(t) = decay * h(t-1) + gate_in(t) * x(t)
    y(t) = gate_out(t) * h(t)
```

In actual notation:
```
h(t) = exp(A · dt(t)) · h(t-1)  +  dt(t) · B(t) ⊗ x(t)
y(t) = C(t) · h(t)  +  D · x(t)
```

Where:
- **h** is the hidden state — a matrix that accumulates game context (positions, momentum, action history). Shape: (nheads, headdim, d_state). Think of it as the model's "memory" of what it's seen.
- **A** is a learned decay rate (always negative). Controls how fast old information fades. Recent frames matter more than distant ones.
- **dt(t)** is a learned per-frame "timestep." This is the **selective** part — dt is computed from the input. During steady-state frames (same action repeating), dt is small: "nothing's changed, keep your memory, don't update much." During transition frames (new action starting), dt is large: "pay attention, incorporate this."
- **B(t)** is the input gate — decides which aspects of the current frame to write into the state. Also input-dependent.
- **C(t)** is the output gate — decides what to read out of the state. Also input-dependent.
- **D** is a skip connection — lets the raw input pass through directly.

The "selective" in "Selective State Space Model" means B, C, and dt are all functions of the current input. The model learns when to update, what to store, and what to read out. An LSTM has similar gating, but Mamba's formulation has mathematical properties that make it trainable as a parallel scan (for long sequences) while behaving like an RNN at inference.

### Concrete example for Melee

- Frame t-10: Fox starts a dash. Mamba writes position + velocity into state.
- Frames t-9 through t-5: Dash continues. Small dt's — state mostly coasts, slight position updates.
- Frame t-4: Dash stops, jump starts. Large dt — Mamba overwrites the "dashing" context with "jumping."
- Frame t-1: Jump apex. State carries full trajectory context.
- Output at t-1: Hidden state encodes "Fox is at apex of a jump that started from a dash at position X." Prediction heads use this + controller input to predict what happens.

The MLP can learn some of this, but it has to figure out the temporal ordering from 1820 unstructured numbers. Mamba gets it for free from the sequential structure.

## Why fewer parameters?

The MLP's bottleneck is its first layer:

```
Linear(1846, 512) = 1846 × 512 + 512 = 945,664 parameters
Linear(512, 256)  = 512 × 256 + 256  = 131,328 parameters
                                Total = ~1.08M
```

That first layer is enormous because the MLP takes the **entire flattened context** (1846 values) as input. Most of those parameters are learning positional patterns — "if the number at index 364 is high AND the number at index 546 is high, then..." That's wasteful.

Mamba's parameter breakdown:
```
frame_proj: Linear(182, 256) = 46,848          # per-frame, not per-sequence
Per Mamba2Block:
  in_proj:  Linear(256, ~1160) = ~297K          # projects 256, not 1846
  conv1d:   640 × 4 = 2,560                     # tiny local convolution
  out_proj: Linear(512, 256) = ~131K
  Per block total: ~400K
2 blocks: ~800K
                                    Total: ~850K
```

Mamba operates on 256-dim per frame, not 1846-dim per sequence. It shares weights across all 10 timesteps — the same Mamba2Block processes frame 1 and frame 10. The MLP can't share across positions because they're all mashed together.

Fewer params, but **structurally aware** of time. The capacity goes toward learning dynamics instead of learning "which index means what."

## Design decisions

**Number of layers (2 vs 3 vs 4).** More layers = more abstraction levels. 2 is conservative and matches the MLP's 2-layer trunk. We start there. If results are good but not great, adding a third layer is a one-line change.

**Last timestep vs pooling.** Taking only the final output means early frames can only influence the prediction through what the hidden state remembers. Pooling all 10 outputs gives every frame a direct vote. For a fighting game where recency dominates, last-timestep is the right default — but we could try both.

**Controller conditioning — add vs concatenate.** The MLP concatenates ctrl because it's just more numbers in a flat vector. With Mamba, we have a clean 256-dim representation, and the ctrl projects to the same space. Addition is cleaner (no dimension change for the heads) and acts as a conditional bias: "given this controller input, shift the prediction this way." If we wanted something heavier, we could concatenate and add a small mixing MLP after, but additive is the right first attempt.

**d_state (64).** This is the dimensionality of Mamba's hidden state memory. Mamba-1 was stuck at 16 because of computational constraints; Mamba-2's architecture lets us go to 64, 128, or 256. 64 is the sweet spot — enough to track the game state richly, not so much that it's slow.

**The conv1d (kernel=4).** Before the SSM recurrence, Mamba runs a small 1D convolution over the sequence. This captures very local patterns (2-3 adjacent frames). For us at 60fps, kernel=4 ≈ 67ms of local context. It's a lightweight way to detect things like "stick just moved" before the SSM processes the broader sequence. We keep this as-is.

## References

- [tommyip/mamba2-minimal](https://github.com/tommyip/mamba2-minimal) — MPS-compatible pure PyTorch Mamba-2
- [Goomba Lab: Mamba-2 Part I](https://goombalab.github.io/blog/2024/mamba2-part1-model/) — Architecture differences from Mamba-1
- [state-spaces/mamba](https://github.com/state-spaces/mamba) — Official implementation (CUDA-only, but algorithm reference)
- Exp results: see `~/.agent/diagrams/worldmodel-experiment-results.html`
