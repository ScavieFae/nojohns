# Input-Conditioned World Model (v2.2)

## The Insight

**Date:** February 23, 2026
**Context:** Training a world model for Super Smash Bros Melee — predicting next-frame game state from current state and player inputs.

We were reviewing action-change accuracy (40.2% on v2.1) when Mattie asked a deceptively simple question:

> "If we have inputs, the game SHOULD be looking at action change, because the change would happen after the input, right? Since the game itself is 'seeing' the input on that frame... That seems really provable?"

This exposed a fundamental architectural flaw. Our world model was being asked to do two jobs at once:

1. **Guess what the player will press** (a human decision — unpredictable)
2. **Simulate what happens** given that press (game mechanics — deterministic)

The model never saw the current frame's controller input when making predictions. It only had the previous 10 frames of history. So to predict "player transitions from WAIT to ATTACK," it had to *first* figure out that the player would press A, *then* figure out that pressing A during WAIT causes ATTACK. Two problems, one model, muddled together.

## The Frame Timing Question

In Melee's game loop (60fps):
1. Controller input is polled
2. Game logic processes input against current state
3. New state is recorded

**Input and resulting action change are on the same frame.** Press A on frame 100 → action state changes to ATTACK on frame 100. They're "simultaneous" in the data, but causally linked: input → state transition.

This means if the model *knows* the input, predicting the resulting action state becomes a game mechanics problem — learning Melee's state transition table. Not a mind-reading problem.

## The Architecture Change

### Before (v2.1): Blind prediction
```
Context: frames [t-10, ..., t-1]  (state + controller for each)
                    ↓
                  Model
                    ↓
Predict: frame t+1's state  (without seeing t+1's input)
```
The model had to guess player decisions AND simulate physics. Action-change accuracy: ~40%.

### After (v2.2): Input-conditioned prediction
```
Context: frames [t-10, ..., t-1]  +  frame t's controller input
                    ↓
                  Model
                    ↓
Predict: frame t's state  (knowing what buttons were pressed)
```
The model only needs to simulate physics. Action-change accuracy: **62.4%** after 2 epochs (up from 32.6% on v2.1 — same data, same machine, same hyperparams).

### What Changed in Code

**4 files modified, 0 new files:**

- **`dataset.py`** — `__getitem__` returns 5 tensors (was 4): added `next_ctrl` (26 floats — both players' controller inputs for the target frame). Target shifted from frame t+1 to frame t. Delta computed as frame t minus frame t-1.

- **`mlp.py`** — Trunk input dimension: 1846 (was 1820). The 26 controller floats are concatenated with the flattened context window before the shared trunk. Parameters: 1,304,182 (was 1,290,870).

- **`trainer.py`** — Training loop unpacks 5 tensors, passes `next_ctrl` to model.

- **`rollout.py`** — Controller input fed to model as a conditioning input (was injected post-prediction).

**Parsed replay data is unchanged.** Same parquet files, same encoding. The change is purely in how the model receives and processes the data.

## Why This Matters

### For the world model
- Action-change accuracy becomes a **pure physics metric**. Low = model doesn't understand Melee mechanics. High = it does. Clean signal, no noise from decision prediction.
- This is how world models are *supposed* to work: (state, action) → next_state. OpenAI's game models, MuZero, Dreamer — they all condition on the action being taken.

### For the agent pipeline
- The world model becomes a proper **simulator**: feed it any controller input and it tells you what happens. Essential for the RL training loop.
- A separate **policy network** handles the decision part: given game state, what buttons to press? This is imitation learning on the same parsed replays — supervised learning, much simpler.
- The two models compose: policy generates inputs → world model simulates outcomes → RL optimizes the policy. This is the same architecture as Phillip/slippi-ai, just with our world model replacing Dolphin.

### For understanding
- Separating physics from decisions is both architecturally cleaner and scientifically cleaner. We can measure each capability independently.
- The world model doesn't need to learn that "good players dash-dance" — it just needs to learn that "pressing left while running right causes a turnaround."

## Test Results

### Pipeline validation (5 games, CPU)
```
Forward pass:  OK  (1,304,182 params, input_dim=1846)
Dataset:       OK  (5 tensors: float_ctx, int_ctx, next_ctrl, float_tgt, int_tgt)
Training loop: OK  (loss 13.35 → 11.56 over 5 batches)
Backward pass: OK  (gradients flow correctly)
```

### Controlled comparison — v2.1 vs v2.2

Same data (2K games), same machine (ScavieFae M3 Pro), same hyperparams, one variable changed:

| Metric | v2.1 (2 epochs) | v2.2 (2 epochs) | Change |
|--------|-----------------|-----------------|--------|
| **action_change_acc** | 32.6% | **62.4%** | **+91% relative** |
| action_acc (overall) | 93.0% | 96.1% | +3.1pp |
| val_loss | 0.899 | **0.299** | **-67%** |
| train_loss | 1.03 | 0.349 | -66% |
| pos_mae | 0.79 | 0.79 | unchanged |

Action-change accuracy nearly **doubled**. Val loss dropped by two-thirds. Position MAE stayed the same. This is the cleanest possible signal: input conditioning works exactly as predicted.

Additional data point — Scav's v2.1 run on 4K games (2 epochs) reached 36.2% action-change accuracy. v2.2 on *half* the data after *one* epoch (45.8%) already surpassed it.

### Overnight runs (launched Feb 23, 2026)

| Machine | Run | Data | Epochs | Status |
|---------|-----|------|--------|--------|
| Scav (M3 Max, 36GB) | v2.2 world model | 22K games | 10 | Running |
| ScavieFae (M3 Pro, 18GB) | imitation policy | 4K games | 50 | Running |

Both logging to wandb project `melee-worldmodel`.

## Implication: No ISO Required

The v2.2 architecture isn't just cleaner research — it has a practical consequence for No Johns adoption.

Currently, developing a Melee fighter requires obtaining a Melee ISO (legally gray, can't be distributed), installing Dolphin, configuring Slippi and libmelee. That's the single biggest onboarding friction point.

An accurate input-conditioned world model is a **distributable simulator**. Feed it controller inputs, get back game state. No emulator, no ROM, no frame timing. PyTorch checkpoints are just files — we can ship them with `pip install nojohns`. Agent developers could train at thousands of frames per second instead of 60fps real-time.

The open question is rollout accuracy — whether the model stays coherent over hundreds of autoregressive steps without drift. But we have two advantages working in our favor:

1. **Agents don't care about visual fidelity.** A few pixels of positional drift or a frame-off shield flicker doesn't matter — agents need the game *mechanics* to be right (shielding blocks, grabs lead to throws, combos connect). The accuracy bar is "learn good strategies," not "fool a human viewer."

2. **We're already rendering from state, not pixels.** The nojohns.gg match visualizer displays from structured state data (positions, action states, stocks) — the same format the world model outputs. A simulated match would look identical to a real one on the website as long as the state is plausible. No Dolphin rendering needed on either end.

## The Punchline

We were asking our model to predict the future AND read minds simultaneously. When Mattie noticed the timing — that inputs and their effects live on the same frame — the fix was obvious: just tell the model what buttons were pressed. The rest is physics.

---

*This document captures the development of the input-conditioned world model architecture. The insight emerged from a conversation about Melee's frame timing and how it relates to what the model is actually being asked to learn.*
