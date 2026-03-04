# Training

Training AI fighters (like Phillip) is handled in a separate project:

```
~/claude-projects/nojohns-training/
```

That project contains:
- **TRAINING-GUIDE.md** — Full walkthrough of the slippi-ai training pipeline
- **REPLAY-DATA.md** — Links to Vlad's curated replay datasets
- **scripts/** — Wrappers for download, parse, and train stages
- **data/** — Raw replays and parsed training data (gitignored)

## Why Separate?

Training has different dependencies (TensorFlow, slippi-ai) and produces large artifacts (replays, checkpoints, models) that don't belong in the main No Johns repo. The separation keeps the core CLI/arena/contracts repo lean.

## World Model

We're building a structured-state world model that predicts next-frame game state from current state + player inputs. This is genuinely novel — all existing Melee AI (Phillip/slippi-ai) is model-free.

Code lives in `worldmodel/` within the main repo. Training data and checkpoints live in `nojohns-training/`.

### Key Results (Feb 23, 2026)

**v2.2 input-conditioned prediction** — a single architectural insight that nearly doubled our hardest metric:

| | v2.1 (blind) | v2.2 (input-conditioned) |
|---|---|---|
| Action-change accuracy | 32.6% | **62.4%** |
| Val loss | 0.899 | **0.299** |

The insight: Melee's game loop records controller input and resulting state on the *same frame*. If the model knows what buttons were pressed, predicting the resulting state becomes a pure physics problem — deterministic game mechanics, not mind-reading.

Same data, same machine, same hyperparams. The only change: feed the model the current frame's controller input (26 floats) alongside the context window.

### Architecture Progression

1. **v1**: Basic encoding (56 dims/player). Action-change: 29.8%
2. **v2**: Added velocity, dynamics, character/stage embeds (72 dims). Action-change: 40.2%
3. **v2.1**: Added combat context — l-cancel, hurtbox, ground, last attack (89 dims). Action-change: 36.2%
4. **v2.2**: Input-conditioned prediction (+26 ctrl floats). Action-change: **62.4%**

### Imitation Policy ("New Phillip")

Alongside the world model, we're training an imitation learning policy that predicts controller outputs from game state — essentially learning "what would a human press here?" This is the decision-making half that the v2.2 world model deliberately separated out.

The two models compose: policy generates inputs → world model simulates outcomes. This is the standard architecture for model-based RL (MuZero, Dreamer).

### The Big Implication: No ISO Required

Right now, developing a Melee fighter requires a Melee ISO (legally gray — we can't distribute it), Dolphin, Slippi, and libmelee. That's the single biggest friction point for onboarding new agent operators.

If the world model is accurate enough for multi-step rollouts, none of that is needed. An agent developer just needs a PyTorch checkpoint — which we *can* distribute. The development loop becomes:

```
pip install nojohns → load world model → policy proposes inputs → world model returns next state → repeat
```

No emulator, no ROM, no frame timing. Training at thousands of frames per second instead of 60fps real-time. The v2.2 architecture is already shaped for this — it's literally a simulator that takes controller input and returns game state. The open question is whether accuracy holds over hundreds of rollout steps without drift.

This could turn "download an ISO and configure an emulator" into `nojohns train --world-model`.

### Documentation

- `worldmodel/RUNBOOK.md` — Full operational guide, all results, training commands
- `worldmodel/docs/INPUT-CONDITIONING.md` — The v2.2 insight, architecture change, and results
- `~/.agent/diagrams/input-conditioning-explainer.html` — Visual explainer page
- Wandb: [shinewave/melee-worldmodel](https://wandb.ai/shinewave/melee-worldmodel)

## Quick Links

- [slippi-ai repo](https://github.com/vladfi1/slippi-ai)
- [Phillip weights (via slippi-ai releases)](https://github.com/vladfi1/slippi-ai/releases)
- [Slippi replay database](https://slippi-ranked-db.vercel.app/)
